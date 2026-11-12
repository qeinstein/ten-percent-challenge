"""Real-text inputs and masked comparison helpers for package parity checks."""
from __future__ import annotations

import torch
import torch.nn.functional as F

PARITY_PARAGRAPHS = (
    "The river rose slowly through the spring as snow melted in the high valleys. "
    "Farmers along the banks moved their cattle to higher pastures, while the village "
    "council met every evening to decide whether the old stone bridge would hold. "
    "Engineers who had surveyed it the previous autumn had warned that the central pier "
    "was undermined, but the repairs were postponed because the harvest could not wait.",
    "In 1998 a small team of researchers published a method for estimating the age of "
    "sedimentary layers from the ratio of two isotopes. The technique relied on careful "
    "calibration against volcanic ash deposits whose dates were known independently. "
    "Later work showed that the method was sensitive to contamination, and several earlier "
    "estimates had to be revised by as much as ten percent.",
    "To make the sauce, melt two tablespoons of butter in a heavy pan over medium heat, "
    "then whisk in an equal amount of flour and cook for a minute without letting it "
    "color. Pour in warm milk a little at a time, stirring constantly so that no lumps "
    "form, and simmer until the mixture coats the back of a spoon. Season with salt, "
    "white pepper and a pinch of nutmeg before serving.",
)

PARITY_PROMPTS = (
    "The capital of France is",
    "Question: Is the sky blue on a clear day?\nAnswer:",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    PARITY_PARAGRAPHS[0] + "\nQuestion: did the council repair the bridge before the harvest?\nAnswer:",
    " ".join(PARITY_PARAGRAPHS) + " " + " ".join(PARITY_PARAGRAPHS),
)


def build_parity_inputs(encode, prompts=PARITY_PROMPTS, pad_id=0, short_limit=64):
    """Encode prompts into single-sequence inputs plus one right-padded batch.

    ``encode(text) -> list[int]`` must add BOS.  The padded batch contains every
    prompt of at most ``short_limit`` tokens, with an attention mask, so padding
    handling is exercised.  Returns a list of dicts with ``ids`` [B, L],
    ``mask`` [B, L] (1 = real token) and ``name``.
    """
    encoded = [list(encode(text)) for text in prompts]
    if any(len(e) < 2 for e in encoded):
        raise ValueError("every parity prompt must encode to at least two tokens")
    inputs = [{"name": f"prompt_{i}", "ids": torch.tensor([e]),
               "mask": torch.ones(1, len(e), dtype=torch.long)} for i, e in enumerate(encoded)]
    short = [e for e in encoded if len(e) <= short_limit]
    if len(short) >= 2:
        width = max(len(e) for e in short)
        ids = torch.full((len(short), width), int(pad_id), dtype=torch.long)
        mask = torch.zeros(len(short), width, dtype=torch.long)
        for row, e in enumerate(short):
            ids[row, :len(e)] = torch.tensor(e)
            mask[row, :len(e)] = 1
        inputs.append({"name": "padded_batch", "ids": ids, "mask": mask})
    return inputs


def masked_equal(reference, actual, mask):
    """Bitwise equality on real-token positions only (``mask`` [B, L])."""
    keep = mask.bool()
    return bool(torch.equal(reference[keep], actual[keep]))


def masked_max_abs_difference(reference, actual, mask):
    keep = mask.bool()
    if not keep.any():
        return 0.0
    return float((reference[keep].float() - actual[keep].float()).abs().max())


def masked_kl_and_agreement(teacher_logits, student_logits, mask):
    """Sum of per-position KL(teacher || student) and top-1 agreements over real tokens."""
    keep = mask.bool()
    teacher = teacher_logits[keep].float()
    student = student_logits[keep].float()
    log_teacher = F.log_softmax(teacher, dim=-1)
    log_student = F.log_softmax(student, dim=-1)
    kl = F.kl_div(log_student, log_teacher, reduction="none", log_target=True).sum(dim=-1)
    return {"positions": int(keep.sum()), "kl_sum": float(kl.double().sum()),
            "agreement_sum": int((teacher.argmax(-1) == student.argmax(-1)).sum())}


def summarize_kl(parts):
    total = sum(p["positions"] for p in parts)
    return {"positions": total,
            "teacher_kl_nats_per_token": sum(p["kl_sum"] for p in parts) / total,
            "dense_top1_agreement": sum(p["agreement_sum"] for p in parts) / total}
