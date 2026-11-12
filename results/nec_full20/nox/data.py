"""Small, explicit dataset loaders used by compression and evaluation."""

import hashlib
import json
import math
import random
import re
import urllib.request
from pathlib import Path

import torch


WIKITEXT2_PROCESSED_BASE = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2"
)
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def load_wikitext2_raw_split(split):
    """Load pinned canonical raw WikiText; never substitute the word-level variant."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    split = "validation" if split == "valid" else split
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported WikiText-2 split: {split!r}")
    filename = f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
    path = hf_hub_download(
        repo_id=WIKITEXT_REPO, repo_type="dataset", revision=WIKITEXT_REVISION,
        filename=filename,
    )
    from pathlib import Path
    source = Path(path).read_bytes()
    rows = pq.read_table(path, columns=["text"])["text"].to_pylist()
    text = "\n\n".join(rows)
    if not text.strip():
        raise ValueError(f"Empty canonical WikiText split: {split}")
    encoded = text.encode("utf-8")
    return text, {
        "dataset": "wikitext-2-raw-v1", "dataset_repo": WIKITEXT_REPO,
        "dataset_revision": WIKITEXT_REVISION, "split": split,
        "source_url": f"https://huggingface.co/datasets/{WIKITEXT_REPO}/resolve/{WIKITEXT_REVISION}/{filename}",
        "source_sha256": hashlib.sha256(source).hexdigest(), "source_bytes": len(source),
        "text_sha256": hashlib.sha256(encoded).hexdigest(), "text_bytes": len(encoded),
        "text_join": "double_newline", "rows": len(rows),
        "literal_unk_count": text.count("<unk>"), "fallback": False,
    }


def load_wikitext2_processed_split(split):
    """Reproduce the legacy PyTorch input, explicitly labeled as processed text."""
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported WikiText-2 split: {split!r}")

    url = f"{WIKITEXT2_PROCESSED_BASE}/{split}.txt"
    request = urllib.request.Request(url, headers={"User-Agent": "NoxResearch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()

    return raw.decode("utf-8"), {
        "dataset": "wikitext-2-processed-pytorch",
        "split": split,
        "source_url": url,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "literal_unk_count": raw.count(b"<unk>"),
    }


# ---------------------------------------------------------------------------
# Additional evaluation and calibration sources.
#
# All network access goes through ``datasets`` imported lazily.  A failed
# download RAISES: a substitute (handwritten or cached-but-different) text
# would silently change what the metrics mean.  Every loader accepts an
# injectable ``loader`` with the ``datasets.load_dataset`` call signature so
# tests can use small in-memory datasets.
# ---------------------------------------------------------------------------
C4_REPO = "allenai/c4"
BOOLQ_REPO = "google/boolq"
HELLASWAG_REPO = "Rowan/hellaswag"
SQUAD_REPO = "rajpurkar/squad"
OBQA_REPO = "allenai/openbookqa"
CODE_REPO = "codeparrot/codeparrot-clean"
BOOKS_REPO = "deepmind/pg19"

# Default mixture over categories; corpora in a category share its weight.
DEFAULT_CALIBRATION_WEIGHTS = {
    "web": 0.40, "wikipedia": 0.20, "code": 0.15, "qa": 0.25, "books": 0.0,
}


class DatasetUnavailableError(RuntimeError):
    """Raised instead of silently substituting different text."""


def _hf_rows(path, *, name=None, split=None, revision=None, streaming=False,
             data_files=None, limit=None, loader=None):
    """Load rows eagerly (up to ``limit``); raise DatasetUnavailableError on failure."""
    kwargs = {"split": split}
    for key, value in (("revision", revision), ("data_files", data_files)):
        if value is not None:
            kwargs[key] = value
    if streaming:
        kwargs["streaming"] = True
    try:
        if loader is None:
            import datasets
            loader = datasets.load_dataset
        dataset = loader(path, name, **kwargs) if name is not None else loader(path, **kwargs)
        rows = []
        for row in dataset:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    except Exception as error:
        raise DatasetUnavailableError(
            f"could not load dataset {path!r} (name={name!r}, split={split!r}); "
            "refusing to substitute other text"
        ) from error
    if not rows:
        raise DatasetUnavailableError(f"dataset {path!r} split {split!r} returned no rows")
    return rows


def _text_metadata(dataset, documents, **extra):
    joined = "\n\n".join(documents).encode("utf-8")
    return {"dataset": dataset, "documents": len(documents), "text_bytes": len(joined),
            "text_sha256": hashlib.sha256(joined).hexdigest(), **extra}


def _c4_data_file(split, shard):
    if split == "validation":
        return f"en/c4-validation.{shard:05d}-of-00008.json.gz"
    if split == "train":
        return f"en/c4-train.{shard:05d}-of-01024.json.gz"
    raise ValueError(f"Unsupported C4 split: {split!r}")


def load_c4_documents(split="validation", max_documents=256, shard=0, revision=None, loader=None):
    """Stream the first ``max_documents`` documents of one C4 ``en`` shard."""
    data_file = _c4_data_file(split, shard)
    rows = _hf_rows(C4_REPO, name="en", split=split, revision=revision, streaming=True,
                    data_files={split: data_file}, limit=max_documents, loader=loader)
    documents = [row["text"] for row in rows if row.get("text", "").strip()]
    if not documents:
        raise DatasetUnavailableError("C4 shard contained no text")
    return documents, _text_metadata(f"{C4_REPO}:en", documents, split=split,
                                     data_file=data_file, revision=revision)


def load_c4_validation_text(max_documents=256, shard=0, revision=None, loader=None):
    """C4 validation as one text (documents joined by a blank line), plus metadata."""
    documents, metadata = load_c4_documents("validation", max_documents, shard, revision, loader)
    return "\n\n".join(documents), metadata


def format_boolq_prompt(passage, question):
    """The lm-eval ``boolq`` prompt: ``{passage}\\nQuestion: {question}?\\nAnswer:``."""
    return f"{passage}\nQuestion: {question}?\nAnswer:"


def load_boolq_prompts(split="validation", max_prompts=None, revision=None, loader=None):
    """BoolQ prompt text only (no answers), formatted as lm-eval does."""
    rows = _hf_rows(BOOLQ_REPO, split=split, revision=revision, limit=max_prompts, loader=loader)
    prompts = [format_boolq_prompt(row["passage"], row["question"]) for row in rows]
    return prompts, _text_metadata(BOOLQ_REPO, prompts, split=split, revision=revision,
                                   prompt_format="{passage}\\nQuestion: {question}?\\nAnswer:")


def format_boolq_train_example(row):
    return format_boolq_prompt(row["passage"], row["question"]) + (" yes" if row["answer"] else " no")


def format_squad_example(row):
    answers = row["answers"]["text"]
    if not answers:
        raise ValueError("SQuAD row without an answer")
    return f"{row['context']}\nQuestion: {row['question']}\nAnswer: {answers[0]}"


def format_obqa_example(row):
    choices = row["choices"]
    answer = choices["text"][list(choices["label"]).index(row["answerKey"])]
    return f"{row['question_stem']} {answer}"


def format_hellaswag_context(row):
    if row.get("ctx"):
        context = row["ctx"]
    else:
        tail = row.get("ctx_b", "")
        context = (row.get("ctx_a", "") + " " + tail[:1].upper() + tail[1:]).strip()
    return f"{row['activity_label']}: {context}"


def format_hellaswag_example(row):
    return format_hellaswag_context(row) + " " + row["endings"][int(row["label"])]


def split_wikitext_articles(text):
    """Split joined raw WikiText on top-level ``= Title =`` headers."""
    parts = re.split(r"\n\n(?= = [^=])", text)
    return [part.strip() for part in parts if part.strip()]


def load_calibration_corpora(
    weights=None, max_documents=None, shard=0, loader=None,
    wikitext_loader=None, revisions=None,
):
    """Load every corpus whose category has positive weight.

    Returns a list of ``{'name', 'category', 'documents', 'metadata'}``.
    Raises (never falls back) if any required dataset cannot be loaded.
    """
    weights = dict(DEFAULT_CALIBRATION_WEIGHTS if weights is None else weights)
    limits = {"web": 20000, "code": 8000, "books": 200, "boolq": 9427,
              "squad": 20000, "obqa": 4957, "hellaswag": 20000}
    limits.update(max_documents or {})
    revisions = revisions or {}
    wikitext_loader = wikitext_loader or (lambda: load_wikitext2_raw_split("train"))
    corpora = []

    def add(name, category, documents, metadata):
        corpora.append({"name": name, "category": category, "documents": documents,
                        "metadata": metadata})

    if weights.get("web", 0) > 0:
        documents, metadata = load_c4_documents(
            "train", limits["web"], shard, revisions.get("c4"), loader)
        add("c4_train", "web", documents, metadata)
    if weights.get("wikipedia", 0) > 0:
        text, metadata = wikitext_loader()
        articles = split_wikitext_articles(text)
        add("wikitext2_train", "wikipedia", articles, {**metadata, "articles": len(articles)})
    if weights.get("code", 0) > 0:
        rows = _hf_rows(CODE_REPO, split="train", streaming=True, limit=limits["code"],
                        revision=revisions.get("code"), loader=loader)
        documents = [row["content"] for row in rows if row.get("content", "").strip()]
        add("code_train", "code", documents,
            _text_metadata(CODE_REPO, documents, split="train", revision=revisions.get("code")))
    if weights.get("books", 0) > 0:
        rows = _hf_rows(BOOKS_REPO, split="train", streaming=True, limit=limits["books"],
                        revision=revisions.get("books"), loader=loader)
        documents = [row["text"] for row in rows if row.get("text", "").strip()]
        add("books_train", "books", documents,
            _text_metadata(BOOKS_REPO, documents, split="train", revision=revisions.get("books")))
    if weights.get("qa", 0) > 0:
        for name, repo, formatter, config, key in (
            ("boolq_train", BOOLQ_REPO, format_boolq_train_example, None, "boolq"),
            ("squad_train", SQUAD_REPO, format_squad_example, None, "squad"),
            ("obqa_train", OBQA_REPO, format_obqa_example, "main", "obqa"),
            ("hellaswag_train", HELLASWAG_REPO, format_hellaswag_example, None, "hellaswag"),
        ):
            rows = _hf_rows(repo, name=config, split="train", limit=limits[key],
                            revision=revisions.get(key), loader=loader)
            documents = [formatter(row) for row in rows]
            add(name, "qa", documents,
                _text_metadata(repo, documents, split="train", revision=revisions.get(key)))
    return corpora


def load_evaluation_exclusion_texts(loader=None, wikitext_loader=None, include_wikitext_test=True):
    """Texts that calibration data must not overlap: eval splits of our benchmarks.

    BoolQ validation, HellaSwag validation/test contexts and WikiText-2 test.
    """
    texts, metadata = [], {}
    rows = _hf_rows(BOOLQ_REPO, split="validation", loader=loader)
    texts += [format_boolq_prompt(r["passage"], r["question"]) for r in rows]
    texts += [r["passage"] for r in rows]
    metadata["boolq_validation"] = len(rows)
    for split in ("validation", "test"):
        rows = _hf_rows(HELLASWAG_REPO, split=split, loader=loader)
        texts += [format_hellaswag_context(r) for r in rows]
        metadata[f"hellaswag_{split}"] = len(rows)
    if include_wikitext_test:
        wikitext_loader = wikitext_loader or (lambda: load_wikitext2_raw_split("test"))
        text, wikitext_metadata = wikitext_loader()
        articles = split_wikitext_articles(text)
        texts += articles
        metadata["wikitext2_test_articles"] = len(articles)
        metadata["wikitext2_test_sha256"] = wikitext_metadata.get("text_sha256")
    return texts, metadata


def _normalize_for_overlap(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def _stable_hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


class ContaminationFilter:
    """Reject documents sharing an exact text or any ``ngram``-word shingle with eval text."""

    def __init__(self, texts=(), ngram=13):
        if ngram < 1:
            raise ValueError("ngram must be positive")
        self.ngram = int(ngram)
        self.exact, self.shingles = set(), set()
        for text in texts:
            self.add(text)

    def add(self, text):
        normalized = _normalize_for_overlap(text)
        if not normalized:
            return
        self.exact.add(_stable_hash(normalized))
        words = normalized.split()
        for index in range(len(words) - self.ngram + 1):
            self.shingles.add(_stable_hash(" ".join(words[index:index + self.ngram])))

    def is_contaminated(self, text):
        normalized = _normalize_for_overlap(text)
        if not normalized:
            return False
        if _stable_hash(normalized) in self.exact:
            return True
        words = normalized.split()
        return any(_stable_hash(" ".join(words[i:i + self.ngram])) in self.shingles
                   for i in range(len(words) - self.ngram + 1))


def _largest_remainder(total, weights):
    """Deterministic integer allocation proportional to ``weights`` (insertion-order ties)."""
    positive = {key: float(value) for key, value in weights.items() if value > 0}
    if not positive:
        raise ValueError("at least one positive weight is required")
    scale = sum(positive.values())
    exact = {key: total * value / scale for key, value in positive.items()}
    counts = {key: int(math.floor(value)) for key, value in exact.items()}
    keys = list(positive)
    order = sorted(keys, key=lambda key: (-(exact[key] - counts[key]), keys.index(key)))
    for key in order[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def _partition_is_holdout(seed, normalized_hash, fraction):
    digest = hashlib.blake2b(f"{seed}:{normalized_hash}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2.0 ** 64 < fraction


class _TokenizedPool:
    """Deterministically shuffled documents tokenized lazily; each used at most once."""

    def __init__(self, name, documents, encode, rng, min_tokens):
        self.name, self.documents, self.encode = name, documents, encode
        self.order = list(range(len(documents)))
        rng.shuffle(self.order)
        self.cursor, self.min_tokens = 0, int(min_tokens)

    def next_tokens(self):
        while self.cursor < len(self.order):
            tokens = self.encode(self.documents[self.order[self.cursor]])
            self.cursor += 1
            if len(tokens) >= self.min_tokens:
                return list(tokens)
        raise ValueError(f"corpus {self.name!r} has too few usable documents for the requested windows")


def _build_long_window(pool, rng, length, bos_id, eos_id):
    window = [bos_id]
    while len(window) < length:
        tokens = pool.next_tokens()
        room = length - len(window)
        if len(window) == 1 and len(tokens) > room:
            start = rng.randrange(0, len(tokens) - room + 1)
            tokens = tokens[start:start + room]
        window.extend(tokens[:room])
        if eos_id is not None and len(window) < length:
            window.append(eos_id)
    return window[:length], [0]


def _build_short_window(pool, length, bos_id, short_range, rng):
    low, high = short_range
    window, starts = [], []
    while len(window) < length:
        remaining = length - len(window)
        target = rng.randint(low, high)
        if remaining - target < low:
            target = remaining
        tokens = pool.next_tokens()[:target - 1]
        starts.append(len(window))
        window.append(bos_id)
        window.extend(tokens)
    return window, starts


def build_calibration_windows(
    corpora, encode, *, bos_id, eos_id=None, num_windows, window_length=2048,
    category_weights=None, long_fraction=0.7, short_range=(256, 512),
    seed=0, contamination_filter=None, holdout_windows=0, holdout_document_fraction=0.1,
    min_document_tokens=8,
):
    """Build a mixed-domain calibration set of fixed-length token windows.

    ``corpora``: ``[{'name', 'category', 'documents', 'metadata'}]`` (see
    :func:`load_calibration_corpora`); ``encode(text) -> list[int]`` must not add
    special tokens.  Every window starts with BOS.  A ``long`` window is one
    stream (documents joined by EOS, a random slice of over-long documents; one
    segment).  A ``short`` window packs several document-aligned 256-512 token
    segments, each beginning with BOS and recorded in ``segment_starts`` so a
    consumer can treat them as independent sequences.  Windows are shuffled
    deterministically so any prefix is a mixture.  Documents are
    exact-deduplicated, filtered against ``contamination_filter`` and split into
    disjoint calibration/holdout pools by document hash.  Each document is used
    at most once per partition; exhaustion raises ValueError.
    """
    if window_length < 2 * short_range[1] or not 0 <= long_fraction <= 1:
        raise ValueError("invalid window_length or long_fraction")
    if num_windows < 1 or holdout_windows < 0:
        raise ValueError("invalid window counts")
    weights = dict(DEFAULT_CALIBRATION_WEIGHTS if category_weights is None else category_weights)
    if len({c["name"] for c in corpora}) != len(corpora):
        raise ValueError("corpus names must be unique")
    by_category = {}
    for corpus in corpora:
        by_category.setdefault(corpus["category"], []).append(corpus)
    for category, weight in weights.items():
        if weight > 0 and category not in by_category:
            raise ValueError(f"category {category!r} has positive weight but no corpus")
    corpus_weights = {c["name"]: weights.get(c["category"], 0) / len(by_category[c["category"]])
                      for c in corpora}
    active = [c for c in corpora if corpus_weights[c["name"]] > 0]

    stats, pools, seen = {}, {"calibration": {}, "holdout": {}}, set()
    for corpus in active:
        kept = {"calibration": [], "holdout": []}
        removed = {"empty": 0, "duplicate": 0, "contaminated": 0}
        for text in corpus["documents"]:
            normalized = _normalize_for_overlap(text)
            if not normalized:
                removed["empty"] += 1
                continue
            digest = _stable_hash(normalized)
            if digest in seen:
                removed["duplicate"] += 1
                continue
            if contamination_filter is not None and contamination_filter.is_contaminated(text):
                removed["contaminated"] += 1
                continue
            seen.add(digest)
            part = ("holdout" if holdout_windows and _partition_is_holdout(
                seed, digest, holdout_document_fraction) else "calibration")
            kept[part].append(text)
        pools["calibration"][corpus["name"]] = kept["calibration"]
        pools["holdout"][corpus["name"]] = kept["holdout"]
        stats[corpus["name"]] = {
            "category": corpus["category"], "input_documents": len(corpus["documents"]),
            "removed": removed, "calibration_documents": len(kept["calibration"]),
            "holdout_documents": len(kept["holdout"]),
        }

    def build(partition, count):
        allocation = _largest_remainder(count, {c["name"]: corpus_weights[c["name"]] for c in active})
        rows = []
        for corpus in active:
            name, n_windows = corpus["name"], allocation.get(corpus["name"], 0)
            if not n_windows:
                continue
            rng = random.Random(f"{seed}:{partition}:{name}")
            pool = _TokenizedPool(name, pools[partition][name], encode, rng, min_document_tokens)
            n_long = int(round(n_windows * long_fraction))
            for kind, n_kind in (("long", n_long), ("short", n_windows - n_long)):
                for _ in range(n_kind):
                    if kind == "long":
                        tokens, starts = _build_long_window(pool, rng, window_length, bos_id, eos_id)
                    else:
                        tokens, starts = _build_short_window(
                            pool, window_length, bos_id, short_range, rng)
                    rows.append((tokens, starts, name, kind))
        random.Random(f"{seed}:{partition}:shuffle").shuffle(rows)
        return rows

    def finalize(rows):
        if not rows:
            return torch.empty(0, window_length, dtype=torch.long), [], [], []
        ids = torch.tensor([row[0] for row in rows], dtype=torch.long)
        return ids, [row[1] for row in rows], [row[2] for row in rows], [row[3] for row in rows]

    calibration = finalize(build("calibration", num_windows))
    holdout = finalize(build("holdout", holdout_windows)) if holdout_windows else finalize([])

    def summary(result):
        ids, starts, sources, kinds = result
        segment_bos = all(int(ids[w, s]) == bos_id for w, ss in enumerate(starts) for s in ss)
        counts = {}
        for source, kind in zip(sources, kinds):
            counts.setdefault(source, {"long": 0, "short": 0})[kind] += 1
        return {
            "windows": int(ids.shape[0]), "tokens": int(ids.numel()),
            "window_counts_by_source": counts,
            "segments": sum(len(s) for s in starts),
            "bos_at_every_window_start": bool(ids.shape[0] == 0 or (ids[:, 0] == bos_id).all()),
            "bos_at_every_segment_start": bool(segment_bos),
            "input_ids_sha256": hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest(),
        }

    manifest = {
        "schema": "nec_calibration_mix_v1", "seed": seed, "window_length": window_length,
        "long_fraction": long_fraction, "short_range": list(short_range),
        "category_weights": weights, "bos_id": bos_id, "eos_id": eos_id,
        "corpus_weights": corpus_weights, "min_document_tokens": min_document_tokens,
        "holdout_document_fraction": holdout_document_fraction if holdout_windows else None,
        "contamination_ngram": getattr(contamination_filter, "ngram", None),
        "contamination_shingles": len(getattr(contamination_filter, "shingles", ())),
        "corpora": {c["name"]: {**stats[c["name"]], "metadata": c["metadata"]} for c in active},
        "calibration": summary(calibration),
        "holdout": summary(holdout) if holdout_windows else None,
    }
    result = {
        "input_ids": calibration[0], "segment_starts": calibration[1],
        "sources": calibration[2], "kinds": calibration[3], "manifest": manifest,
    }
    if holdout_windows:
        result.update(holdout_input_ids=holdout[0], holdout_segment_starts=holdout[1],
                      holdout_sources=holdout[2], holdout_kinds=holdout[3])
    return result


def save_calibration_mix(result, path):
    """Write one torch file plus ``<path>.manifest.json``; returns the file sha256."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    torch.save(result, temporary)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {**result["manifest"], "tokens_file": path.name, "tokens_file_sha256": digest}
    path.with_name(path.name + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return digest


def load_calibration_mix(path, expected_sha256=None):
    """Load a calibration file, verifying its hash against ``expected_sha256`` if given."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("calibration tokens file hash differs from the recorded hash")
    result = torch.load(path, map_location="cpu", weights_only=True)
    ids = result["input_ids"]
    if ids.dim() != 2 or len(result["segment_starts"]) != ids.shape[0]:
        raise ValueError("malformed calibration tokens file")
    result["file_sha256"] = digest
    return result
