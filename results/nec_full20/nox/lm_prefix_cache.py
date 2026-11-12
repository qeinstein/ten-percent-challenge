"""Prefix-cached log-likelihood scoring for lm-eval ``HFLM`` (exact, only faster).

Multiple-choice few-shot benchmarks such as MMLU repeat the same few-shot prefix
in every request of a subject. The stock harness re-encodes that prefix for every
question. This subclass encodes each shared prefix once, keeps its attention
cache, and runs only the per-question suffix, then scores single-token
continuations from the resulting logits exactly as the stock harness does
(log-probability of the continuation token; greedy flag = argmax equals it).

Requests that cannot use the fast path (multi-token continuations, short or
absent shared prefixes) are delegated to the stock implementation, so the
class is a drop-in replacement.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM


def _lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixCachedHFLM(HFLM):
    """HFLM with shared-prefix attention-cache reuse for single-token continuations."""

    min_shared_prefix = 64
    suffix_batch = 8  # each batch row copies the shared-prefix KV cache (~0.5 GB per 1k tokens)
    prefix_stats = None

    def _expand_cache(self, cache, batch):
        cache = copy.deepcopy(cache)
        if hasattr(cache, "batch_repeat_interleave"):
            cache.batch_repeat_interleave(batch)
            return cache
        raise RuntimeError("this transformers version cannot expand a cache over the batch")

    @torch.no_grad()
    def _score_group(self, prefix, contexts):
        """Return ``{context_tuple: log_softmax over vocab at its last position (float32, cpu-free)}``."""
        device = self.device
        prefix_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        base = self.model(input_ids=prefix_ids, use_cache=True)
        cache = base.past_key_values
        unique = list(contexts)
        out = {}
        for begin in range(0, len(unique), self.suffix_batch):
            chunk = unique[begin:begin + self.suffix_batch]
            suffixes = [list(ctx[len(prefix):]) for ctx in chunk]
            lengths = [len(s) for s in suffixes]
            width = max(lengths)
            ids = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
            mask = torch.zeros(len(chunk), len(prefix) + width, dtype=torch.long, device=device)
            mask[:, :len(prefix)] = 1
            for row, suffix in enumerate(suffixes):
                ids[row, :len(suffix)] = torch.tensor(suffix, dtype=torch.long, device=device)
                mask[row, len(prefix):len(prefix) + len(suffix)] = 1
            expanded = self._expand_cache(cache, len(chunk))
            logits = self.model(input_ids=ids, attention_mask=mask, past_key_values=expanded,
                                use_cache=True).logits
            last = torch.stack([logits[row, length - 1] for row, length in enumerate(lengths)])
            logprobs = F.log_softmax(last[:, :self.vocab_size].float(), dim=-1)
            for row, ctx in enumerate(chunk):
                out[ctx] = logprobs[row]
            del expanded, logits
        del cache, base
        return out

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        self.prefix_stats = self.prefix_stats or {"fast_requests": 0, "fallback_requests": 0, "prefix_groups": 0}
        if self.backend != "causal" or not requests or any(len(r[2]) != 1 for r in requests):
            self.prefix_stats["fallback_requests"] += len(requests)
            return super()._loglikelihood_tokens(requests, disable_tqdm=disable_tqdm, override_bs=override_bs)
        answers = [None] * len(requests)
        order = sorted(range(len(requests)), key=lambda i: tuple(requests[i][1]))
        groups, current = [], [order[0]]
        for index in order[1:]:
            previous = requests[current[-1]][1]
            if _lcp(previous, requests[index][1]) >= self.min_shared_prefix:
                current.append(index)
            else:
                groups.append(current)
                current = [index]
        groups.append(current)
        fallback = []
        from tqdm import tqdm
        progress = tqdm(total=len(requests), disable=disable_tqdm or self.rank != 0,
                        desc="Prefix-cached loglikelihood")
        for group in groups:
            contexts = {tuple(requests[i][1]) for i in group}
            shortest = min(len(c) for c in contexts)
            first, last = requests[group[0]][1], requests[group[-1]][1]
            shared = min(_lcp(first, last), shortest - 1)
            if len(group) < 2 or shared < self.min_shared_prefix or len(contexts) < 2:
                fallback.extend(group)
                continue
            prefix = list(first[:shared])
            logprobs = self._score_group(prefix, sorted(contexts))
            self.prefix_stats["prefix_groups"] += 1
            for i in group:
                request_str, ctx, cont = requests[i]
                row = logprobs[tuple(ctx)]
                token = cont[0]
                answer = (float(row[token]), bool(int(row.argmax()) == token))
                answers[i] = answer
                if request_str is not None:
                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                progress.update(1)
            self.prefix_stats["fast_requests"] += len(group)
        if fallback:
            self.prefix_stats["fallback_requests"] += len(fallback)
            results = super()._loglikelihood_tokens([requests[i] for i in fallback],
                                                    disable_tqdm=True, override_bs=override_bs)
            for i, answer in zip(fallback, results):
                answers[i] = answer
                progress.update(1)
        progress.close()
        return answers
