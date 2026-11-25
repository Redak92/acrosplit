"""Generation-time baselines that share AcroSplit's filtered shortlist.

All of these run after temperature / top-k / nucleus. They are shortlist
adaptations: vLLM only returns 20 logprobs, so none of them split the full
vocabulary the way the original papers do.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch

from engine.acrosplit import AcroSplitWatermark, context_seed


def _ids(history: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(history, torch.Tensor):
        return history.tolist()
    return list(history)


def _finite_tokens(logits: torch.Tensor) -> tuple[list[int], torch.Tensor]:
    finite = torch.isfinite(logits[0])
    idx = torch.where(finite)[0]
    return idx.tolist(), logits[0, idx].float()


def _hash_bit(key: str, history: Sequence[int], extra: int = 0) -> int:
    h = hashlib.sha256(key.encode("utf-8"))
    for t in history[-4:]:
        h.update(int(t).to_bytes(8, "little", signed=False))
    h.update(int(extra).to_bytes(8, "little", signed=False))
    return h.digest()[0] & 1


class _PayloadWatermark:
    apply_after_temperature = True

    def __init__(self, secret_key: str = "acrosplit-key", eos_token_id: int | None = None):
        self.secret_key = secret_key
        self.eos_token_id = eos_token_id
        self.message: list[int] = []
        self._cue_count = 0
        self.cue_masses: list[float] = []
        self._w_sum_dev = 0.0
        self._w_sum_var = 0.0

    def set_message(self, message) -> None:
        if isinstance(message, str):
            self.message = [int(c) for c in message]
        else:
            self.message = [int(b) for b in message]

    def reset(self) -> None:
        self._cue_count = 0
        self.cue_masses = []
        self._w_sum_dev = 0.0
        self._w_sum_var = 0.0

    def w_score(self) -> float:
        if self._w_sum_var <= 0.0:
            return 0.0
        return self._w_sum_dev / math.sqrt(self._w_sum_var)

    def _bit(self, history) -> int:
        if not self.message:
            return 0
        return self.message[self._cue_count % len(self.message)]


class SweetWatermark(_PayloadWatermark):
    """Kirchenbauer green bias only when shortlist entropy exceeds a threshold."""

    def __init__(self, secret_key: str = "acrosplit-key", *, entropy_threshold: float = 1.0, delta: float = 2.0, **kw):
        super().__init__(secret_key, kw.get("eos_token_id"))
        self.entropy_threshold = entropy_threshold
        self.delta = delta

    def apply(self, logits: torch.Tensor, tokens_so_far) -> torch.Tensor:
        ids, scores = _finite_tokens(logits)
        if len(ids) < 2:
            return logits
        probs = torch.softmax(scores, dim=-1)
        ent = float((-probs * torch.log(probs.clamp_min(1e-12))).sum())
        if ent < self.entropy_threshold:
            return logits
        history = _ids(tokens_so_far)
        out = logits.clone()
        for tid in ids:
            if _hash_bit(self.secret_key, history, tid) == 1:
                out[0, tid] = out[0, tid] + self.delta
        self._cue_count += 1
        return out


class MpacWatermark(_PayloadWatermark):
    """Position-allocation color bias on the shortlist (MPAC-style)."""

    def __init__(self, secret_key: str = "acrosplit-key", *, delta: float = 2.0, **kw):
        super().__init__(secret_key, kw.get("eos_token_id"))
        self.delta = delta

    def apply(self, logits: torch.Tensor, tokens_so_far) -> torch.Tensor:
        ids, _ = _finite_tokens(logits)
        if len(ids) < 2 or not self.message:
            return logits
        history = _ids(tokens_so_far)
        pos = context_seed(self.secret_key, history[-2:] or history) % len(self.message)
        bit = self.message[pos]
        out = logits.clone()
        for tid in ids:
            color = _hash_bit(self.secret_key, history, tid)
            if color == bit:
                out[0, tid] = out[0, tid] + self.delta
        self._cue_count += 1
        return out


class BalanceNudgeWatermark(AcroSplitWatermark):
    """Same fill-lighter split as AcroSplit, but a logit nudge instead of a hard mask."""

    def __init__(self, *args, delta: float = 2.0, **kwargs):
        kwargs.setdefault("coding", "repeat")
        super().__init__(*args, **kwargs)
        self.delta = delta

    def apply(self, logits: torch.Tensor, tokens_so_far) -> torch.Tensor:
        if not self.message:
            return logits
        history = self._normalize_history(tokens_so_far)
        if not self.should_embed(logits, history):
            return logits
        green, red, token_ids, probs, _ = self.partition_from_logits(logits, history)
        slot = self.message_slot(history, self._cue_count, len(self.message))
        bit = self.message[slot]
        self._cue_count += 1
        chosen = green if bit == 1 else red
        mass_map = dict(zip(token_ids, probs))
        m = sum(float(mass_map[tid]) for tid in chosen if tid in mass_map)
        self.cue_masses.append(m)
        self._w_sum_dev += 1.0 - m
        self._w_sum_var += m * (1.0 - m)
        out = logits.clone()
        for tid in chosen:
            out[0, tid] = out[0, tid] + self.delta
        return out


class CodeIPLogitsWatermark(MpacWatermark):
    """CodeIP's hash-on-logits idea without the grammar LSTM (shortlist only)."""


class DiscopShortlistWatermark(_PayloadWatermark):
    """Distribution-copy sampling on the filtered shortlist (binary Discop)."""

    def apply(self, logits: torch.Tensor, tokens_so_far) -> torch.Tensor:
        return logits

    def sample_token(self, logits: torch.Tensor, tokens_so_far) -> torch.Tensor:
        ids, scores = _finite_tokens(logits)
        device = logits.device
        if not ids:
            return torch.tensor([[0]], device=device)
        probs = torch.softmax(scores, dim=-1)
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        cdf = []
        acc = 0.0
        for i in order:
            acc += float(probs[i])
            cdf.append((ids[i], acc))
        history = _ids(tokens_so_far)
        rng = torch.rand((), device="cpu").item()
        bit = self._bit(history)
        u = (rng + 0.5 * bit) % 1.0
        chosen = cdf[-1][0]
        for tid, end in cdf:
            if u <= end:
                chosen = tid
                break
        self._cue_count += 1
        return torch.tensor([[chosen]], device=device, dtype=torch.long)


def make_baseline(name: str, **kwargs):
    name = name.lower()
    if name in ("sweet",):
        return SweetWatermark(**kwargs)
    if name in ("mpac",):
        return MpacWatermark(**kwargs)
    if name in ("balance", "balance_marking", "balancemarking"):
        return BalanceNudgeWatermark(**kwargs)
    if name in ("codeip", "codeip_logits"):
        return CodeIPLogitsWatermark(**kwargs)
    if name in ("discop", "discop_shortlist"):
        return DiscopShortlistWatermark(**kwargs)
    raise ValueError(f"unknown baseline {name}")
