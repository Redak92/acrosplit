"""
AcroSplit: red/green list steganography at every splittable sampling step.

After temperature / top-k / top-p, if two or more tokens remain, sort them by
probability (high → low) and assign each token to the currently lighter list.
A PRNG seeded from the secret key + local context picks whether red or green
is filled first on ties (including the first token). Bit 1 samples from green;
bit 0 samples from red.

The verifier rebuilds the same lists by teacher-forcing the model with the
same sampling filters and key.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence

import torch

from engine.sampling import (
    apply_repetition_penalty,
    min_p_filter,
    top_k_filter,
    top_p_filter,
)


def context_seed(secret_key: str | bytes, context: Sequence[int]) -> int:
    """SHA-256(key ‖ tokens) → int64 seed for the fill-first coin flip."""
    key = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    h = hashlib.sha256(key)
    for t in context:
        h.update(int(t).to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "little") % (2**63 - 1)


def _as_bits(message: str | Sequence[int]) -> list[int]:
    if isinstance(message, str):
        if not message or any(c not in "01" for c in message):
            raise ValueError("String message must contain only '0' and '1'")
        return [int(c) for c in message]
    bits = list(message)
    if not bits or any(b not in (0, 1) for b in bits):
        raise ValueError("Message must be a non-empty sequence of 0/1 bits")
    return bits


def balanced_red_green(
    token_ids: list[int],
    probs: list[float],
    *,
    red_first: bool = False,
) -> tuple[set[int], set[int], bool]:
    """
    Partition candidates by filling the lighter list first.

    Sort by probability descending (token id as tie-break). On a mass tie,
    fill red first if ``red_first`` else green.
    """
    if not token_ids:
        return set(), set(), False
    if len(token_ids) == 1:
        return {token_ids[0]}, set(), True

    order = sorted(
        range(len(token_ids)),
        key=lambda i: (-float(probs[i]), int(token_ids[i])),
    )
    green: set[int] = set()
    red: set[int] = set()
    g_mass = 0.0
    r_mass = 0.0
    for i in order:
        tid, p = token_ids[i], float(probs[i])
        if red_first:
            fill_red = r_mass <= g_mass
        else:
            fill_red = r_mass < g_mass
        if fill_red:
            red.add(tid)
            r_mass += p
        else:
            green.add(tid)
            g_mass += p
    return green, red, False


def majority_decode(bit_votes: list[list[int]]) -> list[int]:
    recovered: list[int] = []
    for votes in bit_votes:
        if not votes:
            recovered.append(0)
            continue
        ones = sum(votes)
        zeros = len(votes) - ones
        recovered.append(0 if zeros >= ones else 1)
    return recovered


def vote_margin(bit_votes: list[list[int]]) -> int:
    total = 0
    for votes in bit_votes:
        if not votes:
            continue
        ones = sum(votes)
        total += abs(ones - (len(votes) - ones))
    return total


def _votes_for_slots(
    raw_bits: list[int], message_length: int, slot_at
) -> list[list[int]]:
    bit_votes: list[list[int]] = [[] for _ in range(message_length)]
    for i, bit in enumerate(raw_bits):
        if bit in (0, 1):
            bit_votes[slot_at(i) % message_length].append(bit)
    return bit_votes


def decode_resync(
    raw_bits: list[int],
    message_length: int,
    *,
    max_gap: int | None = None,
) -> tuple[list[int], dict]:
    """
    Recover a repeating L-bit word from an ordered cue stream after a shift
    or a single burst of missing cues.

    Tries sequential indexing, cyclic phase, and one gap of d cues.
    Slot indices only need d modulo L, so the gap width defaults to L-1.
    Picks the assignment with the largest majority margin (no extra payload bits).
    """
    if message_length <= 0:
        return [], {"mode": "empty"}
    if not raw_bits:
        return [0] * message_length, {"mode": "empty", "margin": 0}
    if max_gap is None:
        max_gap = max(message_length - 1, 1)

    best_margin = -1
    best_msg: list[int] = [0] * message_length
    best_meta: dict = {"mode": "sequential", "margin": 0}
    min_gain = max(4, len(raw_bits) // 8)
    seq_margin = 0

    def consider(slot_at, meta: dict) -> None:
        nonlocal best_margin, best_msg, best_meta, seq_margin
        votes = _votes_for_slots(raw_bits, message_length, slot_at)
        margin = vote_margin(votes)
        if meta["mode"] == "sequential":
            seq_margin = margin
        elif margin < seq_margin + min_gain:
            return
        if margin > best_margin:
            best_margin = margin
            best_msg = majority_decode(votes)
            best_meta = {**meta, "margin": margin, "seq_margin": seq_margin}

    consider(lambda i: i, {"mode": "sequential"})
    for shift in range(1, message_length):
        consider(lambda i, s=shift: i + s, {"mode": "cyclic", "shift": shift})

    n = len(raw_bits)
    cap = min(max_gap, max(n, 1))
    for gap_at in range(n + 1):
        for d in range(1, cap + 1):
            consider(
                lambda i, g=gap_at, delta=d: i if i < g else i + delta,
                {"mode": "gap", "at": gap_at, "d": d},
            )
    return best_msg, best_meta


class AcroSplitWatermark:
    """
    Split every 2+ candidate nucleus and mask to the bit's color list.

    Plug into LLMEngine.generate(watermark=...) with apply_after_temperature=True.
    """

    apply_after_temperature: bool = True

    def __init__(
        self,
        secret_key: str = "acrosplit-key",
        *,
        context_size: int = 2,
        whitelist_path: str | None = None,
        bit_index: str = "sequential",
        eos_token_id: int | Sequence[int] | None = None,
        **_kwargs,
    ):
        if bit_index not in ("sequential", "context"):
            raise ValueError("bit_index must be 'sequential' or 'context'")
        self.secret_key = secret_key
        self.context_size = context_size
        self.whitelist_path = whitelist_path
        self.bit_index = bit_index
        self.eos_token_id = eos_token_id
        self.message: list[int] = []
        self._cue_count = 0
        self.cue_masses: list[float] = []
        self._w_sum_dev = 0.0
        self._w_sum_var = 0.0

    def set_message(self, message: str | Sequence[int]) -> None:
        self.message = _as_bits(message)

    def reset(self) -> None:
        self._cue_count = 0
        self.cue_masses = []
        self._w_sum_dev = 0.0
        self._w_sum_var = 0.0

    def w_score(self) -> float:
        """Kirchenbauer-style z during hide: every cue hits the bit list.

        z = Σ(1 − m_t) / √Σ m_t(1 − m_t) where m_t is the chosen-list mass.
        """
        if self._w_sum_var <= 0.0:
            return 0.0
        return self._w_sum_dev / math.sqrt(self._w_sum_var)

    def cue_context_size(self) -> int:
        return 0

    def _normalize_history(self, tokens_so_far: Sequence[int] | torch.Tensor) -> list[int]:
        if isinstance(tokens_so_far, torch.Tensor):
            return tokens_so_far.tolist()
        return list(tokens_so_far)

    def _cue_rng(self, history: Sequence[int] | torch.Tensor) -> random.Random:
        history = self._normalize_history(history)
        ctx = (
            history[-self.context_size :]
            if len(history) >= self.context_size
            else history
        )
        return random.Random(context_seed(self.secret_key, ctx))

    def message_slot(
        self,
        history: Sequence[int] | torch.Tensor,
        sequential_index: int,
        message_length: int,
    ) -> int:
        """Which payload bit this cue carries."""
        if message_length <= 0:
            return 0
        if self.bit_index == "context":
            rng = self._cue_rng(history)
            rng.getrandbits(1)
            return rng.randrange(message_length)
        return sequential_index % message_length

    def _eos_ids(self) -> set[int]:
        v = self.eos_token_id
        if v is None:
            return set()
        if isinstance(v, int):
            return {v} if v >= 0 else set()
        return {int(x) for x in v if int(x) >= 0}

    def eos_in_candidates(self, logits: torch.Tensor) -> bool:
        """True if an EOS id survived temperature / top-k / top-p."""
        eos = self._eos_ids()
        if not eos:
            return False
        finite = torch.isfinite(logits[0])
        n = logits.size(-1)
        return any(0 <= eid < n and bool(finite[eid].item()) for eid in eos)

    def candidate_count(self, logits: torch.Tensor) -> int:
        """Number of finite (surviving top-k/p) candidates."""
        return int(torch.isfinite(logits[0]).sum().item())

    def is_splittable_cue(
        self,
        logits: torch.Tensor,
        history: Sequence[int] | torch.Tensor,
    ) -> bool:
        """True when ≥2 candidates remain after top-k/p."""
        return self.candidate_count(logits) >= 2

    def partition_from_logits(
        self,
        logits: torch.Tensor,
        history: Sequence[int] | torch.Tensor,
    ) -> tuple[set[int], set[int], list[int], list[float], bool]:
        """
        Build green/red from already-filtered logits (finite = candidate set).

        Returns (green, red, token_ids, probs, singleton).
        """
        finite = torch.isfinite(logits[0])
        if not finite.any():
            return set(), set(), [], [], False

        indices = torch.where(finite)[0]
        scores = logits[0, indices].float()
        probs_t = torch.softmax(scores, dim=-1)
        token_ids = indices.tolist()
        probs = probs_t.tolist()
        red_first = self._cue_rng(history).getrandbits(1) == 1
        green, red, singleton = balanced_red_green(
            token_ids, probs, red_first=red_first
        )
        return green, red, token_ids, probs, singleton

    @torch.no_grad()
    def apply(self, logits: torch.Tensor, tokens_so_far: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """
        After top-k/p: if 2+ candidates, mask out the list not selected by the bit.

        Bit 1 → green; bit 0 → red. A lone survivor is left unchanged.
        """
        if not self.message:
            return logits

        history = self._normalize_history(tokens_so_far)
        if self.eos_in_candidates(logits):
            return logits
        if not self.is_splittable_cue(logits, history):
            return logits

        green, red, token_ids, probs, _ = self.partition_from_logits(
            logits, history
        )
        if not green or not red:
            return logits

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
        finite = torch.isfinite(out[0])
        cand = torch.where(finite)[0]
        for tid in cand.tolist():
            if tid not in chosen:
                out[0, tid] = float("-inf")
        return out

    def filter_logits(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        generated_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Match LLMEngine: repetition penalty → temperature → top-k / top-p / min-p."""
        processed = logits.clone()
        if repetition_penalty != 1.0 and generated_ids is not None:
            if isinstance(generated_ids, torch.Tensor):
                ids = generated_ids.unsqueeze(0) if generated_ids.dim() == 1 else generated_ids
            else:
                ids = torch.tensor(
                    [list(generated_ids)],
                    dtype=torch.long,
                    device=logits.device,
                )
            ids = ids.to(device=logits.device, dtype=torch.long)
            processed = apply_repetition_penalty(
                processed, ids, repetition_penalty
            )
        if temperature > 0.0:
            processed = processed / temperature
        if top_k > 0:
            processed = top_k_filter(processed, top_k)
        if top_p < 1.0:
            processed = top_p_filter(processed, top_p)
        if min_p > 0.0:
            processed = min_p_filter(processed, min_p)
        return processed
