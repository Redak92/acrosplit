"""Entropy helper (logged in LLMEngine decoding steps)."""

import torch

# HF / Qwen templates and some logprobs tokens use U+2026. Several HTTP
# stacks then .encode("latin-1") that single character (error at position 0).
_PUNCT = {
    "\u2026": "...",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}


def latin1_safe_text(text: str) -> str:
    if not text:
        return text
    for src, dst in _PUNCT.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-10
    log_probs = torch.log(probs + epsilon)
    return -torch.sum(probs * log_probs, dim=-1)
