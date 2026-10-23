"""Entropy helper (logged in LLMEngine decoding steps)."""

import torch


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-10
    log_probs = torch.log(probs + epsilon)
    return -torch.sum(probs * log_probs, dim=-1)
