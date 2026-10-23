import torch

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or generated_ids.numel() == 0: return logits
    score = torch.gather(logits, 1, generated_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, generated_ids, score)
    return logits

def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0: return logits
    top_k = min(top_k, logits.size(-1))
    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
    kth_values = top_k_values[..., -1:]
    return torch.where(logits < kth_values, torch.tensor(-float('inf'), device=logits.device, dtype=logits.dtype), logits)

def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0 or top_p <= 0.0: return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, -float('inf'))

def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0.0 or min_p >= 1.0: return logits
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
    thresholds = min_p * max_probs
    return torch.where(probs < thresholds, torch.tensor(-float('inf'), device=logits.device, dtype=logits.dtype), logits)

def sample_from_distribution(probs: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    if seed is not None: torch.manual_seed(seed)
    return torch.multinomial(probs, num_samples=1)

def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1, keepdim=True)
