"""
Remote LLM engine backed by a vLLM OpenAI-compatible HTTP API.

Tokenizer is local (shared DeepSeek Coder vocab); logits come from top logprobs.
"""

from __future__ import annotations

import torch

from engine.llm_engine import DecodingStep, GenerateResult, load_tokenizer
from engine.sampling import (
    apply_repetition_penalty,
    min_p_filter,
    sample_from_distribution,
    sample_greedy,
    top_k_filter,
    top_p_filter,
)
from engine.utils import compute_entropy
from engine.vllm_client import (
    DEFAULT_VLLM_MODEL,
    DEFAULT_VLLM_URL,
    VLLM_MAX_LOGPROBS,
    VLLMClient,
    fix_tokenizer_artifacts,
)


class RemoteVLLMEngine:
    """Duck-types the pieces of LLMEngine needed by AcroSplit + classic gen."""

    def __init__(
        self,
        tokenizer_path: str,
        *,
        base_url: str = DEFAULT_VLLM_URL,
        model: str = DEFAULT_VLLM_MODEL,
        trust_remote_code: bool = True,
        timeout: float = 300.0,
    ):
        self.tokenizer = load_tokenizer(
            tokenizer_path, trust_remote_code=trust_remote_code
        )
        self.client = VLLMClient(base_url=base_url, model=model, timeout=timeout)
        self.device = torch.device("cpu")
        self.model = None  # marks remote backend for callers
        vocab = len(self.tokenizer)
        self.vocab_size = max(vocab, 32256)
        self.max_logprobs = VLLM_MAX_LOGPROBS
        self.is_remote = True

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def next_logits(self, token_ids: list[int]) -> torch.Tensor:
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        return self.client.next_logits(
            token_ids, self.tokenizer, self.vocab_size, logprobs=self.max_logprobs
        )

    def prompt_logits_series(self, token_ids: list[int]) -> list[torch.Tensor | None]:
        return self.client.prompt_logits_series(
            token_ids, self.vocab_size, logprobs=self.max_logprobs
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.9,
        top_p: float = 0.9,
        top_k: int = 50,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        watermark=None,
        seed: int | None = None,
        stop_strings: list[str] | None = None,
        max_new_tokens: int | None = 1024,
    ) -> GenerateResult:
        """
        Generation via remote logits.

        Without a watermark: one /v1/completions call (fast), detokenized locally
        from token pieces so DeepSeek ``Ċ`` newlines are preserved.

        With a watermark: token-by-token next_logits + AcroSplit apply (same
        sampling loop as LLMEngine).
        """
        if watermark is not None:
            return self._generate_watermarked(
                prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                watermark=watermark,
                seed=seed,
                stop_strings=stop_strings,
                max_new_tokens=max_new_tokens,
            )
        return self._generate_classic(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop_strings=stop_strings,
            max_new_tokens=max_new_tokens,
        )

    def _generate_classic(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
        stop_strings: list[str] | None,
        max_new_tokens: int | None,
    ) -> GenerateResult:
        payload = {
            "model": self.client.model,
            # Prefer token ids so server tokenization matches local tokenizer.
            "prompt": self.tokenizer.encode(prompt, add_special_tokens=False),
            "max_tokens": max_new_tokens or 1024,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            # Need token pieces for correct local detokenization.
            "logprobs": 1,
        }
        if top_k and top_k > 0:
            payload["top_k"] = min(top_k, self.max_logprobs)
        if stop_strings:
            payload["stop"] = list(stop_strings)
            payload["include_stop_str_in_output"] = False
        if seed is not None:
            payload["seed"] = seed
        data = self.client._post("/v1/completions", payload)
        choice = data["choices"][0]
        text = self._detokenize_choice(choice)
        # Cut at the earliest stop occurrence (list order ≠ text order).
        if stop_strings:
            cut = None
            for s in stop_strings:
                if not s:
                    continue
                idx = text.find(s)
                if idx >= 0 and (cut is None or idx < cut):
                    cut = idx
            if cut is not None:
                text = text[:cut]
        finish = choice.get("finish_reason") or "length"
        stop_reason = {
            "stop": "stop_string",
            "length": "max_new_tokens",
            "eos_token": "eos",
        }.get(finish, finish)
        usage = data.get("usage") or {}
        n_tok = int(usage.get("completion_tokens") or 0)
        return GenerateResult(
            text=text,
            steps=[],
            stop_reason=stop_reason,
            generated_tokens=n_tok,
        )

    def _generate_watermarked(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        watermark,
        seed: int | None,
        stop_strings: list[str] | None,
        max_new_tokens: int | None,
    ) -> GenerateResult:
        if seed is not None:
            torch.manual_seed(seed)
        stops = stop_strings or []
        # Match hide_completion / local LLMEngine encode so reveal sees the same prefix.
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt encodes to empty token list")

        max_steps = max_new_tokens or 1024
        if max_steps <= 0:
            raise ValueError("max_new_tokens must be positive")

        apply_after = bool(getattr(watermark, "apply_after_temperature", False))
        if hasattr(watermark, "reset"):
            watermark.reset()

        token_ids = list(prompt_ids)
        new_token_ids: list[int] = []
        decoding_steps: list[DecodingStep] = []
        stop_reason = "max_new_tokens"
        logits = self.next_logits(token_ids)

        for _ in range(max_steps):
            processed = logits.clone()
            hist = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            if not apply_after:
                processed = watermark.apply(processed, hist[0])
            if repetition_penalty != 1.0:
                processed = apply_repetition_penalty(
                    processed, hist, repetition_penalty
                )
            if temperature > 0.0:
                processed = processed / temperature
            if top_k > 0:
                processed = top_k_filter(processed, min(top_k, self.max_logprobs))
            if top_p < 1.0:
                processed = top_p_filter(processed, top_p)
            if min_p > 0.0:
                processed = min_p_filter(processed, min_p)
            if apply_after:
                processed = watermark.apply(processed, hist[0])

            if temperature == 0.0:
                probs = torch.zeros_like(processed)
                probs[0, torch.argmax(processed, dim=-1)] = 1.0
                next_id = sample_greedy(processed)
            else:
                probs = torch.softmax(processed, dim=-1)
                next_id = sample_from_distribution(probs)

            token_id_val = int(next_id.item())
            if token_id_val == self.tokenizer.eos_token_id:
                stop_reason = "eos"
                break

            trial_ids = new_token_ids + [token_id_val]
            trial_text = self.tokenizer.decode(trial_ids, skip_special_tokens=False)
            if any(stop in trial_text for stop in stops):
                stop_reason = "stop_string"
                break

            decoding_steps.append(
                DecodingStep(
                    step=len(new_token_ids) + 1,
                    logits=logits.squeeze(0).detach().float().cpu(),
                    probabilities=probs.squeeze(0).detach().float().cpu(),
                    log_probabilities=torch.log(
                        probs.squeeze(0).detach().float().cpu() + 1e-10
                    ),
                    entropy=compute_entropy(probs).item(),
                    sampled_token_id=token_id_val,
                    decoded_token=self.decode_token(token_id_val),
                    top_k_tokens=[],
                    top_k_probabilities=[],
                    candidate_set_size=int((probs > 0.0).sum().item()),
                )
            )
            new_token_ids.append(token_id_val)
            token_ids.append(token_id_val)
            logits = self.next_logits(token_ids)

        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=False)
        return GenerateResult(
            text=text,
            steps=decoding_steps,
            stop_reason=stop_reason,
            generated_tokens=len(new_token_ids),
        )

    def _detokenize_choice(self, choice: dict) -> str:
        """Prefer tokenizer.decode on token ids / pieces; fall back to text fixes."""
        logprobs = choice.get("logprobs") or {}
        pieces = logprobs.get("tokens") or []
        if pieces:
            # convert_tokens_to_string correctly maps Ġ/Ċ → space/newline.
            try:
                text = self.tokenizer.convert_tokens_to_string(pieces)
                return text
            except Exception:
                ids = []
                for p in pieces:
                    tid = self.tokenizer.convert_tokens_to_ids(p)
                    if isinstance(tid, int) and tid >= 0:
                        ids.append(tid)
                if ids:
                    return self.tokenizer.decode(ids, skip_special_tokens=False)
        return fix_tokenizer_artifacts(choice.get("text") or "")
