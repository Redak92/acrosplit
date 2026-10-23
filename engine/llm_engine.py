import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from dataclasses import dataclass
from engine.utils import compute_entropy
from engine.sampling import apply_repetition_penalty, top_k_filter, top_p_filter, min_p_filter, sample_from_distribution, sample_greedy


def _token_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("content")
    return str(value)


def load_tokenizer(model_path: str, trust_remote_code: bool = False):
    """
    Prefer tokenizer.json via PreTrainedTokenizerFast.

    AutoTokenizer often falls back to slow LlamaTokenizer for DeepSeek, which
    drops whitespace (spaces/newlines become literal Ġ/Ċ artifacts).
    """
    root = Path(model_path)
    tokenizer_json = root / "tokenizer.json"
    config_path = root / "tokenizer_config.json"

    if tokenizer_json.is_file():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
        if config_path.is_file():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            bos = _token_str(cfg.get("bos_token"))
            eos = _token_str(cfg.get("eos_token"))
            pad = _token_str(cfg.get("pad_token"))
            unk = _token_str(cfg.get("unk_token"))
            if bos:
                tokenizer.bos_token = bos
            if eos:
                tokenizer.eos_token = eos
            if pad:
                tokenizer.pad_token = pad
            if unk:
                tokenizer.unk_token = unk
            if cfg.get("chat_template"):
                tokenizer.chat_template = cfg["chat_template"]
        return tokenizer

    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )


@dataclass
class DecodingStep:
    step: int
    logits: torch.Tensor
    probabilities: torch.Tensor
    log_probabilities: torch.Tensor
    entropy: float
    sampled_token_id: int
    decoded_token: str
    top_k_tokens: list[str]
    top_k_probabilities: list[float]
    candidate_set_size: int

@dataclass
class GenerateResult:
    text: str
    steps: list
    stop_reason: str  # "stop_string" | "eos" | "context"
    generated_tokens: int


class LLMEngine:
    """GPU-only causal LM wrapper."""

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required. No CPU fallback — check drivers / CUDA install."
            )
        self.device = "cuda"
        self.dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        print(f"Loading tokenizer from '{model_path}'...")
        self.tokenizer = load_tokenizer(
            model_path, trust_remote_code=trust_remote_code
        )
        print(
            f"Loading model from '{model_path}' on {self.device} "
            f"({torch.cuda.get_device_name(0)}, {self.dtype})..."
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=trust_remote_code,
            device_map="cuda",
        )
        self.model.eval()

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        watermark=None,
        seed: int | None = None,
        stop_strings: list[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerateResult:
        """
        Generate until EOS, a stop string, max_new_tokens, or context limit.
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        stops = stop_strings or []
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]

        ctx = getattr(self.model.config, "max_position_embeddings", None) or 16384
        # Physical context ceiling only — not an intentional generation budget.
        max_steps = max(1, ctx - prompt_len)
        if max_new_tokens is not None:
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be positive")
            max_steps = min(max_steps, max_new_tokens)

        generated_ids = input_ids.clone()
        decoding_steps, past_key_values, current_input = [], None, input_ids
        new_token_ids: list[int] = []
        stop_reason = "max_new_tokens" if max_new_tokens is not None else "context"

        # Hybrid / WorldCup run after temperature/top-k; others may run before.
        apply_after = (
            bool(getattr(watermark, "apply_after_temperature", False))
            if watermark
            else False
        )
        if watermark is not None and hasattr(watermark, "reset"):
            watermark.reset()
        if watermark is not None and getattr(watermark, "eos_token_id", None) is None:
            watermark.eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_steps):
            outputs = self.model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            processed_logits = next_token_logits.clone()
            if watermark is not None and not apply_after:
                processed_logits = watermark.apply(processed_logits, generated_ids[0])
            if repetition_penalty != 1.0:
                processed_logits = apply_repetition_penalty(
                    processed_logits, generated_ids, repetition_penalty
                )

            if temperature > 0.0:
                processed_logits = processed_logits / temperature

            if top_k > 0:
                processed_logits = top_k_filter(processed_logits, top_k)
            if top_p < 1.0:
                processed_logits = top_p_filter(processed_logits, top_p)
            if min_p > 0.0:
                processed_logits = min_p_filter(processed_logits, min_p)

            if watermark is not None and apply_after:
                processed_logits = watermark.apply(processed_logits, generated_ids[0])

            if temperature == 0.0:
                probs = torch.zeros_like(processed_logits)
                probs[0, torch.argmax(processed_logits, dim=-1)] = 1.0
                next_token_id = sample_greedy(processed_logits)
            else:
                probs = torch.softmax(processed_logits, dim=-1)
                next_token_id = sample_from_distribution(probs)

            token_id_val = next_token_id.item()
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
                    logits=next_token_logits.squeeze(0).detach().float().cpu(),
                    probabilities=probs.squeeze(0).detach().float().cpu(),
                    log_probabilities=torch.log(
                        probs.squeeze(0).detach().float().cpu() + 1e-10
                    ),
                    entropy=compute_entropy(probs).item(),
                    sampled_token_id=token_id_val,
                    decoded_token=self.decode_token(token_id_val),
                    top_k_tokens=[
                        self.decode_token(idx.item())
                        for idx in torch.topk(probs, min(5, probs.size(-1)), dim=-1)[1][0]
                    ],
                    top_k_probabilities=torch.topk(
                        probs, min(5, probs.size(-1)), dim=-1
                    )[0][0].tolist(),
                    candidate_set_size=(probs > 0.0).sum().item(),
                )
            )
            new_token_ids.append(token_id_val)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            current_input = next_token_id

        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return GenerateResult(
            text=text,
            steps=decoding_steps,
            stop_reason=stop_reason,
            generated_tokens=len(new_token_ids),
        )
