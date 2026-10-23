"""
API for AcroSplit steganography: nucleus-split balanced red/green lists.

Usage:
    from api import AcroSplitAPI

    api = AcroSplitAPI()
    hidden = api.hide("Write a coinflip script", "10110")
    recovered = api.reveal(hidden)
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import torch

from engine.acrosplit import AcroSplitWatermark, decode_resync, majority_decode
from engine.llm_engine import LLMEngine

DEFAULT_SYSTEM_PROMPT = (
    "You are a Python code generator. "
    "Continue the assistant reply with a valid Python module only: "
    "definitions must start at column 0 (no leading indentation on top-level code). "
    "Include all classes you use (e.g. define Node if needed). "
    "Use only ASCII quotes and comments. "
    "Do not write explanations or prose outside the code."
)

CODE_ASSISTANT_PREFILL = "```python\n"
CODE_STOP_STRINGS = ("```", "<|EOT|>")

DEFAULT_MODEL = (
    "models/Qwen2.5-Coder-7B/"
    "models--Qwen--Qwen2.5-Coder-7B/"
    "snapshots/0396a76181e127dfc13e5c5ec48a8cee09938b02"
)

DEEPSEEK_MODEL = (
    "models/deepseek-coder-1.3b-instruct/"
    "models--deepseek-ai--deepseek-coder-1.3b-instruct/"
    "snapshots/e063262dac8366fc1f28a4da0ff3c50ea66259ca"
)


def _as_bits(message: str | Sequence[int]) -> list[int]:
    if isinstance(message, str):
        if not message or any(c not in "01" for c in message):
            raise ValueError("String message must contain only '0' and '1'")
        return [int(c) for c in message]
    bits = list(message)
    if not bits or any(b not in (0, 1) for b in bits):
        raise ValueError("Message must be a non-empty sequence of 0/1 bits")
    return bits


def _bits_to_str(bits: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in bits)


def normalize_generated_code(text: str) -> str:
    """Fix common DeepSeek / fence artifacts so generated code can compile."""
    text = text.replace("Ċ", "\n").replace("Ġ", " ").replace("č", "\t")
    for bad, good in (
        ("â€™", "'"),
        ("â€˜", "'"),
        ("â€œ", '"'),
        ("â€\x9d", '"'),
        ("â€“", "-"),
        ("â€”", "-"),
        ("â†’", "->"),
        ("â†ĺ", "->"),
        ("âĢĻ", "'"),
        ("âĢľ", '"'),
        ("âĢĿ", '"'),
        ("âĢĺ", "'"),
    ):
        text = text.replace(bad, good)

    lines = text.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if nonempty:

        def _indent(ln: str) -> int:
            return len(ln) - len(ln.lstrip(" \t"))

        min_indent = min(_indent(ln) for ln in nonempty)
        if min_indent > 0:
            lines = [
                (ln[min_indent:] if len(ln) >= min_indent else ln.lstrip(" \t"))
                if ln.strip()
                else ""
                for ln in lines
            ]

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip() + "\n"


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:python|py)?\s*\n([\s\S]*?)\n```\s*$", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            body = parts[1]
            if body.lstrip().lower().startswith(("python", "py")):
                body = body.split("\n", 1)[1] if "\n" in body else ""
            return body.strip()
    return text


@dataclass
class SyntaxResult:
    ok: bool
    error: str | None = None


@dataclass
class HideResult:
    code: str
    prompt: str
    completion: str
    message: list[int]
    cue_tokens: int
    redundancy: float
    confidence: float
    compiles: bool
    syntax_error: str | None = None
    model_prompt: str = ""
    token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    stop_reason: str = ""
    generated_tokens: int = 0
    hit_token_limit: bool = False
    cue_steps: int = 0
    votes_per_bit: list[int] = field(default_factory=list)
    green_hits: int = 0
    red_hits: int = 0
    singleton_green_steps: int = 0
    w_score: float = 0.0
    cue_masses: list[float] = field(default_factory=list)

    @property
    def message_bits(self) -> str:
        return _bits_to_str(self.message)


@dataclass
class RevealResult:
    message: list[int]
    cue_tokens: int
    redundancy: float
    confidence: float
    raw_bits: list[int] = field(default_factory=list)
    cue_steps: int = 0
    cue_step_redundancy: float = 0.0
    votes_per_bit: list[int] = field(default_factory=list)
    green_hits: int = 0
    red_hits: int = 0
    singleton_green_steps: int = 0
    bit_votes: list[list[int]] = field(default_factory=list)
    resync: dict = field(default_factory=dict)

    @property
    def message_bits(self) -> str:
        return _bits_to_str(self.message)


class AcroSplitAPI:
    """
    Hide / reveal binary messages via nucleus-split green/red sampling.

    Sampling path (when 2+ tokens survive top-k/p):
      logits → temperature → top-k → top-p → lighter-list green/red → sample bit list
      (1 → green, 0 → red).

    Reveal teacher-forces the model on the generated prefix so the verifier can
    rebuild the same candidate set and partitions.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        whitelist_path: str = "whitelists/python/white_list.json",
        *,
        trust_remote_code: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        secret_key: str = "acrosplit-key",
        context_size: int = 2,
        temperature: float = 0.9,
        top_p: float = 0.9,
        top_k: int = 50,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
        bit_index: str = "sequential",
    ):
        self.model_path = model_path
        self.whitelist_path = whitelist_path
        self.system_prompt = system_prompt
        self.secret_key = secret_key
        self.context_size = context_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.bit_index = bit_index

        self.engine = LLMEngine(
            model_path=model_path,
            trust_remote_code=trust_remote_code,
        )
        self._steg = self._make_steg()

    @property
    def device(self) -> str:
        return self.engine.device

    def build_model_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        sys_msg = self.system_prompt if system_prompt is None else system_prompt
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]
        return (
            self.engine.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            + CODE_ASSISTANT_PREFILL
        )

    def check_syntax(self, code: str) -> SyntaxResult:
        code = normalize_generated_code(_strip_code_fences(code))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                compile(code, "<acrosplit>", "exec")
            return SyntaxResult(ok=True)
        except SyntaxError as e:
            return SyntaxResult(ok=False, error=f"{e.msg} (line {e.lineno})")

    def hide(
        self,
        prompt: str,
        message: str | Sequence[int],
        *,
        system_prompt: str | None = None,
        whitelist_path: str | None = None,
        secret_key: str | None = None,
        context_size: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = None,
        max_new_tokens: int | None = None,
        bit_index: str | None = None,
    ) -> HideResult:
        bits = _as_bits(message)
        model_prompt = self.build_model_prompt(prompt, system_prompt=system_prompt)

        steg = self._make_steg(
            whitelist_path=whitelist_path,
            secret_key=secret_key,
            context_size=context_size,
            bit_index=bit_index,
        )
        steg.set_message(bits)

        gen = self.engine.generate(
            prompt=model_prompt,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            top_k=top_k if top_k is not None else self.top_k,
            min_p=min_p if min_p is not None else self.min_p,
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty is not None
                else self.repetition_penalty
            ),
            watermark=steg,
            seed=self.seed if seed is None else seed,
            stop_strings=list(CODE_STOP_STRINGS),
            max_new_tokens=max_new_tokens,
        )

        prompt_ids = self.engine.tokenizer.encode(model_prompt)
        generated_ids = [step.sampled_token_id for step in gen.steps]
        token_ids = prompt_ids + generated_ids

        temp = temperature if temperature is not None else self.temperature
        p = top_p if top_p is not None else self.top_p
        k = top_k if top_k is not None else self.top_k
        mp = min_p if min_p is not None else self.min_p
        rp = (
            repetition_penalty
            if repetition_penalty is not None
            else self.repetition_penalty
        )

        decoded = self._reveal_tokens(
            steg,
            token_ids,
            message_length=len(bits),
            prompt_length=len(prompt_ids),
            temperature=temp,
            top_p=p,
            top_k=k,
            min_p=mp,
            repetition_penalty=rp,
        )
        cue_steps = int(decoded["cue_steps"])
        redundancy, confidence = self._stats(cue_steps, len(bits))

        code = normalize_generated_code(_strip_code_fences(gen.text))
        syntax = self.check_syntax(code)

        return HideResult(
            code=code,
            prompt=prompt,
            completion=gen.text,
            message=bits,
            cue_tokens=cue_steps,
            redundancy=redundancy,
            confidence=confidence,
            compiles=syntax.ok,
            syntax_error=syntax.error,
            model_prompt=model_prompt,
            token_ids=token_ids,
            generated_token_ids=generated_ids,
            stop_reason=gen.stop_reason,
            generated_tokens=gen.generated_tokens,
            hit_token_limit=(gen.stop_reason == "context"),
            cue_steps=cue_steps,
            votes_per_bit=list(decoded["votes_per_bit"]),
            green_hits=int(decoded["green_hits"]),
            red_hits=int(decoded["red_hits"]),
            singleton_green_steps=int(decoded["singleton_green_steps"]),
            w_score=float(steg.w_score()),
            cue_masses=list(steg.cue_masses),
        )

    def reveal(
        self,
        source: HideResult | str | Sequence[int],
        message_length: int | None = None,
        *,
        prompt: str | None = None,
        whitelist_path: str | None = None,
        secret_key: str | None = None,
        context_size: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        bit_index: str | None = None,
        resync: bool = False,
    ) -> RevealResult:
        steg = self._make_steg(
            whitelist_path=whitelist_path,
            secret_key=secret_key,
            context_size=context_size,
            bit_index=bit_index,
        )

        if isinstance(source, HideResult):
            msg_len = message_length if message_length is not None else len(source.message)
            model_prompt = source.model_prompt or source.prompt
            prompt_ids = self.engine.tokenizer.encode(model_prompt)
            token_ids = prompt_ids + source.generated_token_ids
            prompt_length = len(prompt_ids)
        elif isinstance(source, str):
            if message_length is None:
                raise ValueError("message_length is required when revealing from text")
            msg_len = message_length
            if prompt is not None:
                prompt_ids = self.engine.tokenizer.encode(prompt)
                generated = self.engine.tokenizer.encode(source)
                token_ids = prompt_ids + generated
                prompt_length = len(prompt_ids)
            else:
                token_ids = self.engine.tokenizer.encode(source)
                prompt_length = 0
        else:
            if message_length is None:
                raise ValueError("message_length is required when revealing from token ids")
            msg_len = message_length
            token_ids = list(source)
            if prompt is not None:
                prompt_length = len(self.engine.tokenizer.encode(prompt))
            else:
                prompt_length = 0

        if msg_len <= 0:
            raise ValueError("message_length must be a positive integer")

        decoded = self._reveal_tokens(
            steg,
            token_ids,
            message_length=msg_len,
            prompt_length=prompt_length,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            top_k=top_k if top_k is not None else self.top_k,
            min_p=min_p if min_p is not None else self.min_p,
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty is not None
                else self.repetition_penalty
            ),
            resync=resync,
        )
        cue_steps = int(decoded["cue_steps"])
        redundancy, confidence = self._stats(cue_steps, msg_len)

        return RevealResult(
            message=list(decoded["message"]),
            cue_tokens=cue_steps,
            redundancy=redundancy,
            confidence=confidence,
            raw_bits=list(decoded["raw_bits"]),
            cue_steps=cue_steps,
            cue_step_redundancy=cue_steps / msg_len if msg_len else 0.0,
            votes_per_bit=list(decoded["votes_per_bit"]),
            green_hits=int(decoded["green_hits"]),
            red_hits=int(decoded["red_hits"]),
            singleton_green_steps=int(decoded["singleton_green_steps"]),
            bit_votes=list(decoded["bit_votes"]),
            resync=dict(decoded.get("resync") or {}),
        )

    def _make_steg(
        self,
        *,
        whitelist_path: str | None = None,
        secret_key: str | None = None,
        context_size: int | None = None,
        bit_index: str | None = None,
    ) -> AcroSplitWatermark:
        return AcroSplitWatermark(
            secret_key=self.secret_key if secret_key is None else secret_key,
            context_size=self.context_size if context_size is None else context_size,
            whitelist_path=whitelist_path or self.whitelist_path,
            bit_index=self.bit_index if bit_index is None else bit_index,
        )

    @torch.inference_mode()
    def _reveal_tokens(
        self,
        steg: AcroSplitWatermark,
        token_ids: list[int],
        *,
        message_length: int,
        prompt_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float = 1.0,
        resync: bool = False,
    ) -> dict:
        bit_votes: list[list[int]] = [[] for _ in range(message_length)]
        raw_bits: list[int] = []
        green_hits = 0
        red_hits = 0
        singleton_green = 0
        cue_steps = 0

        if len(token_ids) <= prompt_length:
            return {
                "message": [0] * message_length,
                "message_bits": "0" * message_length,
                "raw_bits": [],
                "cue_steps": 0,
                "votes_per_bit": [0] * message_length,
                "green_hits": 0,
                "red_hits": 0,
                "singleton_green_steps": 0,
                "bit_votes": bit_votes,
                "resync": {"mode": "empty"},
            }

        device = self.engine.device
        start = max(prompt_length, 1)
        ids = torch.tensor([token_ids[:start]], dtype=torch.long, device=device)
        outputs = self.engine.model(input_ids=ids, use_cache=True)
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        for t in range(start, len(token_ids)):
            history = token_ids[:t]
            tok = token_ids[t]

            filtered = steg.filter_logits(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                generated_ids=history,
            )
            if steg.is_splittable_cue(filtered, history):
                green, red, _, _, singleton = steg.partition_from_logits(
                    filtered, history
                )
                if green and red:
                    if singleton:
                        singleton_green += 1

                    bit_pos = steg.message_slot(history, cue_steps, message_length)
                    cue_steps += 1

                    if tok in green:
                        observed = 1
                        green_hits += 1
                    elif tok in red:
                        observed = 0
                        red_hits += 1
                    else:
                        observed = -1

                    if observed in (0, 1):
                        bit_votes[bit_pos].append(observed)
                        raw_bits.append(observed)

            step_in = torch.tensor([[tok]], dtype=torch.long, device=device)
            outputs = self.engine.model(
                input_ids=step_in,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        if resync:
            recovered, resync_meta = decode_resync(raw_bits, message_length)
        else:
            recovered = majority_decode(bit_votes)
            resync_meta = {"mode": "off"}

        return {
            "message": recovered,
            "message_bits": _bits_to_str(recovered),
            "raw_bits": raw_bits,
            "cue_steps": cue_steps,
            "votes_per_bit": [len(v) for v in bit_votes],
            "green_hits": green_hits,
            "red_hits": red_hits,
            "singleton_green_steps": singleton_green,
            "bit_votes": bit_votes,
            "resync": resync_meta,
        }

    @staticmethod
    def _stats(cue_tokens: int, message_length: int) -> tuple[float, float]:
        if message_length <= 0 or cue_tokens <= 0:
            return 0.0, float("nan")
        redundancy = cue_tokens / message_length
        confidence = 1.0 - 1.0 / (redundancy ** 2)
        return redundancy, confidence
