"""
OpenAI-compatible vLLM HTTP client for token-level AcroSplit.

Uses /v1/completions with prompt token ids + top logprobs (server max 20).
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import torch

# vLLM OpenAI server default. Override with VLLM_BASE_URL.
DEFAULT_VLLM_URL = os.getenv("VLLM_BASE_URL") or os.getenv(
    "FOYER_BASE_URL", "http://127.0.0.1:8000"
)
DEFAULT_VLLM_MODEL = os.getenv("VLLM_MODEL") or os.getenv(
    "FOYER_MODEL", "Qwen/Qwen3-8B"
)
VLLM_MAX_LOGPROBS = 20


def _normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def _optional_auth() -> tuple[str, str] | None:
    user = os.getenv("VLLM_USER") or os.getenv("FOYER_USER") or os.getenv("FOYER_USERNAME")
    password = os.getenv("VLLM_PASSWORD") or os.getenv("FOYER_PASSWORD")
    if user and password:
        return user, password
    return None


def fix_tokenizer_artifacts(text: str) -> str:
    return text.replace("Ġ", " ").replace("Ċ", "\n").replace("č", "\t")


class VLLMClient:
    def __init__(
        self,
        base_url: str = DEFAULT_VLLM_URL,
        model: str = DEFAULT_VLLM_MODEL,
        *,
        timeout: float = 300.0,
        max_logprobs: int = VLLM_MAX_LOGPROBS,
        max_retries: int = 5,
        auth: tuple[str, str] | None = None,
    ):
        self.base_url = _normalize_base_url(base_url or DEFAULT_VLLM_URL)
        self.model = model
        self.timeout = timeout
        self.max_logprobs = min(max_logprobs, VLLM_MAX_LOGPROBS)
        self.max_retries = max(1, max_retries)
        self.session = requests.Session()
        creds = auth if auth is not None else _optional_auth()
        if creds:
            self.session.auth = creds
        self.session.headers.update({"Content-Type": "application/json"})

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(url, json=payload, timeout=self.timeout)
                if r.status_code in (502, 503, 504) and attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 16))
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 16))
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"vLLM POST failed after {self.max_retries} attempts: {url}")

    def next_logits(
        self,
        prompt_token_ids: list[int],
        tokenizer,
        vocab_size: int,
        *,
        logprobs: int | None = None,
        dummy_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Sparse next-token logits given prompt_token_ids.

        Uses prompt_logprobs on ``prompt + [dummy]`` so token ids match reveal
        (completion top_logprobs are string-keyed and can mis-map).
        """
        n = self.max_logprobs if logprobs is None else min(logprobs, self.max_logprobs)
        dummy = (
            tokenizer.eos_token_id
            if dummy_token_id is None
            else dummy_token_id
        )
        if dummy is None:
            dummy = 0
        series = self.prompt_logits_series(
            list(prompt_token_ids) + [int(dummy)],
            vocab_size,
            logprobs=n,
        )
        if not series or series[-1] is None:
            return torch.full((1, vocab_size), float("-inf"), dtype=torch.float32)
        return series[-1]

    def prompt_logits_series(
        self,
        token_ids: list[int],
        vocab_size: int,
        *,
        logprobs: int | None = None,
    ) -> list[torch.Tensor | None]:
        """
        For each prompt position t, logits that predicted token_ids[t].

        Uses prompt_logprobs (token-id keys). Entry 0 is often None.
        """
        n = self.max_logprobs if logprobs is None else min(logprobs, self.max_logprobs)
        data = self._post(
            "/v1/completions",
            {
                "model": self.model,
                "prompt": list(token_ids),
                "max_tokens": 1,
                "prompt_logprobs": n,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
            },
        )
        series_raw = data["choices"][0].get("prompt_logprobs") or []
        out: list[torch.Tensor | None] = []
        for entry in series_raw:
            if not entry:
                out.append(None)
                continue
            logits = torch.full((1, vocab_size), float("-inf"), dtype=torch.float32)
            for tid_str, info in entry.items():
                tid = int(tid_str)
                if 0 <= tid < vocab_size:
                    lp = float(info["logprob"] if isinstance(info, dict) else info)
                    logits[0, tid] = lp
            out.append(logits)
        return out
