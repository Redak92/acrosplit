# AcroSplit

Embed a secret bit-string into LLM-generated Python, then recover it with
the same model and key.

**How it works (short):** after temperature / top-k / top-p, if two or more
tokens remain, they are split into mass-balanced **green** and **red** lists.
A PRNG from `secret_key` + recent tokens picks which list is filled first.
Bit `1` samples from green, bit `0` from red. EOS in the nucleus → no split
(normal sampling). Reveal rebuilds the same lists and majority-votes.

Two ways to run it:

1. **Local GPU** — Transformers loads the weights in this process (CUDA).
2. **vLLM** — a separate OpenAI-compatible server; this repo only sends
   one-token `/v1/completions` calls with logprobs.

---

## 1. Install this repo

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You need a Hugging Face model id (or a local snapshot) that both the
**tokenizer** here and the **generator** (GPU or vLLM) agree on.

vLLM is **not** installed by `requirements.txt`. Install it only if you
serve the model yourself (section 3).

---

## 2. Run on a local GPU

Needs NVIDIA CUDA. First download happens on the first `AcroSplitAPI(...)`.

Save this as `run_local.py` in the repo root (or paste it in a REPL):

```python
from api import AcroSplitAPI

PROMPT = "Write a Python function that returns True if n is even."
MESSAGE = "10110"   # bits to hide; any non-empty string of 0/1

api = AcroSplitAPI(
    model_path="Qwen/Qwen3-8B",  # or "deepseek-ai/deepseek-coder-1.3b-instruct"
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    seed=7,
    secret_key="acrosplit-key",
)

hidden = api.hide(PROMPT, MESSAGE, max_new_tokens=256)
print("--- generated code ---")
print(hidden.code)
print("--- hide ---")
print("compiles", hidden.compiles)
print("stop", hidden.stop_reason, "tokens", hidden.generated_tokens)
print("cues", hidden.cue_steps, "w_score", round(hidden.w_score, 3))

recovered = api.reveal(hidden)
print("--- reveal ---")
print("sent     ", MESSAGE)
print("recovered", recovered.message_bits)
print("match    ", recovered.message_bits == MESSAGE)
```

```bash
cd /path/to/this/repo
source .venv/bin/activate
python run_local.py
```

Use a local folder instead of a Hub id if you already downloaded weights:

```python
api = AcroSplitAPI(model_path="/data/models/Qwen3-8B", seed=7)
```

---

## 3. Run with vLLM

AcroSplit needs the **top-20** next-token distribution. Start vLLM with
`--max-logprobs 20`. The Hugging Face tokenizer loaded here **must be the
same model** the server is running.

### 3a. Start the server (other terminal)

```bash
pip install vllm
vllm serve Qwen/Qwen3-8B --port 8000 --max-logprobs 20
```

Wait until it prints that it is listening. Check:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

You should see `"id": "Qwen/Qwen3-8B"` (or whatever you served).

GPU on another machine / port 4000:

```bash
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 4000 --max-logprobs 20
```

Then use `url="http://HOST:4000"` below (`HOST` = that machine, or
`127.0.0.1` if you SSH-tunnel).

### 3b. Hide and reveal (this repo)

Save as `run_vllm.py`:

```python
from api import AcroSplitAPI

PROMPT = "Write a Python function that returns True if n is even."
MESSAGE = "10110"

api = AcroSplitAPI.from_vllm(
    model="Qwen/Qwen3-8B",          # must match vllm serve
    url="http://127.0.0.1:8000",    # change host/port if needed
    temperature=0.9,
    top_p=0.9,
    top_k=20,                       # vLLM returns at most 20 logprobs
    seed=7,
    secret_key="acrosplit-key",
)

hidden = api.hide(PROMPT, MESSAGE, max_new_tokens=256)
print("--- generated code ---")
print(hidden.code)
print("cues", hidden.cue_steps, "stop", hidden.stop_reason)

recovered = api.reveal(hidden)
print("sent     ", MESSAGE)
print("recovered", recovered.message_bits)
print("match    ", recovered.message_bits == MESSAGE)
```

```bash
source .venv/bin/activate
python run_vllm.py
```

Hide calls the server **once per generated token**. Reveal is one
`prompt_logprobs` request over the full id sequence.

### 3c. URL and auth from the environment

If you omit `url` / `model`, these are read from the environment:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000
export VLLM_MODEL=Qwen/Qwen3-8B
python run_vllm.py   # still pass model= in from_vllm, or also drop it
```

`from_vllm(model=..., url=...)` always wins over the environment.

If the server uses HTTP basic auth:

```bash
export VLLM_USER=test
export VLLM_PASSWORD='your-password'
```

Aliases: `FOYER_BASE_URL`, `FOYER_MODEL`, `FOYER_USER`, `FOYER_PASSWORD`.
A URL that already ends in `/v1` is fine.

Minimal `from_vllm` when env is set:

```python
import os
from api import AcroSplitAPI

api = AcroSplitAPI.from_vllm(
    model=os.environ["VLLM_MODEL"],
    url=os.environ["VLLM_BASE_URL"],
    top_k=20,
    seed=7,
)
```

---

## 4. Hide / reveal API

```python
hidden = api.hide(prompt, message, max_new_tokens=256)
recovered = api.reveal(hidden)
```

`message` is a string of `0`/`1` (`"10110"`) or a list `[1, 0, 1, 1, 0]`.

Useful fields on `hidden`:

| Field | Meaning |
|---|---|
| `code` | Python with markdown fences stripped |
| `completion` | raw model text |
| `generated_token_ids` | ids to keep for later reveal |
| `model_prompt` | chat template + prefill actually fed to the model |
| `cue_steps` | how many steps carried a bit |
| `w_score` | hide-time z-score from chosen-list mass |
| `compiles` | `compile()` succeeded on `code` |
| `stop_reason` | `stop_string` / `eos` / `max_new_tokens` |

**Always call `reveal(hidden)`** (or pass the same token ids). Do not
retokenize the printed `code` and expect the bits to match.

Save ids and reload later:

```python
import json
from api import AcroSplitAPI

# after hide
json.dump(
    {
        "model_prompt": hidden.model_prompt,
        "generated_token_ids": hidden.generated_token_ids,
        "n_bits": len(hidden.message),
    },
    open("hidden.json", "w"),
)

# later, same model / key / sampling
api = AcroSplitAPI.from_vllm(model="Qwen/Qwen3-8B", url="http://127.0.0.1:8000", top_k=20)
blob = json.load(open("hidden.json"))
prompt_ids = api.engine.tokenizer.encode(blob["model_prompt"])
recovered = api.reveal(
    prompt_ids + blob["generated_token_ids"],
    message_length=blob["n_bits"],
    prompt=blob["model_prompt"],
)
print(recovered.message_bits)
```

Hide and reveal must use the **same** model, `secret_key`, `context_size`,
`temperature`, `top_p`, and `top_k`.

Override sampling on a single call:

```python
hidden = api.hide(
    "Write a Python hello world",
    "1100",
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.95,
)
```

---

## 5. What “it worked” looks like

```
--- generated code ---
def is_even(n):
    return n % 2 == 0

--- hide ---
compiles True
stop stop_string tokens 24
cues 8 w_score 2.1
--- reveal ---
sent      10110
recovered 10110
match     True
```

If `cue_steps` is `0` or smaller than the number of bits, recovery can be
wrong: empty bit slots majority-vote to `0`. Generate more tokens or use a
less peaked sampler (higher `temperature` / `top_p`).

---

## 6. Troubleshooting

| Symptom | What to do |
|---|---|
| `CUDA GPU is required` | Local backend needs a GPU. Use vLLM on the GPU box instead. |
| `Connection refused` / timeout | vLLM is not up, or `url` host/port is wrong. `curl` `/v1/models`. |
| `401` / `403` | Set `VLLM_USER` and `VLLM_PASSWORD`. |
| Garbage / Ġ Ċ in the code | Tokenizer ≠ served model. Pass the same Hub id to `from_vllm(model=...)`. |
| Bits do not match | Reveal from `HideResult` or saved token ids, not from `hidden.code` text. Use the same key and sampling. |
| `top_k` silently 20 on vLLM | Server cap is 20 logprobs. Start vLLM with `--max-logprobs 20`. |

---

## Layout

| Path | Role |
|---|---|
| `api.py` | `AcroSplitAPI` and `AcroSplitAPI.from_vllm` |
| `engine/acrosplit.py` | green/red split, hide mask, majority decode |
| `engine/llm_engine.py` | local Transformers generation |
| `engine/remote_engine.py` | vLLM generation loop |
| `engine/vllm_client.py` | HTTP `/v1/completions` + logprobs |
