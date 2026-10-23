# AcroSplit

Red/green-list watermarking for LLM-generated code.

After temperature / top-k / top-p, if two or more tokens remain, they are
split into mass-balanced green and red lists. A PRNG seeded from the secret
key and local context chooses which list is filled first. Bit `1` samples
from green; bit `0` from red. If EOS is in the nucleus, the split is skipped
and sampling is unchanged. Reveal teacher-forces the same filters and key
and majority-votes the recovered bits.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from api import AcroSplitAPI

api = AcroSplitAPI(top_p=0.9, top_k=50, temperature=0.9, seed=7)
hidden = api.hide("Write a coinflip script", "10110")
recovered = api.reveal(hidden)
print(recovered.message_bits, hidden.compiles, recovered.cue_steps)
```

Prefer `api.reveal(hidden)` with a `HideResult` (exact `generated_token_ids`).
Revealing from raw text can desync because of BPE retokenization.

Core code lives in `engine/acrosplit.py` (split / hide / reveal) and
`engine/llm_engine.py` (local generation hook). `api.py` is the hide/reveal
front end.
