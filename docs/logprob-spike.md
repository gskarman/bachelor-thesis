# Log-probability extraction spike — Gemma 4 via Ollama

**Date:** 2026-04-23 · **Owner:** Thread A · **Resolves:** D11, unblocks D6 calibration.

## TL;DR

**Path 1 (Ollama Python client, top-level `logprobs` + `top_logprobs` kwargs) works.** Confirmed on `gemma4:e4b` and `gemma4:31b`. No fallback needed. Two-call trick and `llama-cpp-python` are not required for this cycle.

## Working call (10 lines)

```python
import ollama
c = ollama.Client(host="http://localhost:11434")
resp = c.generate(
    model="gemma4:e4b",         # or "gemma4:31b"
    prompt=PROMPT,
    think=False,                # critical: Gemma 4 is a thinking model
    logprobs=True,              # TOP-LEVEL kwarg, not inside `options`
    top_logprobs=10,            # k candidates per generated token
    options={"num_predict": 1, "temperature": 0.0},
)
d = resp.model_dump()
first = d["logprobs"][0]        # one entry per generated token
# first = {"token": "No", "logprob": -0.58, "top_logprobs": [{"token":"No","logprob":-0.58}, {"token":"Yes","logprob":-0.82}, ...]}
```

## Response shape (Ollama 0.21.1 + Python client 0.6.1)

```
resp.logprobs = [
  {
    "token":        str,            # the emitted token (e.g. "Yes", "No")
    "logprob":      float,          # log-probability of the emitted token
    "bytes":        list[int],      # UTF-8 bytes of the token
    "top_logprobs": [               # length = top_logprobs param (default 0/None)
      {"token": str, "logprob": float, "bytes": list[int]},
      ...
    ],
  },
  ...                               # one entry per generated token
]
```

For **D6 calibration** we emit exactly one token (`num_predict=1`) and read `resp.logprobs[0].top_logprobs` for the per-candidate `{"Yes", "No", "Other", ...}` log-probabilities.

## Probed candidates + live data

**Model:** `gemma4:e4b` · **Prompt:** canonical classifier template in `aitd/classifier.py`. Passage: "The mitochondria is the powerhouse of the cell. It produces ATP through oxidative phosphorylation." (stock textbook phrasing).

```
response = "No"
logprobs[0].token = "No",   logprob = -0.5827
logprobs[0].top_logprobs:
  "No"          -0.5827
  "Yes"         -0.8174
  "<|channel>"  -11.11
  "Low"         -17.24
  "AI"          -17.82
  …
```

E4B yields a ~0.23-nat gap between `No` and `Yes` — a clean, smooth calibration signal.

**Model:** `gemma4:31b` · same prompt.

```
response = "Yes"
logprobs[0].token = "Yes",  logprob ≈ 0.0   (prob ≈ 1.0)
logprobs[0].top_logprobs:
  "Yes"    -0.0000
  "No"    -15.3303
  "AI"    -19.0627
  …
```

31B is near-saturated — AUROC will be near-ceiling and ECE harder to interpret on a skewed score distribution. This is expected for a stronger model and is what D6 calibration is meant to cope with.

## Latency (M4 Max, 64 GB)

| Model | Cold load | Warm single call (`num_predict=1`) | Projected n=1000 warm |
|---|---|---|---|
| `gemma4:e4b` | ~3 s | ~0.05–0.10 s | ~1–2 min |
| `gemma4:31b` | ~18 s | ~0.22–0.30 s | ~4–5 min |

Conclusion: **31B on n=1000 is tractable** — gates D8 Day 2 rollout.

## What did NOT work (for future-me)

1. **`logprobs` / `top_logprobs` inside `options: {...}`.** Ollama silently drops these if nested. The server expects them as top-level request fields.
   ```json
   ❌ {"options": {"logprobs": 5}}        // ignored
   ✅ {"logprobs": true, "top_logprobs": 5, "options": {"num_predict": 1}}
   ```
2. **`think` left default on Gemma 4.** Gemma 4 is a thinking model; without `think=False` it emits hidden think tokens, the `response` field is empty, and logprobs cover the think tokens, not the answer token. Always set `think=False` for single-token classification.

## Caveats for downstream consumers

- **Token identity matters.** `"Yes"` ≠ `" Yes"` ≠ `"YES"`. Gemma tokenizes each distinctly. D6 calibration must match on the exact token string — check the classifier prompt's preceding whitespace/capitalization so the canonical yes/no tokens appear at rank 1/2.
- **"Other" channel.** The current `classifier.PROMPT_TEMPLATE` only asks for Yes/No. For D5 three-class, extend the prompt to explicitly list "Yes/No/Other" so the model has a canonical third-class token to emit. Without that, the `logprob("Other")` signal will be spread across many near-synonyms.
- **Near-saturation on 31B.** When `logprob(chosen) ≈ 0`, the margin `logprob(yes) − logprob(no)` is dominated by the loser token. Still monotonic, still usable — but logistic regression may find `logprob(chosen)` not informative beyond label. Consider features `{margin, max_lp, entropy_over_top_k}`.
- **`top_logprobs` costs.** k=10 adds negligible server-side cost for `num_predict=1`. For longer generations (which we don't do in classification), the payload grows linearly.

## Recommended wiring for `ollama_client.py`

- Add `logprobs: dict[str, float] | None = None` to `GenerationResult`.
- Add `return_logprobs: bool = False` (+ optional `top_logprobs_k: int = 10`) to `OllamaClient.generate()`.
- When enabled, pass `logprobs=True, top_logprobs=top_logprobs_k` to `self.client.generate(...)` and extract `resp.logprobs[0].top_logprobs` into a `{token: logprob}` dict of size `top_logprobs_k`.
- Preserve the existing tenacity retry shape — nothing about logprobs changes retry semantics.

Rejected alternatives:
- **Two-call trick** (compare forced-continuation scores): unnecessary — direct logprobs work.
- **`llama-cpp-python`** direct: unnecessary unless we later need token-level ops Ollama can't expose (e.g. full-vocab logits). Keep as a documented fallback only.
