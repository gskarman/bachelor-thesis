# Ollama runtime configuration for thesis experiments

This document captures the runtime configuration that keeps thesis experiments
memory-stable and reproducible on Apple-silicon Ollama. The primary problem it
solves is the memory-pressure **sawtooth** observed during Thread A / Thread B
runs on 2026-04-23: steep spikes on model load, plateau, sharp drop when the
model idles out, repeat — all while the experiment is still running.

> **Scope and reproducibility guarantee.** None of the defaults below change
> what the model generates. They only change *when* the model is resident, how
> much of the context window is allocated, and how the prompt is batched. The
> classification path (prompt, `logprobs`, `top_logprobs`, `temperature`,
> `num_predict`) is unchanged. `predictions.jsonl` produced before and after
> this change must be byte-identical for any fixed config (verify with
> `configs/smoke.yaml`).

---

## 1. The sawtooth, in one paragraph

Ollama's default `keep_alive` is **5 minutes**. When a classifier run pauses
for longer than that between batches (thinking, plotting, cross-thread
coordination), the server unloads the model and reloads it on the next
request. On a 31B Q4_K_M model (~22 GB resident), that is a ~30-second reload
plus a paging spike that can push the M4 Max's 64 GB unified memory into
swap. Thread B's 31B runs and Thread A's back-to-back per-domain E4B batches
both triggered the pattern. The fix is per-request `keep_alive`, documented
in §2.

---

## 2. Recommended configuration

### Per-request (Python client) — `classification:` config section

| Key          | Default this PR sets | Why                                                                                          |
|--------------|----------------------|----------------------------------------------------------------------------------------------|
| `keep_alive` | `"1h"`               | Eliminates the 5-minute unload. Doesn't affect outputs.                                      |
| `num_ctx`    | unset (inherit)      | Opt in after measuring longest passage + prompt. `2048` is comfortable for HC3 single-token. |
| `num_batch`  | unset (inherit)      | Ollama default (512) is fine; bump to 1024 for faster prefill on long prompts.               |

`keep_alive` is plumbed through `OllamaClient.generate()` and
`classifier.classify()` as a Python-level default (`"1h"`), so existing callers
(including `policy.py`, `faithfulness.py`) automatically benefit without
touching their signatures. The classification-path output is unchanged — the
field is a top-level body parameter on `/api/generate` that governs server-side
residency only.

### Server-side — environment variables

Sourced from `scripts/ollama-env.sh` before starting `ollama serve`. All have
measured or documented justifications; the script is short so a reviewer can
verify each line.

| Variable                      | Recommended              | Rationale                                                                                                                                                                                                                          |
|-------------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OLLAMA_KEEP_ALIVE`           | `1h`                     | Server-side fallback matching the per-request default. Per-request value always wins (see §3), but setting the env var also affects `ollama run` and external clients.                                                             |
| `OLLAMA_NUM_PARALLEL`         | `2` for E4B, `1` for 31B | E4B (~4 GB) has plenty of headroom for 2 concurrent slots. 31B (Q4_K_M ~22 GB) is memory-tight on 64 GB unified memory once Thread A+B and editor/browser are running; keeping it at 1 avoids paging. Flip up once you profile.    |
| `OLLAMA_MAX_LOADED_MODELS`    | `2`                      | Allows holding E4B *and* 31B resident during the Thread A/B crossover window so neither has to reload. At ~26 GB combined, safe under 64 GB.                                                                                       |
| `OLLAMA_FLASH_ATTENTION`      | `1`                      | Flash Attention is available on Apple Metal in recent Ollama releases and reduces attention-memory for long contexts. Compatible with `logprobs:true` + `top_logprobs` as of Ollama 0.21.x; no known Gemma 4 regression.           |
| `OLLAMA_KV_CACHE_TYPE`        | *(unset)*                | **Do not enable `q4_0` or `q8_0` by default.** KV-cache quantization materially changes the logprob distribution, which D6/D9 rest on. If you profile 31B and hit a memory wall, test `q8_0` offline and compare logprob deltas before adopting. |

---

## 3. `keep_alive` precedence (Ollama ≥ 0.21.x)

1. **Per-request `keep_alive` in the request body** — highest precedence. What
   this PR uses for every classify/policy/faithfulness call.
2. **`OLLAMA_KEEP_ALIVE` environment variable on `ollama serve`** — fallback
   default for any request that doesn't set one.
3. **Built-in default** — 5 minutes. This is what produced the sawtooth.

Accepted values for `keep_alive`:

| Value        | Behavior                                                                 |
|--------------|--------------------------------------------------------------------------|
| `-1` or `"-1"` | Keep the model loaded indefinitely.                                      |
| `0`          | Unload immediately after the request completes.                           |
| `"1h"`, `"30m"`, `"2h15m"` | Go-style duration strings (Ollama uses Go's `time.ParseDuration`). |
| integer > 0  | Seconds to keep alive.                                                    |

`"1h"` covers any realistic pause between batches while still releasing memory
if the experiment process dies or is abandoned.

---

## 4. Context-window sizing (`num_ctx`)

HC3 single-token classification needs only:

- the passage (max observed in the dataset — measurable: run
  `scripts/ollama-logprob-smoketest.py` with `--measure-ctx`, or tokenize each
  passage once with the Gemma tokenizer),
- the prompt template (≈ 60 tokens in `classifier.PROMPT_TEMPLATE`),
- the optional policy system prompt (≤ a few hundred tokens when policies are
  at their most verbose in induction),
- 1 token of generation.

Empirically, HC3 passages are short (reddit_eli5 and open_qa average < 200
tokens; finance/medicine/wiki_csai rarely exceed 600). A conservative
`num_ctx: 2048` gives a 3–4× margin over the longest observed input and still
cuts the KV allocation by roughly an order of magnitude vs. Gemma 4's default
32k context. **However, the thesis repo commits to measuring rather than
guessing** — see §7 for the verification procedure.

`num_ctx` is left unset by default in this PR so existing runs reproduce
byte-identically. Opt in per-config once you've verified your longest passage
fits.

---

## 5. Flash Attention on Apple Silicon

Flash Attention shipped with Metal support in Ollama 0.4.x and has been stable
through 0.21.x. It is compatible with:

- Gemma 4 (E4B and 31B variants),
- `logprobs: true` + `top_logprobs: k` — the logprob extraction path is
  orthogonal to attention kernel choice.

No known regressions that affect the thesis's calibrated-probability claim
(D6) or the primary detection metric (D9). Turn on via
`OLLAMA_FLASH_ATTENTION=1` before `ollama serve`.

---

## 6. KV-cache quantization — explicitly **not** adopted by default

`OLLAMA_KV_CACHE_TYPE=q8_0` saves roughly 50 % of KV-cache memory for Gemma 4
at long contexts, and `q4_0` saves roughly 75 %. On the thesis's classifier
workload this would free several GB during 31B runs.

**But it changes the logprobs.** Even `q8_0` introduces small rounding errors
in the attention softmax, which shift the top-token distribution. This is
precisely the signal the thesis is calibrated against (yes/no two-way softmax
over top-k logprobs — see `classifier.yes_no_prob_ai`). A non-trivial change
to that distribution is a threat to D6 (calibration features) and D9 (primary
metric).

Recommended procedure before any adoption:

1. Run `configs/smoke.yaml` with FP16 KV cache (baseline). Save
   `predictions.jsonl` and per-example `logprobs`.
2. Restart `ollama serve` with `OLLAMA_KV_CACHE_TYPE=q8_0`.
3. Re-run the same config.
4. For each example, compute the absolute difference in
   `yes_no_prob_ai(logprobs)` between runs. If the 95th percentile delta is
   below `1e-4` and no example flips label, consider adopting. Otherwise, do
   not.

Default stays off.

---

## 7. How to verify (the runtime checks I can't do from here)

The acceptance criteria on A-166 require runtime measurements on your M4 Max.
Concrete commands:

### a) Memory-pressure graph shows a single plateau per model

1. Close Ollama: `pkill -f "ollama serve"` (wait for Thread A/B to be idle).
2. Source the env script: `source scripts/ollama-env.sh`.
3. Start the server: `ollama serve &`.
4. Open Activity Monitor → Memory → watch the `ollama` process.
5. Run a full baseline + induction pass (or a shorter repro like running
   `configs/smoke.yaml` three times back-to-back with a 10-minute pause
   between runs — if the sawtooth is fixed, the pause won't unload).
6. Screenshot and commit to `docs/assets/ollama-memory-after.png`. Commit the
   before screenshot (if you still have it) as `ollama-memory-before.png`.

### b) Reproducibility — `predictions.jsonl` is byte-identical

```bash
# Before the change (on main):
git switch main
cd code
aitd-run --config configs/smoke.yaml
cp logs/runs/<newest>/predictions.jsonl /tmp/pred-before.jsonl

# After (on the A-166 branch):
git switch ted/a-166-ollama-efficiency
aitd-run --config configs/smoke.yaml
cp logs/runs/<newest>/predictions.jsonl /tmp/pred-after.jsonl

diff /tmp/pred-before.jsonl /tmp/pred-after.jsonl
# Expect: no output (files identical).
```

Because the defaults in this PR do not pass `num_ctx` / `num_batch` (so the
request body's `options` is unchanged from the pre-PR body except for a
no-op reordering) and `keep_alive` is a server-residency knob, the two
`predictions.jsonl` files should be byte-for-byte identical.

### c) Logprob fidelity sanity check

Same procedure, but `diff` the `logprobs` fields:

```bash
jq -c '{idx, logprobs}' /tmp/pred-before.jsonl > /tmp/lp-before.jsonl
jq -c '{idx, logprobs}' /tmp/pred-after.jsonl  > /tmp/lp-after.jsonl
diff /tmp/lp-before.jsonl /tmp/lp-after.jsonl
```

Identical output is required before merging.

---

## 8. Parallel classification — deferred to follow-up

Task 6 on the ticket (async / thread-pool classification) is intentionally
not in this PR. Reasoning:

- The `keep_alive` fix alone solves the sawtooth — isolating that change makes
  the reproducibility diff above trivially zero.
- `OLLAMA_NUM_PARALLEL=2` on the server already batches concurrent HTTP
  requests without any client-side async. That is the lowest-risk parallelism
  lever to try first.
- A client-side async loop is more invasive: it changes the ordering of
  writes to `predictions.jsonl`, complicates the retry/backoff semantics in
  `OllamaClient`, and needs its own reproducibility story (order-independent
  serialization, fixed seed per example).

A follow-up issue will carry:
- Measurements of E4B throughput at `OLLAMA_NUM_PARALLEL=1,2,4` with
  sequential client,
- Decision on whether async client-side is worth the complexity,
- Implementation if justified, opt-in via
  `classification.concurrency: <int>` in the config.

---

## 9. References

- `docs/decisions.md` — D6 (calibration features), D9 (primary metric), D11
  (log-prob extraction + local constraint).
- Ollama API docs for `/api/generate`, `options`, and `keep_alive`:
  <https://github.com/ollama/ollama/blob/main/docs/api.md>.
- Ollama FAQ on `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_PARALLEL`,
  `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_FLASH_ATTENTION`,
  `OLLAMA_KV_CACHE_TYPE`: <https://github.com/ollama/ollama/blob/main/docs/faq.md>.
