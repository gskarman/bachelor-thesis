# Thread A — Calibration & baselines

**Session brief.** Load this file + `docs/decisions.md` + `docs/threads.md` + `CLAUDE.md` + `docs/operating-brief.md`. That's your operating context.

---

## Goal

Deliver the **measurement and calibration half** of the thesis Results section:

1. A working log-probability extraction path for Gemma 4 via Ollama (or documented local fallback).
2. Scaled baseline numbers — n=1000 on HC3 `all` + per-domain breakdown — on both Gemma 4 E4B and 31B.
3. The **calibration layer** that turns three log-probs `{yes, no, other}` into the final F0.5-optimizing classifier on top of Thread B's frozen best policy.

## Scope — which ADRs you own

From `docs/decisions.md`:

- **D4.** Scoring split (60/20/20, seed=42). If Thread B hasn't produced `configs/splits.yaml` yet, you write it. Hash-verify so both threads can't disagree on indices.
- **D6.** Calibration features + model choice (threshold vs. logistic).
- **D8.** Model rollouts — E4B first, 31B once logprob path is stable.
- **D9.** Metric reporting — ROC + AUROC, reliability diagram + ECE, chosen operating point.
- **D11.** Log-prob extraction path (spike, validate, document).

## Out of scope

- `aitd/policy.py` — Thread B.
- Faithfulness ablation — Thread B.
- Writing thesis prose — comes after both threads converge on Day 6.

## Files you own

- `code/src/aitd/ollama_client.py` — expand to expose log-probs.
- `code/src/aitd/calibration.py` *(new)* — threshold / logistic fit over log-prob features.
- `code/configs/*.yaml` — new config variants for each scaled run.
- `code/configs/splits.yaml` — if not yet created by B, create it here.
- `docs/logprob-spike.md` *(new)* — spike findings: what works, what doesn't, Gemma 4 specifics.

## Files you share (coordinate per `docs/threads.md`)

- `code/src/aitd/classifier.py` — add `return_logprobs: bool` flag + extend `Prediction` with `logprobs: dict[str, float] \| None`. If you touch the file first, also stub `system_prompt: str \| None` param (raise `NotImplementedError` until B fills it in).

## First three actions

1. **Log-prob spike (half-day max).** Call Ollama `generate` for Gemma 4 E4B with a one-word completion prompt. Try whatever log-prob knob the current `ollama` Python client exposes (`options={"logprobs": N}`, raw mode, `/api/embed`, etc.). Document what works in `docs/logprob-spike.md`. If nothing works, try `llama-cpp-python` next. Do not cloud-call.
2. **Baseline at scale.** Copy `configs/base.yaml` → `configs/e4b-n1000.yaml` (sample_size: 1000). Run it. Add a row to `logs/RUNS.md`. This is the first "real" number for the thesis.
3. **Per-domain sweep.** 5 configs, one per HC3 split (finance / medicine / open_qa / reddit_eli5 / wiki_csai), sample_size=200 each. Collect into a per-domain table.

## Execution order (dependency-driven, not day-gated)

Push through all of these in one run — open a PR per milestone, orchestrator reviews and merges, keep going. Do not "EOD" between steps.

1. **Logprob spike** → `docs/logprob-spike.md`. (Gemma 4 E4B via Ollama 0.21.1 top-level `logprobs:true` + `top_logprobs:N` — already confirmed.)
2. **Logprob plumbing** → fill in `classifier.py:return_logprobs` (B landed the stub), plumb through `ollama_client.py`, persist on `Prediction.logprobs`.
3. **Splits + F0.5** → `aitd.data.make_splits`, `configs/splits.yaml` (seeded 60/20/20), F0.5 in `evaluation.py`. Unblocks Thread B's full induction.
4. **E4B baselines at scale** → `configs/e4b-n1000.yaml` on HC3 `all` + 5 per-domain n=200 configs. Rows in `RUNS.md`.
5. **31B parity + scale** → confirm wall-clock on a small run first, then scale. Document timing.
6. **Calibration layer** → `aitd/calibration.py`, consumes Thread B's frozen best policy from `logs/policies/<id>.md`. Produces `logs/runs/<id>/calibration.json` with chosen threshold, F0.5 on val + test, precision/recall at threshold, AUROC, ECE, reliability-diagram data.

## Definition of done (by Inl. 5)

- [ ] `docs/logprob-spike.md` documents the working path (or the honest "couldn't make it work, fell back to X").
- [ ] At least **6 rows** in `logs/RUNS.md` under E4B (1 `all` + 5 per-domain at n=200+) and at least **2 rows** under 31B.
- [ ] `aitd/calibration.py` exists, is callable from CLI, and writes `logs/runs/<run_id>/calibration.json` with: chosen threshold, F0.5 on val, F0.5 on test, precision/recall at threshold, AUROC, ECE, reliability-diagram data.
- [ ] A single Results-section table aggregating E4B-no-policy / 31B-no-policy / 31B-with-best-policy (+ calibrated) numbers.
- [ ] All runs reproducible from committed configs (never mutate a run dir; see CLAUDE.md §3).

## Notes

- Thread B freezes its best policy at end-of-Day-4. You consume the frozen string from `logs/policies/<policy_id>.md`. If B slips, run calibration on a stand-in policy (e.g. the baseline yes/no prompt from `classifier.py`) so the infra is ready when the real policy lands.
- Primary score is **F0.5**. F1 and accuracy go in the table but are not the headline.
- If 31B is slower than expected on M4 Max, scale down `sample_size` before skipping domains — a per-domain breakdown on smaller n is more useful than a single big `all` run.
- Don't silently switch to a different model tag. Any model change = note in `logs/RUNS.md`.
