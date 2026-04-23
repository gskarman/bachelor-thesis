# Thread B — Policy induction & faithfulness

**Session brief.** Load this file + `docs/decisions.md` + `docs/threads.md` + `CLAUDE.md` + `docs/operating-brief.md`. That's your operating context.

---

## Goal

Deliver the **explanation half** of the thesis Results section:

1. A working policy-induction loop in `aitd/policy.py` — LLM proposer + F0.5 scorer + accept/reject on training data.
2. One **frozen best policy** string (≈20 lines of natural-language heuristics), selected by F0.5 on the val split.
3. A **faithfulness ablation** — whole-text policy swap (best vs. empty vs. inverted) measuring Δ in label and in logprob(yes) − logprob(no).

## Scope — which ADRs you own

From `docs/decisions.md`:

- **D1** (two-phase split — own Phase 1; coordinate with A on Phase 2 handoff).
- **D2.** Policy form (~20-line NL system prompt).
- **D3.** Induction algorithm (proposer → scorer → accept/reject, plateau stop).
- **D5.** Three-class output (your policies and induction code must treat yes/no/other as first-class).
- **D7.** Faithfulness ablation (whole-text policy swap for Inl. 5; sentence-level deferred).

## Out of scope

- `aitd/calibration.py` — Thread A.
- Log-prob extraction plumbing — Thread A.
- Writing thesis prose — comes after both threads converge on Day 6.

## Files you own

- `code/src/aitd/policy.py` *(new)* — induction proposer, scorer, accept/reject, trajectory logger.
- `code/src/aitd/faithfulness.py` *(new, Day 5)* — policy-ablation test harness.
- `code/configs/induction-*.yaml` — induction runs (seed prompts, iteration budgets, temperatures).
- `logs/policies/<policy_id>.md` — frozen best policy + its F0.5 on val + derivation trail.
- `code/configs/splits.yaml` — if not yet created by A, create it here.

## Files you share (coordinate per `docs/threads.md`)

- `code/src/aitd/classifier.py` — add `system_prompt: str \| None` param, route into Ollama as a system message. If you touch the file first, also stub `return_logprobs: bool` flag + `Prediction.logprobs` field (raise `NotImplementedError` until A fills it in).

## First three actions

1. **Scaffold `aitd/policy.py`.** Three functions minimum:
   - `propose_initial(client, labeled_examples) -> str` — one-shot call returning the first policy.
   - `score_policy(client, policy, examples) -> dict` — returns `{f0_5, precision, recall, per_class, hard_preds}`.
   - `refine(client, current_policy, misclassified_examples) -> str` — next-iteration call returning a revised policy.
   Keep `policy.py` under 200 lines (CLAUDE.md §9).
2. **Smoke-test on n=20.** Seed with 10 labeled examples (balanced yes/no). Run 5 iterations on a 20-example val subsample. Confirm the trajectory logs and accept/reject logic work. Commit both a passing run and the config.
3. **Full induction run.** n=200 training subsample for scoring, seed pool of 20 labeled examples, iteration budget 30, plateau at Δ < 0.005 over 3 accepted iters. Model: Gemma 4 E4B. Save the winning policy to `logs/policies/<policy_id>.md`.

## Execution order (dependency-driven, not day-gated)

Push through all of these in one run — open a PR per milestone, orchestrator reviews and merges, keep going. Do not "EOD" between steps.

1. **Scaffolding** (✅ landed in PR #5) — `aitd/policy.py`, induction loop, n=20 smoke.
2. **Faithfulness module** → `aitd/faithfulness.py` (whole-text policy-swap harness). Can be built in parallel with Thread A's logprob work since the module signature is mostly orthogonal.
3. **Full induction run** → n=200 using `aitd.data.make_splits` from Thread A's next PR (not the inline split in the smoke config). Max iters=30, plateau Δ<0.005 × 3, Gemma 4 E4B. Save winner to `logs/policies/<id>.md`. **Freeze** the policy and announce to orchestrator + Thread A.
4. **Faithfulness ablation** → on n=100 test subsample: best vs. empty vs. inverted policy. Output `logs/runs/<id>/faithfulness.md` (or equivalent) with Δlabel rate + mean Δ(logprob_yes − logprob_no) per policy, per-class breakdown.
5. **Trajectory figure** → F0.5 vs. iter with accepted/rejected markers. Committed image (PNG/SVG) + inline in the policy md.

## Definition of done (by Inl. 5)

- [ ] `aitd/policy.py` is callable from CLI (e.g. `aitd-induce --config configs/induction-default.yaml`) and produces a reproducible run directory under `logs/runs/`.
- [ ] One **frozen** best policy string at `logs/policies/<policy_id>.md` with its F0.5 on val (and its full derivation trail).
- [ ] Iteration-trajectory figure committed (F0.5 vs. iter; accepted/rejected markers).
- [ ] Faithfulness-ablation table: three policies × {label-change rate, mean Δ(logprob_yes − logprob_no), per-class breakdown}.
- [ ] `logs/RUNS.md` has rows for at least the smoke test, the full induction, and the faithfulness ablation.

## Notes

- You **do not need log-probs** to run induction. Phase 1 uses only hard `yes/no/other` labels. If Thread A's logprob path isn't ready by Day 3, start without it.
- Faithfulness (Day 5) **does** want log-probs — if Thread A's path isn't ready, fall back to label-only Δ for the ablation and note the limitation.
- Keep each policy file self-contained: the policy text + the val F0.5 + the parent policy's ID + the date. That's the derivation trail future-you will want.
- The seed labeled examples matter a lot. Pick a balanced set (≈5 AI + ≈5 human) from the train split; don't cherry-pick obvious cases.
- Watch for policy thrash: if F0.5 oscillates, damp the accept rule (require Δ > 0.01, not just > 0).
- Primary score is **F0.5**. Accept a candidate only if F0.5 strictly improves on val.
- Do **not** touch `decisions.md` without surfacing to Thread A via `#thesis-design`.
