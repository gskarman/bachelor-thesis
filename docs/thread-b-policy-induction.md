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

## Daily deliverables

| Day | Date | Ship |
|---|---|---|
| 1 | Fri 24 Apr | Idle / support Thread A on logprob spike |
| 2 | Sat 25 Apr | Idle / support Thread A. Or start reading seed labeled examples and drafting the initial seed prompt by hand. |
| 3 | Sun 26 Apr | `aitd/policy.py` scaffolded + 5-iteration smoke-test committed |
| 4 | Mon 27 Apr | Full induction run at n=200, 30-iter budget. Pick best policy on val F0.5. Save to `logs/policies/<id>.md`. Commit trajectory figure (F0.5 vs. iter). **Freeze and announce on #thesis-design.** |
| 5 | Tue 28 Apr | `aitd/faithfulness.py` + run whole-text policy-ablation (best vs. empty vs. inverted) on n=100 test-subsample. Output: ablation table with Δlabel rate + Δ(logprob_yes − logprob_no) per policy. |
| 6 | Wed 29 Apr | Write Results "policy induction + faithfulness" subsection |
| 7 | Thu 30 Apr | Support-only |

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
