# Scope audit — what was delivered vs. what V2 promised

**Audit refreshed:** 2026-04-26 (afternoon — superseded the morning audit at the same path).
**Inputs:** `docs/decisions.md` (now including new ADRs D12 + D13, frozen 2026-04-26), `docs/inl5-draft-v1.md` and `docs/inl5-draft-v2.md`, `docs/thread-a-calibration.md`, `docs/thread-b-policy-induction.md`, `docs/thread-b-results.md`, `docs/logprob-spike.md`, `logs/RUNS.md`, run artefacts under `logs/runs/` and `logs/policies/`, the meta-experiment in `logs/meta-experiment/`, code under `code/src/aitd/` (with unstaged updates to `calibration.py` and `ollama_client.py`), original Kexjobbsspecifikation V2.

> **What changed since the morning audit.** Two new ADRs landed in `decisions.md` (D12 pipeline scale-up, D13 harness resilience). Two of the three larger runs landed (induction-large at n=500 val and calibration-large at n=4000 test); the n=300 faithfulness-large run is currently in flight. A meta-experiment ran the induced classifier against the thesis draft itself and surfaced an adversarial-robustness finding now folded into §5.5. A v1 and v2 prose draft both landed under `docs/`. The morning audit's three "must-fix" items have been re-scored — calibration at scale is closed, 31B+policy is now an explicit deferral (D12), and only the literature-baseline gap remains red.

---

## 1. Headline — the numbers as of this afternoon

These are real, deterministic, reproducible from committed configs.

| metric | E4B (default) | E4B (induced policy + T2 calibration, n=4000 test) | E4B (induced policy + T2 calibration, n=200 test — superseded) | 31B (default) |
|---|---|---|---|---|
| F0.5 (HC3 `all`) | 0.933 (n=1000) | **0.934** (95% CI ≈ ±0.005) | 0.942 (95% CI ≈ ±0.025) | 0.977 (n=1000) |
| F1 | 0.951 | — | — | 0.984 |
| Precision (AI) | — | **0.943** (was 0.913 raw) | 0.933 | — |
| Recall (AI) | — | 0.898 (was 0.964 raw) | 0.980 | 1.000 across all 1800 31B examples |
| AUROC | 0.992 | 0.982 | 0.993 | **0.9998** |
| ECE | 0.035 | **0.013** (was 0.050 raw — 74 % reduction) | 0.030 (was 0.042 raw — 29 % reduction) | 0.015 |

**Per-domain F0.5** (E4B / 31B, n=200 each, default prompt — unchanged from morning audit):
- finance 0.890 / 0.992 · medicine 0.952 / 0.984 · reddit_eli5 0.917 / 0.992
- open_qa **0.615 / 0.727** · wiki_csai **0.625 / 0.868**

**Faithfulness ablation** — n=100 (committed) and n=300 (in flight):

| pair | n=100 Δlabel rate | n=100 mean Δlp(yes)−lp(no) | n=300 status |
|---|---|---|---|
| best vs. empty | 0.040 | −1.38 nats | run dir exists, in flight as of 18:10 UTC; resume-safe per D13 |
| best vs. inverted | **0.460** | **+8.90 nats** | same |
| empty vs. inverted | 0.480 | +10.27 nats | same |

The `best_vs_inverted` row remains the load-bearing claim for RQ2: a 46 % label-flip rate plus a 9-nat margin shift means the policy is causally driving decisions, satisfying behavioural faithfulness in the Jacovi & Goldberg / Lanham sense. n=300 is expected to tighten the CI roughly √3-fold but should not change the qualitative conclusion.

**Induction trajectory** — superseded by induction-large (run `2026-04-26T17-42-47_3d67db`):
- Winner at iter 1, F0.5 = 0.956 on n=500 val (P=0.953, R=0.968)
- Trajectory 0.936 → 0.956 → 0.941×5 → early-stop, ~36 min wall-clock
- The drop from the prior n=200 winner's 0.980 is the small-sample-bias correction D12 names explicitly. The n=4000 calibrated F0.5 of 0.934 is closer to the n=500 val estimate of 0.956 than to the n=200 estimate of 0.980, which is the right direction — the new numbers are honest, the old ones were optimistic.

---

## 2. Scope deviations from V2 spec — all documented, all defensible

Six original V2 → decisions.md adjustments, plus two new ones from today.

| V2 promised | What was actually done | Why | Where documented |
|---|---|---|---|
| Models: gpt-oss:20b + Qwen2.5:32b | Gemma 4 E4B + Gemma 4 31B | Local-only; faster iteration; cross-model deferred | decisions.md D8 |
| Primary metric: F1 | F0.5 | Use case "don't falsely accuse" | decisions.md D4 + context block |
| Binary yes/no output | Three-class yes/no/other | Surfaces refusals; feeds calibration | decisions.md D5 |
| Faithfulness via feature masking | Faithfulness via whole-text policy-swap ablation | Faithful-by-construction framing | decisions.md D7 |
| Per-sample NL explanation | Policy-as-explanation | Pre-empts Madsen 2024 unfaithfulness critique | decisions.md D2 + draft §5.4 |
| Re-run DetectGPT + RoBERTa baselines | Cite from literature | Saves 5–10 days; thesis contribution is policy + faithfulness | decisions.md D10 |
| ⨯ (no V2 commitment) | **Pipeline re-run at scale** (n=500 val induction; n=4000 test calibration; n=300 faithfulness) | Tighter confidence intervals; honest estimates; the prior small-n runs touched 0.2–1.2% of HC3 | **decisions.md D12 (new)** |
| ⨯ (no V2 commitment) | **Harness resilience** — keep_alive=4h, request timeout=120s, incremental writes, `--resume` | The first calibration-large attempt stalled at 3150/4000; D13 makes long runs survive Ollama hiccups | **decisions.md D13 (new)** |

D12 and D13 are pure scope additions — neither contradicts V2. They are improvements that strengthen the eventual claims rather than reframings of the method. The Method section of the report does *not* need to mention D13 (it's harness/infrastructure); D12 only needs to be reflected as the n's in §4.1, §4.2, §4.3.

---

## 3. Real gaps — what's still missing for Inl. 5

### 🔴 Literature-baseline numbers — still missing, still the highest-leverage open task

`decisions.md` D10 says DetectGPT and fine-tuned RoBERTa numbers come from published HC3 results. Both v1 and v2 drafts of §4.1 still show `[TODO: cite HC3 figure from [9]]` and `[TODO: cite HC3 figure from [4]]` in Table 1. RQ1 explicitly compares to these baselines; without the comparison the Results section under-claims.

**Action.** Pull F1 / AUROC numbers from Mitchell et al. 2023 (DetectGPT) and from MGTBench (He et al. 2024) or Pudasaini et al. 2025. Document comparability axes (HC3 subset, AI source LLM, threshold convention) in the cell or a footnote. ~30–60 min of citation work, no compute. *Same priority as the morning audit. Still uncompleted.*

### 🟡 v2 draft §4.1 narrative paragraph not reconciled with the new table

The §4.1 headline paragraph in `docs/inl5-draft-v2.md` reads:

> With the induced policy and a logistic calibrator (T2) the F0.5 of Gemma 4 E4B rises from 0.933 (default prompt, n=1000) to 0.942 (n=200 test) while ECE drops from 0.035 to 0.030 — a small absolute change in F0.5, but a 14% reduction in expected calibration error...

But the Table 1 immediately below shows the n=4000 calibrated row at F0.5 = 0.934 and ECE = 0.013. The headline paragraph is leftover v1 wording; the per-pattern analysis paragraph further down (the one starting "At larger scale, calibration moves both the hard label...") *is* updated with the new numbers (0.913 → 0.943 precision, ECE 0.050 → 0.013 — a 74 % reduction).

**Action.** Single-paragraph rewrite of the §4.1 headline para to use the n=4000 numbers consistently. ~10 min.

### 🟡 v2 draft §4.3 not yet updated with n=300 faithfulness results

The faithfulness-large run is in flight as of audit time. v2 §4.3 still uses the n=100 numbers. Once the run completes, two things to update: replace n in prose ("n = 100" → "n = 300"), refresh the Tables 4 and 5 numbers if they shift, and re-run `_append_runs_log` (D12 already wrote a row stub).

**Action.** ~20 min once the run lands.

### 🟡 Three figures still unrendered

§4.1 Figure 1 (ROC + reliability diagram from `2026-04-26T19-07-51_137899/features_test.jsonl`), §4.2 Figure 2 (trajectory plot — exists for old run at `logs/policies/2026-04-23T23-06-25_b15d8f.png` but a fresh one for `2026-04-26T17-42-47_3d67db.png` would match the current draft), §4.3 Figure 3 (margin histogram from the n=300 faithfulness predictions when it lands).

**Action.** `aitd/plots.py` already has `plot_trajectory`. Add `plot_roc`, `plot_reliability`, and `plot_margin_histogram`. ~1 hour total.

### 🟢 31B + policy + calibration — now an explicit deferral, not a gap

D12 explicitly defers this past Inl. 5. The morning audit had this as 🔴; it's now ✅ closed by acceptance. Keep the deferral mention in §5.5 Future work.

### 🟢 Sentence-level / structured-feature ablation — deferred (D7, unchanged)

### 🟢 Cross-model gpt-oss / Qwen2.5 — deferred (D8, unchanged)

### 🟢 Human evaluation — deferred (Q14, unchanged)

---

## 4. New since the morning audit — the meta-experiment

Worth its own subsection because it produced a real methodological finding now in §5.5.

The run at `logs/meta-experiment/` did the following: classify each prose section of the v2 draft (17 sections) under E4B + best policy, then iteratively rewrite the AI-flagged sections to drive the AI-flag rate down. The findings:

- **iter-1.** Style rewrites (contractions, parenthetical asides, conversational openers): 9/17 → 4/17 AI flags.
- **iter-2.** Structural rewrites (dissolve numbered lists, parallel boldface captions, set notation, formal definitions): 4/17 → 1/17. The structural triggers in the policy text are real and identifiable.
- **iter-3.** Single edit — collapsing the parallel `**RQ1.** ... **RQ2.** ...` blockquote into one continuous-prose blockquote: 1/17 → **0/17**.
- **iter-4 (transferability test).** The same 0/17 v2 prose tested against E4B + policy was tested against 31B + same policy: also 0/17, with stronger margins (mean −12 nats vs E4B's −3.79). Adversarial evasion *transfers within model family*.
- **External-detector test.** Pasted into GPTZero: **100% AI**, "highly confident". Adversarial evasion *does not transfer to a different detector class*. Two detectors looking at different surface features give opposite verdicts on the same text.
- **iter-5.** Folded the finding into §5.5 Future work as a new ~190-word adversarial-robustness paragraph (already in `inl5-draft-v2.md` §5.5). Whole-file 0/17 holds — adding the paragraph didn't re-flag anything.

**What this changes for the report.** Two paragraphs in the report directly absorb the finding:
1. §5.5 already has the adversarial-robustness paragraph.
2. §5.4 (threats to faithfulness) should optionally carry one sentence noting that *adversarial robustness is a different property from faithfulness* — Madsen-style faithfulness is about whether the explanation is causally true; the meta-experiment shows the explanation can be both causally true *and* easy to evade. Worth one explicit sentence so the reader sees the distinction.

**What it doesn't change.** The §4.3 faithfulness ablation remains the load-bearing RQ2 evidence — meta-experiment is auxiliary. Don't move it into §4 as a primary result; keep it as the §5.5 anchor.

---

## 5. Things that are *not* gaps but read like they might be

- **F0.5 dropped from 0.942 (old, n=200) to 0.934 (new, n=4000).** This is *not* a regression. It's a small-sample-bias correction. The narrower CI at n=4000 means the new number is closer to the truth; the old number was an optimistic outlier from a tiny test slice. D12 names this honesty correction explicitly.
- **The new n=500 induction winner (F0.5=0.956) scores lower than the old n=200 winner (0.980).** Same correction, same direction, same explanation. Both numbers are val estimates of the same architecture; the n=500 one is more honest.
- **Calibration test recall went down (0.980 → 0.898).** Yes — and precision went *up* (0.933 → 0.943). That's exactly what F0.5-targeted calibration is supposed to do, and it's invisible at n=200 because the calibrator's actual decision boundary was masked by small-sample noise. The new behaviour is both intended and reportable as a contribution.
- **D13 isn't an architectural change.** It's harness/infrastructure resilience. It belongs in a paragraph in §3 Method only as a passing note ("runs are resume-safe and survive any single Ollama-side hiccup"), not as a method-level change.

---

## 6. A good way to view §4 Results — refreshed

Same overall structure as the morning audit, but the artefact paths and table contents are now updated. Total budget 4–5 pages.

```
§4.1 Detection performance — RQ1                   (~1.5 p)
  Table 1: F0.5 / F1 / accuracy / AUROC / ECE for
           {E4B-default n=1000, E4B-policy+cal n=4000,
            31B-default n=1000,
            DetectGPT (literature), RoBERTa (literature)}.
           Lit rows still [TODO] — see §3 above.
  Table 2: per-domain F0.5 × 5 sub-domains × {E4B, 31B} (default).
  Figure 1: ROC curve (E4B+policy+cal, n=4000) with F0.5-optimal
            point marked.
  Figure 2: reliability diagram before/after T2.
  ~3 short paragraphs of factual prose. The first one needs to be
  reconciled to the n=4000 numbers (see §3.2 above).

§4.2 Policy induction trajectory                   (~1 p)
  Figure 3: F0.5 vs. iter for run 2026-04-26T17-42-47_3d67db
            (need to render — old run's PNG won't match new draft).
  Box: frozen policy text verbatim (~150 words). v2 draft has it.
  ~2 paragraphs.

§4.3 Faithfulness ablation — RQ2                   (~1.5 p)
  Table 4: per-policy F0.5 + class distribution
           (best / empty / inverted on n=300 test — pending run).
  Table 5: pairwise Δlabel + mean Δ(lp(yes)−lp(no)) — pending.
  Figure 4: histogram of per-example Δ(lp(yes)−lp(no))
            under best vs inverted, n=300.
  ~3 paragraphs. v2 has the prose; just swap n=100 → n=300.
```

Of the four figures listed (1, 2, 3, 4) — none are currently rendered. All four can be produced by extending `aitd/plots.py`. None require new compute.

---

## 7. Pre-Inl. 5 punch list — refreshed

Order is impact for the next 4 days.

1. 🔴 Compile DetectGPT + RoBERTa literature numbers into Table 1 row. *(~60 min, no compute. Same as the morning audit. Still open.)*
2. 🟡 Reconcile v2 draft §4.1 headline paragraph to the n=4000 numbers. *(~10 min, prose only.)*
3. 🟡 Once the in-flight n=300 faithfulness run lands, update §4.3 prose and tables. *(~20 min, prose only.)*
4. 🟡 Render Figures 1–4 via `aitd/plots.py`. *(~60 min, no new compute.)*
5. ✍️ Add the §5.4 single-sentence note distinguishing faithfulness-as-causality from adversarial robustness. *(~5 min.)*
6. ✍️ Sammanfattning + Abstract — both still `[TODO]` in v2 draft. Write last, after §4 numbers finalise. *(~60 min total.)*
7. ✍️ Migrate v2 from markdown to ACM template (Word or LaTeX) — final formatting pass before submission. Best done on Wednesday once the rest is locked. *(~2-3 hours.)*

The 🔴 item is the difference between a draft that under-claims the contribution and one that lands the RQ1 comparison cleanly. Items 2–5 are roughly an evening of writing. Items 6–7 are the Wednesday-evening / Thursday-morning slot before the 19:00 submission window closes.
