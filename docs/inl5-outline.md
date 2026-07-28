# Inlämning 5 — outline

> Section-by-section outline of the thesis, mirroring the heading structure of `docs/inl5-draft-v3.md`. Each section lists the points it argues, the artefacts it leans on, and the length budget. Useful as a planning view — the full prose lives in v1/v2/v3.

---

# [Title — TBD]

**Gustav Skarman** — gskarman@kth.se · DM128X Examensarbete inom Medieteknik, grundnivå · KTH · VT 2026 · Supervisor: Jarmo Laaksolahti.

---

## Sammanfattning *(Swedish, ~300 ord)*

- Six-element structure (relevance, theory, RQ, method, result, discussion-back-to-problem) mirrored with the EN abstract.
- Closes on the policy-is-system-prompt-is-classifier equivalence.

## Abstract *(English, ~300 words)*

- Same six elements as Sammanfattning, paragraph-for-paragraph.

---

## 1. Introduction (~1 page)

- LLMs are everywhere in text generation; AI-vs-human is now an everyday call for educators, editors, reviewers.
- Existing detectors return a number, not a reason — useless for the human who has to make the call.
- This thesis treats the LLM as the detector and holds it to a higher bar: the explanation must be faithful to the actual decision, not a plausible post-hoc rationalisation (Madsen 2024, Turpin 2023).
- Operating point: *don't falsely accuse; when you do, give the reason*. Motivates F0.5 + the explanation-faithfulness frame.
- The two-phase induce-policy-then-run-policy technique is the same pattern used in industry VC deal-sourcing (Harmonic / SignalFire Beacon / EQT Motherbrain) — same auditability shape, no peer-reviewed work has applied it to AI-text detection. This thesis tests whether the pattern transfers.

---

## 2. Background (~1–1.5 pages)

### 2.1 LLMs and token probabilities *(Theory)*

- A *token* is a sub-word piece; per-token log-probability is a free measurement of how expected each token was.
- *Single-token classification*: ask a yes/no question, generate one token, read `logp(yes) − logp(no)` as a continuous decision margin.
- DetectGPT family reads the *generator's* log-probabilities; this thesis reads the *classifier's* — same object, different question.

### 2.2 Related work in AI-text detection

- Three families: statistical (DetectGPT [9], Fast-DetectGPT [1], GLTR [3]), supervised (fine-tuned RoBERTa via MGTBench [4, 11]), watermarking [6].
- Watermarking out of scope (requires control over generation).

### 2.3 Faithfulness of LLM explanations *(new — needed for RQ2)*

- Faithfulness defined behaviourally (Jacovi & Goldberg [5]), not by plausibility.
- Turpin et al. [12] / Madsen et al. [8] / Lanham et al. [7]: default-prompted LLM self-explanations are often unfaithful.
- This thesis pre-empts the critique structurally: *policy = system prompt = classifier*, so explanation is faithful by construction.

### 2.4 Purpose and research questions

- **RQ1**: How does an LLM-based single-token classifier compare to established baselines (DetectGPT, fine-tuned RoBERTa) on HC3?
- **RQ2**: Can the classification decision be decomposed into human-interpretable explanations, and how faithful are those explanations to the model's actual decision mechanism?

---

## 3. Method (~0.5 page)

### 3.1 Data

- HC3 English subset; 60/20/20 train/val/test with `seed=42`, `min_chars=32`. Splits SHA frozen at `5393e028…`. Test never seen during induction or calibration.

### 3.2 Two-phase classifier *(D1, D2, D3)*

- **Phase 1 — induction**: proposer LLM reads ~10–20 labelled examples, writes a ~20-line natural-language policy. Score on held-out subset using F0.5; misclassified examples fed back; revisions accepted iff F0.5 improves; stop at plateau (Δ<0.005×3) or 30 iters.
- **Phase 2 — calibration**: frozen policy used as system prompt; `logp(yes) / logp(no) / logp(other)` collected over val; small calibrator (T1 threshold or T2 logistic) fit to maximise F0.5; reported on test.

### 3.3 Three-class output *(D5)*

- Outcomes: `yes` (AI), `no` (human), `other` (refusal / off-format / hedge). `other` counts as wrong in scoring and as a calibration feature.

### 3.4 Faithfulness evaluation *(D7)*

- Same-policy consistency: 5 re-samples at T=0, expect identical output.
- Policy ablation: same text under (best induced, empty, adversarially-inverted) — label and `logp(yes) − logp(no)` should track policy content.
- Whole-text only for Inl. 5; sentence-level / feature-level deferred.

### 3.5 Models and baselines *(D8, D10)*

- Gemma 4 E4B (iteration loop) and Gemma 4 31B (final-quality runs), both via Ollama.
- DetectGPT and fine-tuned RoBERTa cited from published HC3 results, not re-run.

### 3.6 Metrics

- Primary **F0.5** (precision-weighted, β=0.5). Secondary AUROC, ECE. Per-domain breakdown over the six HC3 subsets.

---

## 4. Results (~4–5 pages)

### 4.1 Detection performance — RQ1

- Headline table: F0.5 / precision / recall / AUROC / ECE for E4B (default), E4B (induced policy + T2 calibration), 31B (default), DetectGPT (literature), RoBERTa (literature).
- Headline numbers: E4B + induced policy + T2 calibration on n=4000 test → F0.5 = **0.934**, AUROC = **0.982**, ECE = **0.013**. Default-prompt 31B at n=1000 → F0.5 = 0.977.
- Per-domain breakdown (E4B / 31B, n=200 each subset).
- Calibration story at scale: T2 measurably moves the operating point (P 0.913→0.943, R 0.964→0.898) and ECE 74% (0.050→0.013); at n=200 the same calibrator was a no-op on hard predictions.
- Figure 1: ROC curve + reliability diagram for E4B+policy+calibration on n=4000 test (`logs/runs/2026-04-26T19-07-51_137899/`).

### 4.2 Policy induction trajectory

- Run `2026-04-26T17-42-47_3d67db`, pool=30, scoring=500, max_iters=30 with plateau early-stop. Winner at iter 1, F0.5 = 0.956 on n=500 val. Healthy 0.024 drop from the n=200 winner's 0.980 — small-sample bias correction.
- Trajectory table (iter 0–6) + Figure 2 (`logs/policies/2026-04-26T17-42-47_3d67db.png`).
- Full verbatim policy text (~150 words). Notable additions vs predecessor: explicit transitional phrases ("First and foremost", "In addition", "Furthermore") as AI markers; "EDIT," self-corrections as a human signal.

### 4.3 Faithfulness ablation — RQ2

- n=100 HC3 test, three policies (best, empty, inverted). Run `2026-04-26T20-23-43_2f80b2`.
- Per-policy F0.5: best 0.969, empty 0.965, inverted 0.242.
- Pairwise: best_vs_inverted **Δlabel = 0.490, mean Δ(lp_yes − lp_no) = +9.565 nats**. Far above the figurehead threshold; meets behavioural-faithfulness criterion.
- Figure 3: histogram of per-example margin shift (`logs/runs/2026-04-26T20-23-43_2f80b2/`).

---

## 5. Discussion (~2 pages)

### 5.1 Summary

- One paragraph linking RQ1 + RQ2 + headline numbers. Mirrors the abstract.

### 5.2 Relation to prior work

- DetectGPT / RoBERTa numbers from literature, not re-run; comparison sensitive to HC3 subset, source LLM, threshold convention.
- Conceptual closest: ProTeGi [10] (algorithmic precedent for D3) and Hypotheses-to-Theories [15] (induced rules become the inference prompt — closest to D1/D2).
- No peer-reviewed paper has applied the loop specifically to AI-text detection; that's the contribution.

### 5.3 Method discussion

- F0.5 vs F1 vs accuracy vs Youden's J: F0.5 chosen because false accusations are more costly than missed detections; the use case dictates the operating point.
- Three-class output: was inert on the test set but earned its keep during induction by surfacing refusals as scoring failures.
- Policy induction vs OPRO / ProTeGi / DSPy / HtT: textbook proposer/scorer/accept-reject loop; the thesis-specific choice is *natural-language policy retained as the explanation*.

### 5.4 Threats to faithfulness — pre-empting Madsen et al.

- Madsen 2024 / Turpin 2023: default-prompted LLM self-explanations often unfaithful.
- Defence is structural: policy *is* the system prompt *is* the classifier — there is no separate self-explanation step that could diverge.
- §4.3 ablation supplies the empirical evidence: 0.490 label-flip rate + 9.6 nat margin shift. The defence holds for the induced policy explanation only; any free-text rationalisation generated alongside would inherit the Madsen risk.

### 5.5 Future work

- Finer-grained ablations (sentence-level, feature-level).
- Cross-model comparison (gpt-oss 20B, Qwen2.5 32B; deferred from D8).
- Human evaluation of explanation usefulness.
- Adversarial robustness — induced policies enumerate a finite set of surface features; an aware writer can rewrite around them. Distinct property from faithfulness, would need layered detectors / periodic re-induction / human-in-the-loop on borderline cases for practical deployment.

### 5.6 Contribution

- An open-source LLM-based single-token detector at F0.5 = 0.934 / AUROC = 0.982 / ECE = 0.013 on HC3 `all` n=4000 test (E4B + induced policy + calibration), within ~0.04 F0.5 of the four-times-larger 31B default-prompt baseline and meaningfully better calibrated.
- A policy-induction protocol whose output doubles as the system prompt — explanation faithful by construction.
- A quantitative faithfulness measurement (0.490 label-flip rate, +9.6 nat margin shift) that supplies the behavioural evidence Madsen's critique demands.

---

## 6. Conclusion *(≤300 words)*

- Qualified yes: small open-weights model + ~150-word induced policy + small calibrator reaches F0.5 = 0.934 on the n=4000 test split, within ~0.04 of a four-times-larger baseline and better calibrated.
- 49% label-flip + 9.6 nat margin shift between best and inverted policies = direct behavioural evidence that the policy drives the decision (not a post-hoc rationalisation in the Madsen sense).
- The rule the model uses to flag a text *is* the rule the human reading the explanation will see.
- Open: usefulness of the explanation to a human reader, portability across model families.

---

## References *(ACM format — target 10–15)*

15 entries currently in the draft. Re-verify ACM formatting and arXiv IDs before submission. Notable: Bao et al. 2024 (Fast-DetectGPT), Mitchell et al. 2023 (DetectGPT), He et al. 2024 (MGTBench), Pudasaini et al. 2025 (GenAIDetect), Jacovi & Goldberg 2020, Madsen et al. 2024, Turpin et al. 2023, Lanham et al. 2023, Pryzant et al. 2023 (ProTeGi), Zhu et al. 2024 (HtT), Kirchenbauer et al. 2023 (watermarking), Gehrmann et al. 2019 (GLTR).

---

## Pre-flight checklist before submitting Inl. 5

- [ ] Page count (Henrik's clarification: ~10,000 words ≈ ~20 pages in the new ACM template; the older 10–12-page line is template-version-specific)
- [ ] Sammanfattning + Abstract present, ~300 words each
- [ ] Background cites ≥10 peer-reviewed refs
- [ ] Method matches `decisions.md` D1–D11 + scale-up D12 + resilience D13
- [ ] Results section uses real numbers, not placeholders — n=4000 calibration, n=500 induction, n=100 faithfulness
- [ ] §4.1 Table 1 baseline cells filled (DetectGPT [9] + RoBERTa MGTBench [4]) or non-comparability flagged
- [ ] §4.1 Figure 1 generated from `logs/runs/2026-04-26T19-07-51_137899/`
- [ ] §4.2 Figure 2 — induction trajectory plot at `logs/policies/2026-04-26T17-42-47_3d67db.png` (auto-generated)
- [ ] §4.3 Figure 3 generated from `logs/runs/2026-04-26T20-23-43_2f80b2/`
- [ ] Discussion answers RQ1 and RQ2 explicitly
- [ ] Conclusion ≤300 words
- [ ] References in ACM format, 10–15 entries, all peer-reviewed
- [ ] No long verbatim quotes (the §4.2 policy quote is a self-produced artefact, not a third-party quote)
- [ ] File named `Gustav-Skarman_<title>.pdf` per Kexjobbsspecifikation §0
- [ ] Uploaded to Canvas before **Thu 2026-04-30, 19:00 CEST** (not 23:59)
