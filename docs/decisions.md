# Method & design decisions

Append-only ADR-style log of thesis-level method and architecture decisions.

- For per-run notes see `logs/RUNS.md`.
- For deadline / course facts see `docs/operating-brief.md`.
- For report-level structure see `docs/outline.md`.

Each decision block: **context → decision → consequences → applicable files**. Do not edit past decisions — supersede with a new dated block.

---

## 2026-04-23 — v1 method frozen

**Context.** Inlämning 5 (first draft to supervisor) due 2026-04-30 19:00 CEST. Solo author (no co-author). Use-case framing: *"don't falsely accuse; when we do accuse, give the reason why."* This operating point dictates F0.5 as the primary metric (precision-weighted) and a policy-based classifier whose decision is explainable by construction (the policy *is* the classifier).

---

### D1. Two-phase architecture: policy induction → log-prob calibration

- **Phase 1 — induction (no log-probs).** LLM proposes a natural-language policy from a few labeled examples. Policy is scored on a training split via F0.5 on hard yes/no/other outputs. Misclassified examples are fed back; the LLM revises. Accept if F0.5 improves. Iterate to plateau. Output: one frozen "best policy" string.
- **Phase 2 — calibration (with log-probs).** Freeze the policy. Run the classifier with that policy over training data; extract per-example log-probs for `yes`, `no`, `other` tokens. Fit a small ML layer (threshold on `logprob(yes) − logprob(no)`, or logistic regression over the three features) to optimize F0.5 on a held-out split. This is the final classifier used at test time.

**Rationale.** Phase 1 optimizes the *policy* (language). Phase 2 optimizes the *decision boundary* (numerical). The policy remains faithful-by-construction (it *is* the system prompt), while calibration unlocks AUROC / ECE / a tunable operating point without altering the explanation shown to the user.

**Applicable files.**

- `code/src/aitd/policy.py` *(new)* — induction proposer/scorer loop.
- `code/src/aitd/calibration.py` *(new)* — threshold / logistic fit over log-prob features.
- `code/src/aitd/classifier.py` — accept a `system_prompt` argument; emit a three-way hard label and the three log-probs.
- `code/src/aitd/ollama_client.py` — expose log-probs per request (see D11).

---

### D2. Policy form: ~20-line natural-language system prompt

Free-form natural-language rules / heuristics concatenated into a single system-instruction string. No enforced structure (no JSON, no numbered rule schema) — the LLM writes prose that works. Soft cap ≈ 20 lines / 200–400 tokens to keep inference cheap and the policy legible in the thesis.

**Rationale.** Mirrors Gustav's working pattern from a prior LLM-as-classifier system (VC founder/company identification). Maximum readability for the Explanations section. Free-form prose is what LLMs both produce and consume best.

**Faithfulness corollary.** Because *policy = system prompt = classifier*, an explanation of why a text was flagged is always: "because this policy said to." Swapping the policy and observing changes in label and log-prob gives a direct faithfulness measurement (see D7).

---

### D3. Induction algorithm: proposer → scorer → accept/reject

1. **Seed.** Feed ~10–20 labeled examples (balanced AI / human) to the LLM; ask for a first policy.
2. **Score.** Run the candidate policy over a held-out training subset (n=100–500); compute F0.5.
3. **Propose variant.** Feed the LLM: current policy + 10–20 *misclassified* training examples + request a revised policy.
4. **Accept iff** F0.5_new > F0.5_current. Otherwise discard.
5. **Stop** when F0.5 plateaus (`Δ < 0.005` over last 3 accepted iterations) or iteration budget is exhausted (default 30).

Keep a full trajectory log: `{iter, policy_text, F0.5, precision, recall, accepted}` per step. That log becomes a figure in the Results section.

**Prior art to cite.** Pattern is the "LLM-as-policy-writer" loop used in VC deal-sourcing and founder scoring. Direct algorithmic matches in the literature: **ProTeGi** (Pryzant 2023) — minibatch error → textual gradient → prompt edit; **HtT** (Zhu 2024) — induced NL rules become the classifier; **TextGrad** (Yuksekgonul 2024) — per-example NL "gradients". Broader neighbours: APE, OPRO, DSPy / MIPROv2, Promptbreeder, EvoPrompt, PromptWizard. Full bibliography at the end of this document.

---

### D4. Scoring: F0.5 on held-out training set

Primary score during induction: **F0.5** (precision-weighted, β=0.5). Rationale: the thesis use-case penalizes false accusations (human flagged as AI) more than missed detections.

**Split.** HC3 → 60 % train (scoring for induction) / 20 % val (calibration threshold fit) / 20 % test (final reported numbers). Seeded once; frozen in `configs/`. Test split is never seen during induction or calibration.

---

### D5. Three-class output: yes / no / other

Classifier emits one of three labels on each example:

- `yes` — AI-generated (label 1).
- `no` — human-written (label 0).
- `other` — refusal, off-format, or explicit hedge (label -1).

`other` is a first-class class in both phases: it counts as wrong in Phase 1 scoring, and it appears as a feature in Phase 2 calibration.

**Rationale.** Avoids forcing the model to guess when the policy genuinely cannot decide; surfaces policy failure modes during induction; gives the calibration layer a real uncertainty signal.

---

### D6. Calibration features: log-probs of yes / no / other

At Phase 2, for each example we capture `logprob(yes)`, `logprob(no)`, `logprob(other)` over the first generated token (or, if a label spans multiple tokens, the sum of token log-probs for the canonical form).

Candidate calibration models (pick best on val set):

- **T1.** Single threshold on `logprob(yes) − logprob(no)`, with an `other`-penalty.
- **T2.** Logistic regression over `{logprob(yes), logprob(no), logprob(other), logprob(yes) − logprob(no)}`.

Operating-point choice: threshold tuned to maximize F0.5 on val.

---

### D7. Faithfulness test: full-text, policy-swap

Two tests per text:

- **Same-policy consistency.** Same text × same policy × 5 re-samples at T=0 → expect identical output. Flags non-determinism in the stack.
- **Policy ablation.** Same text under three policies — (a) best induced policy, (b) empty system prompt, (c) adversarially-inverted policy (e.g. "assume all texts are human"). Measure Δ in predicted label and in `logprob(yes) − logprob(no)`.

A policy is *faithful* to the model's decision if the label/log-prob tracks the policy content. If the model predicts the same way regardless of policy, the policy is a figurehead and faithfulness is low.

**Granularity for Inl. 5.** Whole-text only. Sentence-level and structured-feature ablations deferred; only run if time permits after the first draft lands.

**Framing & refs.** Faithfulness is defined behaviorally (Jacovi & Goldberg 2020), not via plausibility. The ablation protocol mirrors Lanham et al. 2023 ("Measuring Faithfulness in CoT Reasoning"). The thesis must pre-empt the Turpin et al. 2023 / Madsen et al. 2024 critique that LLM self-explanations can be systematically unfaithful — our faithful-by-construction framing (policy *is* the prompt) is the central defence.

---

### D8. Models: Gemma 4 E4B (iteration) + Gemma 4 31B (full runs)

- `gemma4:e4b` — dev/iteration loop, policy induction, fast feedback.
- `gemma4:31b` — full-quality runs for the Results section numbers.
- `gpt-oss:20b`, `qwen2.5:32b` — **kept in the operating brief** as listed methods but **not run this cycle**. Reserved for later cross-model comparison if time permits before Inl. 7.

**Compute assumption.** M4 Max + 64 GB RAM; 31B at n=1000 per split is tractable. Validation: schedule a 31B run early (Day 1–2) to confirm wall-clock before relying on it.

---

### D9. Metric: F0.5 primary; AUROC + ECE secondary

Reported in Results:

- **Phase 1 (hard-label).** Precision, recall, F0.5, F1, accuracy, per-domain breakdown.
- **Phase 2 (calibrated).** ROC curve + AUROC; reliability diagram + ECE; chosen operating point (F0.5-optimizing threshold) with its precision/recall.

**Open: Q9.** Presentation of the operating point — single chosen point in a table vs. full PR curve in a figure. Default plan: **both** (curve in Results figures, chosen point in summary table). Revisit after supervisor feedback on Inl. 5.

---

### D10. Baselines: cite from literature; do not re-run

DetectGPT (Mitchell et al. 2023) and fine-tuned RoBERTa numbers are pulled from published results on HC3 (or directly comparable benchmarks). RADAR (Hu et al. 2023) is the adversarial-training neighbour to acknowledge but not necessarily benchmark. We do not re-run any of them on our split.

Writing discipline: explicitly document which papers we cite, which HC3 split/subset they reported on, and why the comparison is fair. Flag any non-comparable axes (e.g. different AI source model) in the Discussion.

**Rationale.** Saves ~5–10 days of infra work; the thesis's contribution is policy induction + faithfulness, not a re-benchmarking of existing detectors.

---

### D11. Log-prob extraction: validate Ollama first; stay local

Validation spike on Day 1 before anything else depends on it:

- Preferred path: Ollama `generate` with a log-prob-returning option for Gemma 4.
- Fallback 1: `llama.cpp` / `llama-cpp-python` direct with Gemma 4 weights.
- Fallback 2: `vLLM` with Gemma 4 — heavier install, still local.
- **No** cloud APIs this cycle.

If all local paths fail for Gemma 4 specifically, fall back to the two-call trick (num_predict=1 twice, compare token scores) — doubles inference cost but preserves the design.

---

## Open questions

- **Q9 — Operating-point presentation.** Curve + highlighted point vs. single-point narrative. Default to both; revisit with supervisor feedback.
- **Q10-tech — Ollama log-prob extraction path.** Technical feasibility for Gemma 4 on M4 Max pending spike (Day 1).
- **Q14 — Human eval for faithfulness.** Defer decision until Inl. 6; automated-only for Inl. 5.

---

## Schedule — 7 days to Inlämning 5 (2026-04-30 19:00 CEST)

| Day | Date | Focus | Deliverable |
|---|---|---|---|
| 1 | Fri 24 Apr | Log-prob spike + n=1000 E4B baseline | `docs/logprob-spike.md` with a working call; first 1000-row E4B run on HC3 `all` |
| 2 | Sat 25 Apr | Per-domain + 31B runs | Per-split breakdown table on E4B; first 31B numbers (n=500 sample if full is slow) |
| 3 | Sun 26 Apr | Policy induction scaffolding | `aitd/policy.py` proposer + scorer; first 5-iteration induction loop on n=200 subsample |
| 4 | Mon 27 Apr | Policy induction full run | Final induced policy + trajectory figure (F0.5 vs iter); accepted/rejected count |
| 5 | Tue 28 Apr | Calibration + faithfulness | `aitd/calibration.py`; ROC + ECE numbers on val/test; policy-ablation table |
| 6 | Wed 29 Apr | Results + Discussion draft | 4–5 pp Results, 2 pp Discussion in ACM template |
| 7 | Thu 30 Apr | Abstract (SV + EN), Conclusion, References, submit | Canvas upload before 19:00 |

Intro / Background / Method are ported from `Kexjobbsspecifikation.pdf` in parallel with Days 4–7 (cheap text work; interleave with experiment wait-time).

---

## Literature landscape (2026-04-23 research pass)

**Dominant paradigm.** LLM-as-optimizer prompt search: the proposer LLM reads a trajectory of (instruction, score) pairs (OPRO) or per-example error critiques (ProTeGi / TextGrad), and emits a refined instruction. The scorer is a task metric on held-out labels. Across APE, OPRO, DSPy/MIPROv2, ProTeGi, TextGrad, Promptbreeder, EvoPrompt, and PromptWizard the shape is identical; the axes of difference are the critique signal (single score vs. error traces vs. NL gradients), the search operator (greedy / beam / evolutionary / Bayesian), and the target (instruction alone vs. instruction + few-shot demos).

**Most directly related to this thesis.**

1. **ProTeGi** (Pryzant et al. 2023) — minibatch errors → textual gradient → prompt edit. Closest algorithmic match for D3.
2. **HtT** (Zhu et al. 2024) — induced NL rules become the classifier. Closest conceptual match for D1+D2.
3. **Madsen et al. 2024** — "Are Self-Explanations from LLMs Faithful?" The evaluation critique we need to pre-empt in Discussion.

**Novelty angle (and its risk).** No peer-reviewed paper I can confirm applies the APE/OPRO/ProTeGi loop specifically to AI-generated-text detection. This is a plus for the thesis's contribution claim, and a risk: we must motivate the choice without a direct baseline in the same niche.

**VC deal-sourcing caveat.** Industry practice (Harmonic, SignalFire Beacon, EQT Motherbrain) is public only via blog posts / product pages — no peer-reviewed methodology paper exists to our knowledge as of 2026. Thesis should frame VC use as *reported industry practice*, cite grey literature, and note the absence of academic sources. Reviewers may object; flag this with the supervisor at Inl. 5.

**Confidence caveat.** Research pass ran in a sandbox without web access; URLs below reconstructed from training-cutoff memory (Jan 2026). Verify arXiv IDs before final citation — especially grey-lit entries.

---

## Bibliography

### Prompt-optimization / policy-induction methods

- **[1] APE — Large Language Models Are Human-Level Prompt Engineers.** Zhou et al., ICLR 2023. arXiv:2211.01910. Proposer/scorer loop with top-k resampling; the canonical reference. <https://arxiv.org/abs/2211.01910>
- **[2] OPRO — Large Language Models as Optimizers.** Yang et al., ICLR 2024. arXiv:2309.03409. LLM reads sorted (instruction, score) trajectory and proposes a better one. <https://arxiv.org/abs/2309.03409>
- **[3] DSPy / MIPROv2.** Khattab et al., ICLR 2024. arXiv:2310.03714 and follow-up arXiv:2406.11695. Declarative pipelines + Bayesian bootstrapping over instructions and demos. <https://arxiv.org/abs/2310.03714> · <https://arxiv.org/abs/2406.11695>
- **[4] TextGrad — Automatic "Differentiation" via Text.** Yuksekgonul et al., Nature 2024 / arXiv:2406.07496. Per-example NL gradients aggregated into prompt edits. <https://arxiv.org/abs/2406.07496>
- **[5] Promptbreeder.** Fernando et al., ICML 2024. arXiv:2309.16797. Evolutionary search over task prompts + mutation prompts. <https://arxiv.org/abs/2309.16797>
- **[6] EvoPrompt.** Guo et al., ICLR 2024. arXiv:2309.08532. GA/DE operators executed by an LLM. <https://arxiv.org/abs/2309.08532>
- **[7] PromptWizard.** Agarwal et al. (Microsoft Research), 2024. arXiv:2405.18369. Critique-and-synthesize loop — the most "policy-like" of the prompt optimizers. <https://arxiv.org/abs/2405.18369>
- **[8] ProTeGi — Automatic Prompt Optimization with "Gradient Descent" and Beam Search.** Pryzant et al., EMNLP 2023. arXiv:2305.03495. Direct precursor to TextGrad; the closest algorithmic match for D3. <https://arxiv.org/abs/2305.03495>

### Rule / policy induction as classifier

- **[9] Language Models as Inductive Reasoners.** Yang et al., EACL 2024. arXiv:2212.10923. LLM proposes NL rules from examples, then uses them to classify. <https://arxiv.org/abs/2212.10923>
- **[10] Hypotheses-to-Theories (HtT).** Zhu et al., ICLR 2024. arXiv:2310.07064. Induced rules retained if they generalize; retained rules become the inference prompt. Closest conceptual match for D1+D2. <https://arxiv.org/abs/2310.07064>
- **[11] Hypothesis Search.** Wang et al., ICLR 2024. arXiv:2309.05660. Propose-verify-refine over NL hypotheses for ARC-style tasks. <https://arxiv.org/abs/2309.05660>
- **[12] Self-Refine.** Madaan et al., NeurIPS 2023. arXiv:2303.17651. Canonical iterative self-critique/refine loop. <https://arxiv.org/abs/2303.17651>
- *[13] Tree-of-Rules / PRoMPT-based text-classification rule induction.* 2024 workshop papers (Singh et al. NAACL Findings 2024; Bhatia et al. EMNLP 2024). **Low-confidence on exact titles** — verify via forward-citation search from HtT before using.

### VC / founder-identification applications (grey literature)

- *[14] Harmonic AI blog — "How we use LLMs to find founders."* 2023–2024. **No verified stable URL** — search harmonic.ai/blog.
- *[15] SignalFire Beacon.* Public product materials (2023–2024) on ML-ranked founder signals.
- *[16] EQT Motherbrain.* medium.com/eqtventures / eqtgroup.com — platform with recent LLM augmentations.
- *[17] Correlation Ventures, Connetic, Specter.* Marketing materials on "AI-sourced deals"; no published methodology paper known as of 2026.
- **[18] Anthropic Cookbook — Classification with Claude.** Practical LLM-as-classifier reference. <https://docs.anthropic.com/>
- *[19] LangChain / LangSmith Prompt Hub.* Iterative prompt refinement tooling against eval sets.
- **[20] OpenAI Cookbook — Prompt optimization & Evals notebooks.** <https://github.com/openai/openai-cookbook>
- **Patents.** No granted patent confirmed as of early 2026 on "LLM-proposed heuristic policies iteratively refined against labeled founder outcomes." USPTO PatFT search left as a todo if Inl. 6 review flags the gap.

### Faithfulness evaluation

- **[21] Jacovi & Goldberg 2020 — Towards Faithfully Interpretable NLP Systems.** ACL 2020. arXiv:2004.03685. Canonical framing: faithfulness = behavior-under-perturbation, not plausibility. <https://arxiv.org/abs/2004.03685>
- **[22] Turpin et al. 2023 — Language Models Don't Always Say What They Think.** NeurIPS 2023. arXiv:2305.04388. CoT rationales can be systematically unfaithful — must pre-empt in the thesis. <https://arxiv.org/abs/2305.04388>
- **[23] Lanham et al. 2023 — Measuring Faithfulness in CoT Reasoning.** arXiv:2307.13702. Truncation/corruption/paraphrase ablations on rationales — protocol mirrors our D7 design. <https://arxiv.org/abs/2307.13702>
- **[24] Atanasova et al. 2023 — Faithfulness Tests for NL Explanations.** ACL 2023. arXiv:2305.18029. Counterfactual and input-reconstruction tests. <https://arxiv.org/abs/2305.18029>
- **[25] Madsen et al. 2024 — Are Self-Explanations from LLMs Faithful?** ACL 2024 Findings. arXiv:2401.07927. Newest canonical ref; directly tests LLM self-explanations via ablation. <https://arxiv.org/abs/2401.07927>

### Baselines (detection field)

- **[B1] DetectGPT.** Mitchell et al. 2023. Zero-shot detector using perturbation-based curvature of model log-probabilities. To cite as baseline.
- **[B2] RADAR.** Hu et al. 2023. Adversarial training framework for AI-text detection; the adjacent non-policy neighbour.
- **HC3 dataset.** Guo et al. 2023. The corpus used throughout this thesis.
