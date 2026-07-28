# Inlämning 5 — draft skeleton

> **Purpose.** A section-by-section scaffold of the thesis report, ready to paste into [Kexjobbsuppsats](https://docs.google.com/document/d/1-0yFrBHGKaZgofwFds5WdUvnNcwN8nQF3hrO-CHesyw/edit). Where content already exists in [Kexjobbsspecifikation V2](https://docs.google.com/document/d/1vgsi4AlcWnbH0WxjGzK6R-DepT8yPNjx6-centHyMm0/edit), it has been ported verbatim and **flagged with `[FROM V2]`**. Where the spec is out of date relative to `decisions.md` (frozen 2026-04-23), the corrected text is here and **flagged `[UPDATED — see decisions.md DN]`**. Empty placeholders are explicitly tagged with `[TODO]` so nothing slips through.
>
> **Format reality check.** Inl. 5 is graded on content, not formatting. Stay in Google Docs through the draft, but match the Kexjobbsramar lengths and headings. Migration to the ACM Word/LaTeX template is a separate task before Inl. 6 (7 May).
>
> **Length budget (10–12 pages total).** Abstract (SV+EN, ~300 each) · Intro 1p · Background 1–1.5p · Method 0.5p · Results 4–5p · Discussion 2p · Conclusion ≤300 words · References ~1p.

---

# [Title — TBD]

**Gustav Skarman** — gskarman@kth.se
DM128X Examensarbete inom Medieteknik, grundnivå · KTH · VT 2026
Supervisor: Jarmo Laaksolahti

---

## Sammanfattning *(Swedish abstract, ~300 ord — write last)*

[TODO] Skriv sist, parallellt med EN abstract. Sex meningar minimum, en per punkt: (1) varför problemet är relevant, (2) vilken teori/relaterad forskning arbetet vilar på, (3) tydlig forskningsfråga, (4) använd metod, (5) resultat, (6) diskussion som kopplar tillbaka till problembeskrivningen.

## Abstract *(English abstract, ~300 words — write last)*

[TODO] Same six elements. Mirror the Sammanfattning closely so the bilingual reader sees identical structure.

---

## 1. Introduction (~1 page)

**`[FROM V2 — light edits flagged]`**

The rapid adoption of Large Language Models (LLMs) in text generation has created an urgent need for reliable detection methods across academia, publishing, and content moderation. While modern detectors achieve high accuracy on benchmarks, they typically expose only a confidence score — *"82% likely AI-generated"* — without communicating *why*. This is a usable answer for engineers and a useless one for the educators, editors, and reviewers whose decisions actually depend on it.

This thesis investigates whether LLMs can themselves be made to do this work *transparently*: classify text as AI-generated or human-written, and produce explanations that can be audited and trusted. The contribution is twofold — measuring how an LLM-based single-token classifier compares to established detection baselines, and assessing whether explanations derived from a small set of inducible rules ("policy") are *faithful* to the model's actual decision mechanism, not merely plausible.

`[UPDATED — see decisions.md context block, 2026-04-23]` The use case treated as the operating point throughout this work is *"do not falsely accuse; when we do accuse, give the reason why."* This framing makes precision more costly than recall to give up, motivates the choice of F0.5 as the primary metric, and aligns the explanation produced with the rule that triggered it.

**[TODO]** One paragraph positioning the work — Gustav's prior production experience with LLM classifiers (founder/company identification at the VC firm) and how that shapes the methodological choice. Land the problem statement explicitly in the closing sentence of this section.

---

## 2. Background (~1–1.5 pages)

### 2.1 LLMs and token probabilities *(Theory)*

**`[FROM V2 — light edits]`** LLMs generate text by sampling tokens from learned probability distributions; the per-token log-probability of the chosen token is, in principle, a direct measurement of how "expected" that token was under the model. Because this signal is available at no extra inference cost, every detection method that relies on it is operating on the same underlying object.

**[TODO]** ~3–4 sentences defining: token, log-probability, single-token classification (yes/no), and the difference between using the *generation-model's* log-probabilities (DetectGPT) and the *classifier-model's* log-probabilities (this thesis). This is where the reader learns the mental model.

### 2.2 Related work in AI-text detection

**`[FROM V2 — preserve all citations]`** Existing detection approaches fall into three families.

**Statistical methods** exploit that LLM-generated text occupies regions of high log-probability under the source model. *GLTR* visualised token-rank histograms for human inspection [3]; *DetectGPT* introduced probability-curvature as a zero-shot signal that needs no extra training [9]; *Fast-DetectGPT* improved its speed and accuracy via conditional curvature [1].

**Supervised classifiers** — most prominently fine-tuned RoBERTa — learn to distinguish human from machine text from labelled examples. *MGTBench* showed these methods are strong in-distribution and brittle across models and domains [4, 11].

**Watermarking** embeds detectable statistical patterns at generation time [6]. It cannot be applied to text produced without watermarking-aware decoding, so it does not address the operational case treated in this thesis.

### 2.3 Faithfulness of LLM explanations *(new — needed for RQ2)*

`[UPDATED — V2 only cited Turpin 2023; decisions.md D7 lists the full set]` The standard framing — faithfulness as behaviour-under-perturbation rather than human-judged plausibility — is from Jacovi & Goldberg [5]. Turpin et al. [12] showed Chain-of-Thought rationales can be systematically unfaithful; Lanham et al. [7] developed truncation/corruption/paraphrase ablation protocols for measuring this; Madsen et al. [8] specifically tested LLM self-explanations and found significant unfaithfulness in default prompting. This thesis must pre-empt that critique. The defence is structural: when the policy *is* the system prompt and the system prompt *is* the classifier, the explanation is faithful by construction (see §3 and Discussion).

### 2.4 Purpose and research questions

**`[FROM V2 — preserve verbatim]`** This thesis investigates whether LLMs can detect AI-generated text while producing faithful, human-interpretable explanations.

> **RQ1.** How does an LLM-based single-token classifier compare to established baselines (e.g., DetectGPT, fine-tuned RoBERTa) on standard AI-text detection benchmarks?
>
> **RQ2.** Can the classification decision be decomposed into human-interpretable explanations, and how faithful are those explanations to the model's actual decision mechanism?

---

## 3. Method (~0.5 page)

> ⚠️ `[UPDATED — V2 spec says "F1 primary" and "gpt-oss/Qwen2.5"; both have changed in decisions.md (2026-04-23). Use the text below, not the V2 wording.]`

### 3.1 Data

HC3 (Hello-SimpleAI/HC3), English subset only. Frozen 60/20/20 train/val/test split with seeded sampling (seed and hashed config recorded per `logs/runs/<run_id>/config.yaml`). The test split is never seen during induction or calibration.

### 3.2 Two-phase classifier *(D1, D2, D3)*

**Phase 1 — policy induction.** A proposer LLM reads ~10–20 labelled training examples and emits a natural-language policy (~20 lines, prose form). The policy is scored on a held-out training subset (n=100–500) using F0.5. Misclassified examples are fed back; the proposer revises. A revision is accepted iff F0.5 improves; iteration stops at a `Δ<0.005` plateau over the last three accepted edits or at 30 iterations.

**Phase 2 — log-probability calibration.** The frozen policy is used as the classifier's system prompt. For each example we capture the log-probabilities of the first generated token over `yes`, `no`, and `other`. A small calibration model (threshold on `logprob(yes) − logprob(no)`, or a logistic regression over the three log-probs) is fit on the val split to maximise F0.5. This is the test-time classifier.

### 3.3 Three-class output *(D5)*

The classifier emits one of `{yes, no, other}` per example. `other` covers refusals, off-format, and explicit hedges; it is a first-class outcome in both phases (counts as wrong in Phase 1 scoring, appears as a feature in Phase 2 calibration).

### 3.4 Faithfulness evaluation *(D7)*

Two whole-text tests per example: *(i) same-policy consistency* (5 re-samples at T=0; expect identical output), and *(ii) policy ablation* — same text run under three policies (best induced, empty, adversarially-inverted). The label and `logprob(yes) − logprob(no)` should track the policy content; if they don't, the policy is a figurehead and faithfulness is low. Sentence-level and structured-feature ablations are deferred past Inl. 5.

### 3.5 Models and baselines *(D8, D10)*

Models this cycle: **Gemma 4 E4B** (iteration loop) and **Gemma 4 31B** (final-quality runs), both run locally via Ollama. Baselines (DetectGPT, fine-tuned RoBERTa) are *cited from published HC3 results, not re-run*; the comparability of each baseline's reported split is documented in §5.

### 3.6 Metrics

Primary: **F0.5** (precision-weighted, β=0.5). Secondary: AUROC and ECE (after calibration), plus per-domain breakdown across HC3's six subsets.

---

## 4. Results (~4–5 pages)

> ⚠️ **Highest-risk section for Inl. 5.** This subsection structure assumes the runs in `logs/runs/` produce usable numbers. If they don't, surface that to Jarmo *now*, not Wednesday night.

### 4.1 Detection performance — RQ1

[TODO] Table: F0.5, F1, precision, recall, AUROC, ECE for Gemma 4 E4B (n=1000) and Gemma 4 31B (where available), against literature numbers for DetectGPT and fine-tuned RoBERTa. Per-domain breakdown across the six HC3 subsets.

[TODO] Figure: ROC curve with the F0.5-optimal operating point highlighted. Reliability diagram for ECE.

### 4.2 Policy induction trajectory

[TODO] Figure: F0.5 vs iteration, accepted vs rejected edits marked. One short table showing 2–3 representative policy edits with the misclassified examples that drove them. This is the most novel-looking figure in the report — keep it.

### 4.3 Faithfulness ablation — RQ2

[TODO] Table: label-flip rate and Δ`logprob(yes)−logprob(no)` distribution under (best, empty, inverted) policies, n=100 test subsample. The headline number is *"the inverted-policy condition flipped X% of labels"* — this is the faithfulness claim.

[TODO] Same-policy consistency: confirm 5/5 identical outputs at T=0 (one sentence, footnote if non-determinism appears).

---

## 5. Discussion (~2 pages)

### 5.1 Summary

[TODO] Open with one paragraph linking purpose, RQ1, RQ2, and headline results. Mirrors the abstract.

### 5.2 Relation to prior work

[TODO] Compare against DetectGPT, RoBERTa baselines from §2.2. Flag any non-comparable axes (different AI source model, different HC3 subset, different threshold convention).

### 5.3 Method discussion

[TODO] Was F0.5 the right operating point for this thesis (vs F1 / accuracy / Youden's J)? Was the three-class output worth its cost? Was policy-induction the right framing relative to OPRO / ProTeGi / DSPy / HtT — all of which solve adjacent problems? Be explicit about what would move the contribution to a stronger position.

### 5.4 Threats to faithfulness — pre-empting Madsen et al.

[TODO] Madsen 2024 [8] showed default-prompted LLM self-explanations are often unfaithful. The defence here is structural — *the policy is the system prompt is the classifier*, so an explanation that names a triggered rule is grounded in the same object that drives the prediction. Acknowledge that this defence holds for the *induced* explanation only; any free-text rationalisation generated alongside is subject to the same Madsen critique.

### 5.5 Future work

[TODO] Sentence-level and feature-level ablations (deferred from Inl. 5), gpt-oss 20B + Qwen2.5 32B cross-model comparison (deferred from D8), human evaluation of explanation usefulness (deferred from Q14).

### 5.6 Contribution

[TODO] State explicitly: a working LLM-based single-token detector achieving X on HC3, a policy-induction protocol whose output doubles as a faithful-by-construction explanation, and a small ablation result quantifying that faithfulness. Practical contribution: a detector that gives an auditor a reason, not just a percentage.

---

## 6. Conclusion *(≤300 words)*

[TODO] One paragraph. State whether the answer to *"can LLMs reliably and explainably classify AI-generated text"* is yes, qualified-yes, or no, and on what evidence. Close with the single sentence you most want the reader to remember.

---

## References *(ACM format — target 10–15)*

1. Bao, G., Zhao, Y., Teng, Z., Yang, L., and Zhang, Y. 2024. Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature. In *Proc. ICLR 2024*.
2. Gao, T., Fisch, A., and Chen, D. 2021. Making Pre-trained Language Models Better Few-shot Learners. In *Proc. ACL/IJCNLP 2021*.
3. Gehrmann, S., Strobelt, H., and Rush, A. 2019. GLTR: Statistical Detection and Visualization of Generated Text. In *Proc. ACL 2019 System Demonstrations*.
4. He, X., Shen, X., Chen, Z., Backes, M., and Zhang, Y. 2024. MGTBench: Benchmarking Machine-Generated Text Detection. In *Proc. ACM CCS 2024*.
5. Jacovi, A. and Goldberg, Y. 2020. Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness? In *Proc. ACL 2020*.
6. Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., and Goldstein, T. 2023. A Watermark for Large Language Models. In *Proc. ICML 2023*.
7. Lanham, T., et al. 2023. Measuring Faithfulness in Chain-of-Thought Reasoning. *arXiv:2307.13702*.
8. Madsen, A., et al. 2024. Are Self-Explanations from LLMs Faithful? In *Findings of ACL 2024*.
9. Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., and Finn, C. 2023. DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature. In *Proc. ICML 2023*.
10. Pryzant, R., et al. 2023. Automatic Prompt Optimization with "Gradient Descent" and Beam Search. In *Proc. EMNLP 2023*. *(ProTeGi — algorithmic precedent for §3.2 Phase 1.)*
11. Pudasaini, S., Miralles, L., Lillis, D., and Llorens Salvador, M. 2025. Benchmarking AI Text Detection: Assessing Detectors Against New Datasets, Evasion Tactics, and Enhanced LLMs. In *Proc. 1st Workshop on GenAI Content Detection (GenAIDetect), COLING 2025*.
12. Turpin, M., Michael, J., Perez, E., and Bowman, S. R. 2023. Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. In *Proc. NeurIPS 2023*.
13. Wang, Q. and Li, H. 2025. On Continually Tracing Origins of LLM-Generated Text and Its Application in Detecting Cheating in Student Coursework. *Big Data and Cognitive Computing* 9, 3 (2025), 50.
14. Zhao, Z., Wallace, E., Feng, S., Klein, D., and Singh, S. 2021. Calibrate Before Use: Improving Few-shot Performance of Language Models. In *Proc. ICML 2021*.
15. Zhu, Z., et al. 2024. Hypotheses-to-Theories: Inducing Rules with LLMs. In *Proc. ICLR 2024*. *(Closest conceptual match for §3.2 — induced rules become the classifier.)*

---

## Pre-flight checklist before submitting Inl. 5

- [ ] Page count 10–12 pages (in Google Docs ≈ a tight count; verify after migrating to ACM template if there's time)
- [ ] Abstract written in **both** SV and EN, ~300 words each
- [ ] Background cites ≥10 peer-reviewed refs
- [ ] Method matches `decisions.md` (F0.5, three-class, two-phase, policy ablation — *not* V2's F1/binary/feature-masking wording)
- [ ] Results section has actual numbers, not placeholders
- [ ] Discussion answers RQ1 and RQ2 explicitly
- [ ] Conclusion ≤300 words
- [ ] References in ACM format, 10–15 entries, all peer-reviewed
- [ ] No long verbatim quotes; paraphrase + cite
- [ ] File named `Gustav-Skarman_<title>.pdf` (per Kexjobbsspecifikation instruction §0)
- [ ] Uploaded to Canvas before **Thu 30 Apr 2026, 19:00 CEST** (not 23:59)
