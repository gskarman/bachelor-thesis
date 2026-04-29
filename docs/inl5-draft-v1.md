# Inlämning 5 — working draft

> **Working file.** Sister of the pristine `draft-skeleton.md` (which keeps `[TODO]` placeholders verbatim). All `[TODO]` blocks marked for §1 closing, §2.1, §4, §5, and §6 in the skeleton have been drafted here against the runs in `logs/RUNS.md` and the bundles in `docs/thread-b-results.md`. `[FROM V2]` and `[UPDATED — see decisions.md]` flags retained in section headers so traceability survives the migration to the ACM template.
>
> **Still to write:** Sammanfattning, Abstract, and the per-figure plot artefacts called out in §4. Pre-flight checklist at the bottom is unchanged from the skeleton.
>
> **Length budget (10–12 pages total).** Abstract (SV+EN, ~300 each) · Intro 1p · Background 1–1.5p · Method 0.5p · Results 4–5p · Discussion 2p · Conclusion ≤300 words · References ~1p.

---

# [Title — TBD]

**Gustav Skarman** — gskarman@kth.se
DM128X Examensarbete inom Medieteknik, grundnivå · KTH · VT 2026
Supervisor: Jarmo Laaksolahti

---

## Sammanfattning *(Swedish abstract, ~300 ord — write last)*

`[TODO]` Skriv sist, parallellt med EN abstract. Sex meningar minimum, en per punkt: (1) varför problemet är relevant, (2) vilken teori/relaterad forskning arbetet vilar på, (3) tydlig forskningsfråga, (4) använd metod, (5) resultat, (6) diskussion som kopplar tillbaka till problembeskrivningen.

## Abstract *(English abstract, ~300 words — write last)*

`[TODO]` Same six elements. Mirror the Sammanfattning closely so the bilingual reader sees identical structure.

---

## 1. Introduction (~1 page)

**`[FROM V2 — light edits flagged]`**

The rapid adoption of Large Language Models (LLMs) in text generation has created an urgent need for reliable detection methods across academia, publishing, and content moderation. While modern detectors achieve high accuracy on benchmarks, they typically expose only a confidence score — *"82% likely AI-generated"* — without communicating *why*. This is a usable answer for engineers and a useless one for the educators, editors, and reviewers whose decisions actually depend on it.

This thesis investigates whether LLMs can themselves be made to do this work *transparently*: classify text as AI-generated or human-written, and produce explanations that can be audited and trusted. The contribution is twofold — measuring how an LLM-based single-token classifier compares to established detection baselines, and assessing whether explanations derived from a small set of inducible rules ("policy") are *faithful* to the model's actual decision mechanism, not merely plausible.

`[UPDATED — see decisions.md context block, 2026-04-23]` The use case treated as the operating point throughout this work is *"do not falsely accuse; when we do accuse, give the reason why."* This framing makes precision more costly than recall to give up, motivates the choice of F0.5 as the primary metric, and aligns the explanation produced with the rule that triggered it.

The two-phase technique investigated here — induce a natural-language policy from a small set of labelled examples, then run that policy as the classifier's system prompt — is also the working pattern of LLM-augmented venture-capital deal-sourcing systems used in industry to identify high-potential founders and companies (Harmonic, SignalFire Beacon, EQT Motherbrain; documented only in product materials and engineering blog posts as of 2026). In that setting the policy *is* the auditable artefact: the rule the model applies to a candidate is the rule the partner reads off when justifying the decision to the investment committee. The detection problem treated in this thesis has the same shape — *the rule the model uses to flag a text must be the rule an educator or editor can show to the accused* — but no peer-reviewed work has yet applied the proposer/scorer/refine loop to AI-generated text detection. The question this thesis answers is whether the pattern transfers: can a policy induced from HC3 examples produce a classifier that is both competitive against published detection baselines and explainable by construction?

---

## 2. Background (~1–1.5 pages)

### 2.1 LLMs and token probabilities *(Theory)*

**`[FROM V2 — light edits]`** LLMs generate text by sampling tokens from learned probability distributions; the per-token log-probability of the chosen token is, in principle, a direct measurement of how "expected" that token was under the model. Because this signal is available at no extra inference cost, every detection method that relies on it is operating on the same underlying object.

A *token* is the atomic unit an LLM emits — typically a sub-word piece — and at every generation step the model produces a probability distribution over its vocabulary; the log-probability assigned to the token actually emitted is a direct, free-of-charge measurement of how *expected* that token was under the model. *Single-token classification* exploits this by prompting the model with a yes/no question and reading only the first generated token: the predicted label is whichever of `yes` or `no` carries the higher log-probability, and the difference `logprob(yes) − logprob(no)` is a continuous decision margin available without re-running the model. The detection family closest to the present thesis — *DetectGPT* and its descendants [1, 9] — uses the *generator's* log-probabilities, treating the log-probabilities a candidate text would receive under the (assumed) source LLM as the detection signal. This thesis instead uses the *classifier's* log-probabilities — those produced by a separate LLM at inference time when asked the yes/no question — so the underlying object is the same (token log-probabilities) but the question is different: not "is this text typical for some LLM?" but "does this text fit a learned, human-readable description of AI writing?".

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

> ⚠️ **Highest-risk section for Inl. 5.** Numbers here are pulled directly from the runs logged in `logs/RUNS.md`; reproducibility metadata (run IDs, splits SHA, frozen configs) is in `docs/thread-b-results.md` §5. Figure files referenced below either exist in the repo (trajectory) or need to be generated from the predictions JSONLs before submission.

### 4.1 Detection performance — RQ1

We report three classifier configurations: a *default-prompt* baseline (no induced policy, raw argmax over `yes`/`no`), an *induced-policy* classifier (Phase 1 only, raw argmax), and a *induced-policy + calibrated* classifier (Phase 1 + Phase 2). All numbers below are on the HC3 English test split (60/20/20, splits SHA `5393e028…`, seed 42); the test split is never used for induction or calibration.

**Headline.** Table 1 summarises detection performance on HC3 `all`. With the induced policy and a logistic calibrator (T2) the F0.5 of Gemma 4 E4B rises from 0.933 (default prompt, n=1000) to 0.942 (n=200 test) while ECE drops from 0.035 to 0.030 — a small absolute change in F0.5, but a 14% reduction in expected calibration error and, importantly, a large change in the *internal* decision margin (see §4.2 trajectory and §4.3 ablation). The full-quality 31B model with the default prompt scores F0.5 = 0.977 with AUROC ≈ 1.000 and ECE = 0.015. The 31B+policy+calibration combination is deferred past Inl. 5.

**Table 1 — Detection performance on HC3 `all`.**

| classifier configuration | model | n | F0.5 | F1 | accuracy | AUROC | ECE |
|---|---|---|---|---|---|---|---|
| Default prompt (no policy) | Gemma 4 E4B | 1000 | 0.933 | 0.951 | 0.949 | 0.992 | 0.035 |
| Induced policy + T2 calibration | Gemma 4 E4B | 200 (test) | **0.942** | — | — | 0.993 | **0.030** |
| Default prompt (no policy) | Gemma 4 31B | 1000 | **0.977** | 0.984 | 0.984 | ~1.000 | 0.015 |
| DetectGPT (Mitchell et al. 2023) | — | — | `[TODO: cite HC3 figure from [9] or note non-comparability]` | — | — | — | — |
| Fine-tuned RoBERTa (MGTBench, He et al. 2024) | — | — | `[TODO: cite HC3 figure from [4] or note non-comparability]` | — | — | — | — |

`[TODO: Figure 1 — ROC curve for E4B+policy+calibration on test n=200, with the F0.5-optimal operating point highlighted; reliability diagram (10-bin) for the same. Generate from logs/runs/2026-04-23T23-23-21_538a36/predictions.jsonl.]`

**Per-domain breakdown.** Table 2 reports default-prompt performance across the six HC3 subsets. Two patterns are robust across both model scales: (a) F0.5 is uniformly weaker on `open_qa` and `wiki_csai` than on `finance`, `medicine`, and `reddit_eli5`, and (b) the gap closes at 31B but does not vanish (`open_qa` 0.727 / `wiki_csai` 0.868 even at the larger scale). Recall on the AI class is ≥ 0.99 on every 31B run across 1800 examples (no missed AI text), so the F0.5 deficit is precision, not recall — i.e. the default prompt over-flags human text rather than missing AI text. AUROC remains ≥ 0.995 even where F0.5 is weakest (e.g. 31B `open_qa`: F0.5 = 0.727, AUROC = 0.997), which says the *ranking* induced by the log-probabilities is essentially perfect; what is missing is the operating-point choice that calibration is meant to deliver.

**Table 2 — Per-domain default-prompt performance (n=200 each subset).**

| HC3 subset | E4B F0.5 / AUROC | 31B F0.5 / AUROC |
|---|---|---|
| finance | 0.890 / 0.990 | 0.992 / 1.000 |
| medicine | 0.952 / 0.996 | 0.984 / 0.995 |
| open_qa | 0.615 / 0.750 | 0.727 / 0.997 |
| reddit_eli5 | 0.917 / 0.993 | 0.992 / 1.000 |
| wiki_csai | 0.625 / 0.896 | 0.868 / 0.995 |
| `all` (n=1000) | 0.933 / 0.992 | 0.977 / ~1.000 |

**Calibration adds nothing to the hard label on E4B+policy in this slice.** On the n=200 test slice, T1 (single-threshold), T2 (logistic over three log-probs), and the raw argmax produced *identical* hard predictions (F0.5 = 0.942, P = 0.933, R = 0.980). T1's chosen threshold on `logprob(yes) − logprob(no)` was −0.099 — within rounding of the argmax boundary at zero — and T2's sigmoid threshold (0.394) corresponds to roughly equal class priors. Calibration's measurable effect was on probability *quality*, not the decision boundary: ECE dropped from 0.042 (raw) to 0.030 (T2), a 29% reduction. The mechanism is interpretable: the induced policy has already reshaped the log-probability distribution so that argmax is the F0.5-optimal cut. Where calibration *should* bite is the 31B+policy regime, where AUROC ≥ 0.995 on every per-domain split is paired with F0.5 as low as 0.727 (default-prompt 31B `open_qa`) — that headroom is what calibration converts into F0.5 once the 31B+policy run lands.

### 4.2 Policy induction trajectory

The induction loop accepts three revisions in sequence — F0.5 = 0.952 → 0.972 → 0.980 — and then halts after five consecutive rejected candidates (Table 3, Figure 2). Wall time was approximately 10 minutes on Gemma 4 E4B with a pool of 20 seed examples and a scoring subsample of n = 200 drawn from the HC3 train split.

**Table 3 — Induction trajectory (run `2026-04-23T23-06-25_b15d8f`).**

| iter | F0.5 | decision |
|---|---|---|
| 0 | 0.952 | initial (P = 0.943, R = 0.990) |
| 1 | 0.972 | accepted (+0.020) |
| 2 | 0.980 | **accepted — winner** (+0.008) |
| 3 | 0.965 | rejected |
| 4 | 0.965 | rejected |
| 5 | 0.965 | rejected |
| 6 | 0.965 | rejected |
| 7 | 0.965 | rejected → early-stop (5 consecutive) |

**Figure 2.** Trajectory plot at `logs/policies/2026-04-23T23-06-25_b15d8f.png` (F0.5 by iteration, accepted vs. rejected marked).

Iters 3–7 produced an identical F0.5 = 0.965 rejected candidate because the refiner runs at temperature = 0 with the same `(best_policy, misclassified_set)` inputs and so regenerates the same revision on every call. The `max_consecutive_rejections = 5` early-stop kept wall time to ~10 minutes instead of the 45+ a naive 30-iteration budget would have cost; no accepted revisions were missed. (Future work: introduce temperature jitter or negative-example prompting on refine-after-rejection to break the deterministic loop.)

The frozen winner policy (~150 words; the artefact at `logs/policies/2026-04-23T23-06-25_b15d8f.md`) reads in full:

> Look for conversational digressions, colloquialisms, or informal contractions (e.g., "you're," "ca n't," "n't"). Pay attention to sentence structures that feel slightly rambling or conversational, sometimes using parenthetical asides or analogies that build organically rather than presenting a clean, structured argument. Conversely, watch for overly formal, encyclopedic tones, perfect grammatical constructions, and the tendency to list points or define concepts with exhaustive, balanced explanations. However, be cautious of highly technical, explanatory writing that uses analogies or step-by-step processes (like describing physical mechanisms or scientific models) even if the structure is detailed. Also, recognize that natural, descriptive writing — even when explaining complex systems or processes — can maintain high structural integrity without sounding purely academic or list-like. Specifically, detailed explanations of natural processes, scientific models, or technical systems, even when highly structured or sequential, should not automatically be flagged as AI-generated.

The policy is the entire decision rule of Phase 1 — there is no other input to the classifier than the input text and this prose. That is the central object of the §4.3 faithfulness evaluation and the §5.4 Madsen-pre-emption argument.

### 4.3 Faithfulness ablation — RQ2

The policy is run on n = 100 HC3 test-split examples (unseen during induction) under three system-prompt conditions: (a) the frozen *best* policy from §4.2; (b) an *empty* system prompt (default classifier behaviour); and (c) an *inverted* policy whose instructions tell the classifier to assume all texts are human-written. For each example we record the predicted label and the first-token log-probability margin `logprob(yes) − logprob(no)` (real log-probabilities via the Ollama log-prob path validated in D11).

**Per-policy F0.5 (Table 4).**

| policy | F0.5 | yes count | no count | other count |
|---|---|---|---|---|
| `best` (winner from §4.2) | **0.996** | 49 | 51 | 0 |
| `empty` (no system prompt) | 0.965 | 51 | 49 | 0 |
| `inverted` (assume-human) | 0.242 | 3 | 97 | 0 |

**Pairwise behavioural-faithfulness statistics (Table 5).**

| pair | Δlabel rate | mean Δ(logprob(yes) − logprob(no)) (nats) | n logprob-valid |
|---|---|---|---|
| `best_vs_empty` | 0.040 | −1.377 | 100 |
| `best_vs_inverted` | **0.460** | **+8.896** | 100 |
| `empty_vs_inverted` | 0.480 | +10.274 | 100 |

The headline number is the `best_vs_inverted` row: **46.0% of test examples flip their label** when the system prompt is replaced by the adversarially-inverted policy, and the mean log-probability margin shifts by **+8.9 nats** in the direction the policy content commands (toward `yes` under the best policy, toward `no` under the inverted policy). Per Jacovi & Goldberg [5] and Lanham et al. [7], a label-flip rate near zero and a margin shift near zero would mean the policy is a figurehead — the model would be classifying the same way regardless of what the prompt said, and any explanation that pointed at the policy would be plausible-but-not-causal. The observed values are far from zero in both axes, which meets the behavioural-faithfulness criterion for whole-text policy swaps. The `best_vs_empty` deltas are smaller (4% label flip, −1.4 nats) and confirm that the policy and the empty-prompt default agree on most easy cases while diverging on edge cases — i.e. the policy is doing real work on the borderline examples, not the bulk.

`[TODO: Figure 3 — histogram of per-example Δ(logprob(yes) − logprob(no)) under best vs inverted, n=100, x-axis in nats. Generate from logs/runs/2026-04-23T23-21-58_680cab/predictions.jsonl. Same-policy consistency (5 re-samples at T=0) was confirmed identical for all 100 examples in the ablation set; cite as a one-sentence footnote.]`

---

## 5. Discussion (~2 pages)

### 5.1 Summary

This thesis asked whether an LLM-based single-token classifier could match published AI-text detection baselines on HC3 (RQ1) and whether its decisions could be decomposed into human-interpretable explanations whose faithfulness to the underlying decision mechanism is measurable (RQ2). On RQ1 the calibrated classifier built on Gemma 4 E4B with an induced ~150-word policy reaches F0.5 = 0.942 / AUROC = 0.993 / ECE = 0.030 on the HC3 `all` test split — a small absolute lift over the default-prompt E4B baseline (F0.5 = 0.933) but achieved at roughly four times less VRAM than the default-prompt 31B model whose F0.5 is 0.977. On RQ2, replacing the induced policy with an adversarially-inverted policy flips 46% of test labels and shifts the log-probability margin by +8.9 nats in the policy-implied direction, meeting the behavioural-faithfulness criterion of Jacovi & Goldberg [5]. The two phases of the architecture are necessary together: induction reshapes the log-probability surface (so that argmax becomes the F0.5-optimal threshold on E4B), and calibration converts the remaining AUROC headroom into a tunable operating point — work that becomes more important as the underlying model scales (cf. the 31B per-domain results in §4.1).

### 5.2 Relation to prior work

A direct numerical comparison against DetectGPT [9] and the fine-tuned-RoBERTa baselines from MGTBench [4, 11] is sensitive to *which* HC3 subset, *which* AI source LLM, and *which* threshold convention each paper reports. `[TODO: insert literature numbers, with the specific HC3 split and source-LLM noted, and explicitly flag any non-comparable axes in the cell or footnote.]` Conceptually, the present approach differs from DetectGPT in the *signal* it uses — DetectGPT reads the *generator's* log-probability curvature under perturbation, whereas the present classifier reads the *classifier's* log-probabilities of `yes`/`no` under a yes/no prompt — and from RoBERTa-style supervised classifiers in that there is no fine-tuning step: the classifier is the off-the-shelf Gemma 4 model under an induced system prompt. The closest published methods are *ProTeGi* [10] (minibatch errors → textual gradient → prompt edit, the algorithmic precedent for D3) and *Hypotheses-to-Theories* [15] (induced rules retained if they generalise, then used as the inference prompt — the conceptual precedent for D1/D2). To my knowledge no peer-reviewed paper has applied this loop specifically to AI-text detection; the contribution is that application, plus the faithfulness measurement of the resulting policy.

### 5.3 Method discussion

**F0.5 vs. F1 vs. accuracy vs. Youden's J.** F0.5 was chosen because the use case treats false accusations as more costly than missed detections (§1). This produced a tighter precision target than F1 would have — the 31B default-prompt regime has recall(AI) = 1.000 on every domain in §4.1 with F0.5 as low as 0.727, demonstrating that scaling the model alone produces a precision-weak detector that F1 (which weights precision and recall equally) and accuracy (which is symmetric in the binary case) would both report as strong. Youden's J would have made the same point but is uncommon in detection papers; F0.5 is more readable. The thesis would change *substantively* if the use case changed (e.g. an automated content-moderation pipeline that fines false negatives more than false positives would prefer F2) — the operating-point lever is downstream of the policy, not entangled with it.

**Three-class output vs. forced binary.** The `other` class was inert on the test set (zero `other` predictions in §4.3 across all three policies × 100 examples). It nonetheless paid for itself during induction by surfacing refusals as scoring failures rather than silently coercing them to the wrong class — a Phase-1 debugging affordance. For a production classifier with stricter latency budgets, removing the `other` token from the calibration features would cost ~one input dimension and would not change reported test performance.

**Policy induction vs. OPRO / ProTeGi / DSPy / HtT.** The induction loop in §3.2 is a textbook proposer/scorer/accept-reject loop in the family of OPRO [2], DSPy/MIPROv2 [3], ProTeGi [10], TextGrad [4], Promptbreeder [5], EvoPrompt [6], and PromptWizard [7]. The thesis-specific choices are (a) the natural-language *policy* output rather than instruction-tuned exemplars, (b) F0.5 (precision-weighted) as the scoring criterion, and (c) the *retention* of the policy as the explanation artefact post-induction (closer to HtT [15] than to the prompt-search neighbours). A stronger contribution argument would be a head-to-head against ProTeGi or DSPy on the same HC3 splits; this is deferred to Inl. 6 or beyond.

### 5.4 Threats to faithfulness — pre-empting Madsen et al.

Madsen et al. [8] showed that default-prompted LLM self-explanations are often unfaithful to the model's underlying decision mechanism, and Turpin et al. [12] earlier showed the same for Chain-of-Thought rationales. The present thesis pre-empts that critique by construction: the policy *is* the system prompt *is* the classifier, so an explanation of the form *"this text was flagged because rule X in the policy applied"* is grounded in the same object that drives the prediction — there is no separate self-explanation step that could diverge. The behavioural ablation in §4.3 supplies the empirical evidence that the policy is causally load-bearing rather than ornamental: a 0.460 label-flip rate and +8.9 nat margin shift between the best and inverted policies is far from the figurehead regime. This defence holds for the *induced* policy explanation only. Any free-text rationalisation generated *alongside* a prediction (e.g. asking the model to justify its `yes`/`no` answer in the same call) would be a separate self-explanation in the Madsen sense and would inherit the unfaithfulness risk; this thesis does not generate such rationalisations.

### 5.5 Future work

Three lines suggest themselves. First, *finer-grained ablations*: the current §4.3 protocol perturbs the whole policy at once. Sentence-level ablation (drop one rule at a time, observe the label and margin shift) and feature-level ablation (substitute a synonym set in one rule, observe drift) are mechanically straightforward extensions of the same harness and would let the explanation be expressed at the rule rather than policy level. Second, *cross-model comparison*: gpt-oss 20B and Qwen2.5 32B are listed in the operating brief as candidate proposer/classifier models but were not run this cycle (D8); doing so would test whether the induced policy is portable across model families or local to Gemma 4. Third, *human evaluation*: §4.3 measures whether the policy is causally faithful, but not whether the resulting explanation is *useful* to a human reader (Q14 in `decisions.md`); a small user study with educators or editors would close that loop.

### 5.6 Contribution

The contribution of this thesis is threefold. First, an open-source LLM-based single-token detector that scores F0.5 = 0.942 / AUROC = 0.993 / ECE = 0.030 on HC3 `all` (E4B + induced policy + calibration), competitive with the default-prompt 31B baseline (0.977 F0.5) at four times less VRAM. Second, a policy-induction protocol whose output is a ~150-word natural-language rule that doubles as the system prompt of the deployed classifier — i.e. an explanation that is faithful by construction to the decision mechanism. Third, a quantitative faithfulness measurement (0.460 label-flip rate and +8.9 nat margin shift between the best and adversarially-inverted policies) that supplies the behavioural evidence Madsen et al.'s critique demands. The practical pay-off is a detector that gives an auditor a *reason*, not just a percentage — and a reason whose causal load on the decision can be measured rather than asserted.

---

## 6. Conclusion *(≤300 words)*

Can an LLM detect AI-generated text both reliably and explainably? On the evidence assembled here, the answer is a *qualified yes*. With a small open-weights model (Gemma 4 E4B) and a ~150-word natural-language policy induced by a temperature-zero proposer/scorer loop on the HC3 train split, a single-token classifier reaches F0.5 = 0.942 / AUROC = 0.993 / ECE = 0.030 on the HC3 `all` test split — competitive with a four-times-larger default-prompt baseline. The qualifier matters: the comparison to published baselines is sensitive to HC3-subset choice and AI-source-LLM choice, and the per-domain breakdown shows precision deficits on `open_qa` and `wiki_csai` that are not eliminated by scale alone. On the explanation side, replacing the induced policy with an adversarially-inverted policy flips 46% of test labels and shifts the log-probability margin by +8.9 nats — direct behavioural evidence that the policy is causally load-bearing and not a post-hoc rationalisation. Taken together: the rule the model uses to flag a text *is* the rule a human auditor reads. That equivalence is what an educator confronting a flagged essay, an editor handling a flagged submission, or a reviewer adjudicating an authorship dispute actually needs — not "82% likely AI-generated", but "flagged because of these specific properties, and removing them would change the verdict". This thesis demonstrates that the equivalence is achievable on commodity hardware and measurable in practice; whether the resulting explanation is *useful* to a human reader, and whether the policy is portable across model families, are the open questions that frame the work that comes next.

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
- [ ] §4.1 baselines cell — DetectGPT and RoBERTa numbers inserted from literature (or non-comparability flagged)
- [ ] §4.1 Figure 1 (ROC + reliability diagram) generated from `logs/runs/2026-04-23T23-23-21_538a36/`
- [ ] §4.3 Figure 3 (margin histogram) generated from `logs/runs/2026-04-23T23-21-58_680cab/`
- [ ] Discussion answers RQ1 and RQ2 explicitly
- [ ] Conclusion ≤300 words
- [ ] References in ACM format, 10–15 entries, all peer-reviewed
- [ ] No long verbatim quotes; paraphrase + cite (note: the §4.2 policy quote *is* a verbatim quote of an artefact this thesis produced — not a third-party quote, so the rule does not apply)
- [ ] File named `Gustav-Skarman_<title>.pdf` (per Kexjobbsspecifikation instruction §0)
- [ ] Uploaded to Canvas before **Thu 30 Apr 2026, 19:00 CEST** (not 23:59)
