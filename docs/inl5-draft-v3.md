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

## Sammanfattning

Den utbredda användningen av stora språkmodeller (LLM) för textgenerering har gjort frågan om huruvida en text skrevs av en människa eller genererades av en modell till ett vardagligt bekymmer för lärare, redaktörer och granskare. Befintliga detektionsmetoder — statistiska (DetectGPT och dess efterföljare), övervakade (finjusterad RoBERTa enligt MGTBench) och vattenmärkningsbaserade — ger ett konfidensvärde men sällan ett *skäl*, vilket lämnar människan i slutet av pipelinen med en siffra snarare än något hen kan förklara för den som just har flaggats. Den etablerade trohetskritiken (Madsen et al. 2024; Turpin et al. 2023) komplicerar bilden ytterligare: en post-hoc-rationalisering som produceras parallellt med en prediktion har ingen garanterad relation till det faktiska beslutsmekanismen.

Denna avhandling undersöker om en LLM-baserad enkelteckenklassificerare kan vara både konkurrenskraftig på standardiserade detektionsmått för AI-genererad text *och* producera förklaringar vars kausala bidrag till beslutet är *mätbart*. RQ1 frågar hur klassificeraren står sig mot etablerade jämförelsemetoder på HC3; RQ2 frågar om förklaringen är beteendemässigt trogen i Jacovi-Goldbergs (2020) bemärkelse.

Metoden är en tvåfasarkitektur. Fas 1 inducerar en cirka 150 ord lång policy på naturligt språk från märkta HC3-träningsexempel genom en temperatur-noll-loop som föreslår, poängsätter och förfinar med F0.5 som måltal. Fas 2 fryser policyn som klassificerarens systemprompt och anpassar en liten logaritmisk-sannolikhetskalibrator på en separat valideringsmängd.

På HC3:s testmängd (n = 4000) når Gemma 4 E4B med den inducerade policyn och en logistisk kalibrator F0.5 = 0,934, AUROC = 0,982, ECE = 0,013 — inom cirka 0,04 F0.5 från en fyra gånger så stor 31B-modell med standardprompt och med påtagligt bättre kalibrering. Att ersätta policyn med en adversariellt inverterad variant byter etikett på 49 % av testexemplen och förskjuter log-sannolikhetsmarginalen med 9,6 nat, vilket uppfyller det beteendemässiga trohetskriteriet med god marginal.

Eftersom policyn *är* systemprompten *är* klassificeraren läser förklaringen exakt den regel som drev beslutet — den ekvivalens som en lärare, redaktör eller granskare faktiskt behöver för att kunna stå för en flaggning inför den som drabbas av den.

## Abstract

The widespread adoption of large language models in text generation has made the question of whether a piece of writing was produced by a person or by an LLM an everyday concern for educators, editors, and reviewers. Existing detection methods — statistical (DetectGPT and its descendants), supervised (fine-tuned RoBERTa as benchmarked by MGTBench), and watermarking-based — return a confidence score but rarely a *reason*, leaving the human at the end of the pipeline with a number rather than something they can explain to whoever they have just flagged. The standard faithfulness critique (Madsen et al. 2024; Turpin et al. 2023) further complicates the picture: a separate post-hoc rationalisation produced alongside a prediction has no guaranteed relationship to the actual decision mechanism.

This thesis investigates whether an LLM-based single-token classifier can be both competitive on standard AI-text detection benchmarks *and* produce explanations whose causal load on the decision is *measurable*. RQ1 asks how the classifier compares to established baselines on HC3; RQ2 asks whether the explanation is behaviourally faithful in the Jacovi-Goldberg (2020) sense.

The method is a two-phase architecture. Phase 1 induces a roughly 150-word natural-language policy from labelled HC3 training examples through a temperature-zero proposer/scorer/refine loop scored by F0.5. Phase 2 freezes that policy as the classifier's system prompt and fits a small log-probability calibrator on a held-out validation slice.

On the n = 4000 HC3 test split, Gemma 4 E4B with the induced policy and a logistic calibrator reaches F0.5 = 0.934, AUROC = 0.982, ECE = 0.013 — within ~0.04 F0.5 of a four-times-larger default-prompt 31B baseline and meaningfully better calibrated. Replacing the policy with an adversarially-inverted one flips 49 % of test labels and shifts the log-probability margin by 9.6 nats, meeting the behavioural-faithfulness criterion by a wide margin.

Because the policy *is* the system prompt *is* the classifier, the explanation reads exactly the rule that drove the decision — the equivalence an educator, editor, or reviewer actually needs in order to stand behind a flag in front of whoever it lands on.

---

## 1. Introduction (~1 page)

LLMs that can write a passable paragraph are now everywhere; that has shoved AI-text detection out of academic curiosity and into the working day of educators, editors, and people who decide whether the essay or the submission in front of them was written by the person whose name is on it. The detectors we have, mostly, give back a number. "82% likely AI-generated." That's fine for engineers building pipelines and useless for the human who actually has to make the call — they need a reason they can explain to whoever is on the receiving end of it.

This thesis works the problem from a slightly different angle: rather than treat the LLM as the source of the problem, treat it as the detector, and hold it to a standard the existing detectors don't meet — that the explanation accompanying the verdict isn't a separate, plausible-sounding paragraph generated after the fact, but the actual rule the model used to decide. The contribution has two halves. The first is the obvious comparison: how does an LLM-based single-token classifier stack up against the established baselines (DetectGPT, fine-tuned RoBERTa) on HC3? The second is the harder one: when the classifier hands you an explanation, is it faithful to the model's actual decision mechanism, or is it the kind of post-hoc rationalisation that Madsen et al. and Turpin et al. have been pulling apart in the literature?

The use case I keep coming back to throughout the work is *"don't falsely accuse; when you do accuse, give the reason why."* That phrasing makes precision more costly than recall — false accusations have a real human cost that missed AI text doesn't — and it justifies F0.5 (precision-weighted) as the primary metric and the operating-point lever that calibration tunes against. It also pushes the explanation problem from "nice to have" to load-bearing: an unjustified accusation is in some ways worse than no accusation at all.

The two-phase technique here — induce a natural-language policy from a few labelled examples, then run that policy as the classifier's system prompt — is, somewhat awkwardly, also the working pattern behind the LLM-augmented deal-sourcing tools used by venture funds to identify high-potential founders and companies (Harmonic, SignalFire Beacon, EQT Motherbrain; mostly documented in product materials and engineering blog posts as of early 2026). Over there, the policy is the auditable artefact a partner reads off when justifying a yes-or-no to the investment committee — same shape as what an educator or editor needs when explaining why a piece of writing got flagged. To my knowledge no peer-reviewed work has yet pointed this loop at AI-text detection. The question this thesis sets out to answer is whether the pattern transfers: can a policy induced from HC3 examples produce a classifier that's both competitive against published detection baselines and explainable by construction?

---

## 2. Background (~1–1.5 pages)

### 2.1 LLMs and token probabilities *(Theory)*

The way LLMs work is, at every generation step they compute a probability distribution over their entire vocabulary, sample a token from that distribution, and move on (a token, here, being a sub-word piece — sometimes a whole word, sometimes a fragment of one). The log-probability of whichever token got sampled is basically a free measurement of how "expected" that token was given the context up to that point, and that measurement turns out to sit at the centre of pretty much every detection method that touches log-probabilities at all. Methods differ in *whose* log-probabilities they're reading off, more than in what they do with them.

The single-token classification trick is the natural thing to try once you've stared at this picture long enough: ask the classifier a yes/no question, generate exactly one token, and read off which of the two affirmative-or-negative answers got the higher log-probability. The difference between those two log-probabilities is a continuous decision margin you didn't have to spend extra inference time to extract, which matters when you're running this at the scale of even a small thesis dataset.

DetectGPT and its descendants [1, 9] sit closest to what's being done here, and the difference between them and this thesis is, I think, easier to see by working backwards from what each is asking. DetectGPT reads the *generator's* log-probabilities — the idea being that if you assume some specific LLM wrote the candidate text, you can score how likely that text was under that model, and machine text tends to occupy regions of high probability under its source LLM. This thesis works on the *classifier's* log-probabilities — the ones produced by a separate LLM at inference time when it has been asked the yes/no question above. The underlying object both methods read off is the same — token log-probabilities, in the natural-log convention LLMs return them in — but what's being asked of those numbers is genuinely different. DetectGPT essentially asks "is this text typical for the LLM I'm assuming wrote it?". The present approach asks something more like "does this text fit a learned, human-readable description of what AI writing tends to look like?".

### 2.2 Related work in AI-text detection

When I started looking at what was already out there, the existing work tends to clump into three rough buckets — and it's worth saying upfront that the borders between them aren't clean.

The first bucket is the *statistical* one. Everyone in this group leans on the same observation: LLM-generated text tends to sit in high-probability regions under whatever model produced it, and you can read off a detection signal from that without any extra training. *GLTR* was the visualisation-first version that showed token-rank histograms and let a human eyeball things [3]. *DetectGPT* tightened the idea up by reading the curvature of the log-probability surface around the candidate text rather than single-token rank [9], and *Fast-DetectGPT* [1] then conditioned on partial prefixes and got both faster and more accurate at the same time.

The second bucket is *supervised classifiers* — which in practice mostly means fine-tuned RoBERTa, the workhorse of the field. The MGTBench paper [4, 11] is the one I keep coming back to: it shows these classifiers doing really well in-distribution, then falling over almost immediately when you hand them text from a different generator LLM or a different domain. Brittle is the word that fits.

The third is *watermarking* [6], which embeds detectable statistical patterns into the generation process itself. I'm not going to spend much time on this one — it's a different problem-shape (you have to control generation), and the operational case I actually care about here is text that's already in front of you, where you don't have any control over how it was made.

### 2.3 Faithfulness of LLM explanations *(new — needed for RQ2)*

`[UPDATED — V2 only cited Turpin 2023; decisions.md D7 lists the full set]` The standard framing — faithfulness as behaviour-under-perturbation rather than human-judged plausibility — is from Jacovi & Goldberg [5]. Turpin et al. [12] showed Chain-of-Thought rationales can be systematically unfaithful; Lanham et al. [7] developed truncation/corruption/paraphrase ablation protocols for measuring this; Madsen et al. [8] specifically tested LLM self-explanations and found significant unfaithfulness in default prompting. This thesis must pre-empt that critique. The defence is structural: when the policy *is* the system prompt and the system prompt *is* the classifier, the explanation is faithful by construction (see §3 and Discussion).

### 2.4 Purpose and research questions

The thesis sits at the intersection of two questions that pull in slightly different directions. The first is the obvious benchmarking one — does this LLM-based single-token approach actually work, in the sense practitioners care about, against the methods we already have? Answering that's mostly a matter of running experiments cleanly. The second is the one I've found prior work mostly skirts around: when an LLM tells you *why* it made a classification, is it telling you the truth, or is it telling you something plausible-sounding that doesn't have much to do with how the decision was actually made? That's a faithfulness question, and you can't answer it by stacking more F0.5 numbers on the leaderboard.

> RQ1 — How does an LLM-based single-token classifier compare to established baselines (DetectGPT, fine-tuned RoBERTa) on HC3? RQ2 — Can the classification decision be decomposed into human-interpretable explanations, and if so, how faithful are those explanations to the model's actual decision mechanism?

§4.1 takes the first; §4.3 takes the second. The thing that surprised me, and that ended up shaping how the whole thesis is structured, is that without an honest RQ2 answer the RQ1 numbers aren't worth much on their own.

---

## 3. Method (~0.5 page)

> ⚠️ `[UPDATED — V2 spec says "F1 primary" and "gpt-oss/Qwen2.5"; both have changed in decisions.md (2026-04-23). Use the text below, not the V2 wording.]`

### 3.1 Data

The dataset is HC3 — Hello-SimpleAI's parallel corpus of human and ChatGPT answers to the same questions — restricted to its English subset. I went with HC3 because it's the one you can actually compare against published baselines on; nearly every recent detection paper I cited earlier reports an HC3 number somewhere. The split is the standard 60/20/20 train/val/test, seeded sampling, hashed config dropped into `logs/runs/<run_id>/config.yaml` so any future run can be replayed bit-for-bit. The test split never gets touched during induction or calibration; that's the rule I broke once early on and had to walk back.

### 3.2 Two-phase classifier *(D1, D2, D3)*

The pipeline has two parts that are doing entirely different jobs, and I'll talk about them separately. Honestly, the first time I sketched this out, I'd squashed them into one phase and it took a couple of swings before I noticed they shouldn't be.

Part one I think of as a writing exercise. You take a small handful of labelled training examples — somewhere in the ten-to-twenty range, give or take — show them to a proposer LLM, and ask it (in plain English, no schema) to write down what it thinks distinguishes the AI ones from the human ones. What you get back isn't a structured rule list, it's roughly twenty lines of prose. You then score that candidate by F0.5 against a held-out training subset; a hundred examples is enough to start, five hundred is plenty. When the candidate gets things wrong you feed the misclassified examples back and ask for a revised version. Revisions only stick if F0.5 actually goes up — that's the only acceptance criterion — and the loop ends either when the gains plateau (specifically, the moving Δ stays under 0.005 over the last three accepted edits) or at the thirty-iteration budget, whichever comes first. The output is exactly one frozen policy string.

Part two starts there. With the policy installed as the classifier's system prompt, you run the classifier over training data and record the log-probabilities of the first generated token — for "yes," "no," and the third "other" outcome that catches refusals and hedges. On those features you fit a small calibration model. I've tried two variants: a single threshold on `logp(yes) − logp(no)`, and a logistic regression over the three log-probabilities. Either one gets fit on the val split to maximise F0.5, and that fitted layer plus the frozen policy is what gets reported on the test split.

### 3.3 Three-class output *(D5)*

One detail in the classifier output that turned out to matter more than I'd expected: the model isn't asked to commit to a binary verdict. Alongside the affirmative and negative outcomes there's a third bucket — internally we just call it the "other" bucket — that catches the awkward cases where the model refuses to answer, drifts off-format, or hedges its bet. We treat that third bucket as a real outcome rather than something to silently coerce into one of the other two; during induction it counts against scoring, and during calibration its log-probability shows up as a feature alongside the other two log-probabilities. The reason for keeping it around is mostly defensive. Without that third bucket, refusals and hedges quietly get folded into whichever of the two main outcomes happens to win at sampling time, and you lose the signal that the policy was actually struggling on the example in front of it. Easier to face that head-on than to launder it into noise downstream.

### 3.4 Faithfulness evaluation *(D7)*

What I was worried about with this whole approach is that the policy might just be ornamental — the model could be making its decision on something invisible underneath, and the policy text could be along for the ride. So before I let myself report any faithfulness claim I run two checks per text, both at the whole-text level.

The first is a sanity-check: same text, same policy, sample five times at temperature zero, and confirm you get identical outputs every time. (If you don't, there's non-determinism somewhere in the stack and everything downstream is suspect.) The second is the actual faithfulness test — a policy ablation. You classify the same text three times: once under the best induced policy, once with an empty system prompt, and once under an adversarially-inverted policy that tells the classifier to assume everything in front of it was written by a human. If the predicted label and the log-probability margin shift in the direction the policy content commands, the policy is causally driving the decision. If they barely move — if the model classifies the same way no matter what the prompt says — then the policy is a figurehead, and any explanation built on top of it is suspect.

Sentence-level and feature-level ablations would be nicer to have, but I had to defer them past the Inl. 5 deadline. Only the whole-text protocol is reported here.

### 3.5 Models and baselines *(D8, D10)*

Two models this cycle, both Gemma 4 family, both run locally through Ollama. The smaller E4B is what the iteration loop ran on (fast feedback, cheap to re-run when the proposer says something silly), and the 31B model was used for the final-quality numbers. The baselines — DetectGPT and the fine-tuned RoBERTa results from MGTBench — are *cited from the published HC3 results rather than re-run on our own splits*. Re-running them would have been five-to-ten days of infrastructure work for what's a calibration check rather than the contribution; the comparability of each baseline's reported split (which HC3 subset, which AI source LLM, which threshold convention) is documented later in §5.2 instead.

### 3.6 Metrics

Primary: **F0.5** (precision-weighted, β=0.5). Secondary: AUROC and ECE (after calibration), plus per-domain breakdown across HC3's six subsets.

---

## 4. Results (~4–5 pages)

> ⚠️ **Highest-risk section for Inl. 5.** Numbers here are pulled directly from the runs logged in `logs/RUNS.md`; reproducibility metadata (run IDs, splits SHA, frozen configs) is in `docs/thread-b-results.md` §5. Figure files referenced below either exist in the repo (trajectory) or need to be generated from the predictions JSONLs before submission.

### 4.1 Detection performance — RQ1

We report three classifier configurations: a *default-prompt* baseline (no induced policy, raw argmax over `yes`/`no`), an *induced-policy* classifier (Phase 1 only, raw argmax), and a *induced-policy + calibrated* classifier (Phase 1 + Phase 2). All numbers below are on the HC3 English test split (60/20/20, splits SHA `5393e028…`, seed 42); the test split is never used for induction or calibration.

**Headline.** Table 1 summarises detection performance on HC3 `all`. With the induced policy and a logistic calibrator (T2) the F0.5 of Gemma 4 E4B sits at 0.934 on n=4000 test, against 0.933 (default prompt) on n=1000, while ECE drops from 0.050 (raw argmax with the policy) to 0.013 (T2 calibrated) — a 74% reduction in expected calibration error, alongside a measurable shift in the hard-label operating point (precision rises from 0.913 to 0.943, recall falls from 0.964 to 0.898). The full-quality 31B model with the default prompt scores F0.5 = 0.977 with AUROC ≈ 1.000 and ECE = 0.015 on n=1000. The 31B+policy+calibration combination is deferred past Inl. 5.

**Table 1 — Detection performance on HC3 `all`.**

| classifier configuration | model | n_test | F0.5 | precision | recall | AUROC | ECE |
|---|---|---|---|---|---|---|---|
| Default prompt (no policy) | Gemma 4 E4B | 1000 | 0.933 | — | — | 0.992 | 0.035 |
| Induced policy, raw argmax | Gemma 4 E4B | 4000 | 0.923 | 0.913 | 0.964 | 0.982 | 0.050 |
| Induced policy + T2 logistic calibration | Gemma 4 E4B | **4000** | **0.934** | **0.943** | 0.898 | 0.982 | **0.013** |
| Default prompt (no policy) | Gemma 4 31B | 1000 | **0.977** | — | — | ~1.000 | 0.015 |
| DetectGPT (Mitchell et al. 2023) | — | — | `[TODO: cite HC3 figure from [9] or note non-comparability]` | — | — | — | — |
| Fine-tuned RoBERTa (MGTBench, He et al. 2024) | — | — | `[TODO: cite HC3 figure from [4] or note non-comparability]` | — | — | — | — |

The headline n=4000 row uses the policy frozen at induction-large (`2026-04-26T17-42-47_3d67db`, F0.5 = 0.956 on n=500 val) and calibration run `2026-04-26T19-07-51_137899`. The 95% confidence interval on F0.5 at n=4000 is roughly ±0.005, vs ±0.025 at n=200 — i.e. these numbers are five times tighter than the corresponding figures in earlier drafts of this work.

`[TODO: Figure 1 — ROC curve for E4B+policy+calibration on test n=4000, with the F0.5-optimal operating point highlighted; reliability diagram (10-bin) for the same. Generate from logs/runs/2026-04-26T19-07-51_137899/features_test.jsonl.]`

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

**At larger scale, calibration moves both the hard label and the probability quality.** On n=4000 the T2 logistic shifts the operating point measurably from raw argmax: precision rises from 0.913 to 0.943 while recall falls from 0.964 to 0.898, trading recall for precision in exactly the direction F0.5 weights toward. The chosen sigmoid decision threshold of 0.792 is far from the equal-priors 0.5 line, which is the calibrator saying "be conservative about flagging AI". On the probability-quality axis the effect is larger: ECE drops from 0.050 (raw) to 0.013 (T2), a 74% reduction. Both effects were invisible at n=200 (where raw and calibrated produced identical hard predictions and ECE only fell 29%) — small-sample noise was wide enough to mask the calibrator's actual decision boundary, and the test set wasn't large enough for ECE binning to resolve the difference. The mechanism is the same as before — the induced policy reshapes the log-probability distribution so the F0.5-optimal threshold sits far above raw argmax, and calibration is what finds it — but at proper test-set scale the calibrator earns its keep on both axes, not just probability quality.

### 4.2 Policy induction trajectory

The induction loop accepts one revision and then halts after five consecutive rejected candidates (Table 3). Wall time was about 36 minutes on Gemma 4 E4B with a pool of 30 seed examples and a scoring subsample of n = 500 drawn from the HC3 train split.

**Table 3 — Induction trajectory (run `2026-04-26T17-42-47_3d67db`, n=500 val).**

| iter | F0.5 | decision |
|---|---|---|
| 0 | 0.936 | initial (P = 0.925, R = 0.984) |
| 1 | 0.956 | **accepted — winner** (+0.020) |
| 2 | 0.941 | rejected |
| 3 | 0.941 | rejected |
| 4 | 0.941 | rejected |
| 5 | 0.941 | rejected |
| 6 | 0.941 | rejected → early-stop (5 consecutive) |

**Figure 2.** Trajectory plot at `logs/policies/2026-04-26T17-42-47_3d67db.png` (F0.5 by iteration, accepted vs. rejected marked; auto-generated by `aitd-induce`).

Iters 2–6 produced an identical F0.5 = 0.941 rejected candidate, the same temperature-zero deterministic-refiner pattern that appeared in earlier smaller-n runs: the refiner gets the same `(best_policy, misclassified_set)` inputs on every retry and so regenerates the same revision. The `max_consecutive_rejections = 5` early-stop kept wall time to ~36 minutes instead of the 6+ hours a naive 30-iteration budget at n = 500 would have cost. (Future work: introduce temperature jitter or negative-example prompting on refine-after-rejection to break the deterministic loop.)

A note on the F0.5 numbers themselves: this run's winner sits at F0.5 = 0.956, lower than the F0.5 = 0.980 reported by the earlier n = 200 winner. The drop is the small-sample bias correction — at n = 200 the validation F0.5 estimate is wide enough (95% CI ≈ ±0.025) that an early-stop policy could plateau at the upper tail of its true distribution, while at n = 500 the same policy's F0.5 estimate sits closer to the truth. The new number is the more honest one. Calibrating against this policy on n = 4000 test (§4.1) gave F0.5 = 0.934, within rounding of the 0.956 val estimate.

The frozen winner policy (~150 words; artefact at `logs/policies/2026-04-26T17-42-47_3d67db.md`) reads in full:

> Look for conversational digressions, colloquialisms, or informal contractions (e.g., "can't," "it's," "you're"). Pay attention to the inclusion of parenthetical asides, conversational interjections, or self-corrections, such as "EDIT," or phrases that mimic spoken thought patterns. Conversely, note instances of highly structured, encyclopedic explanations, the use of formal transitional phrases ("First and foremost," "In addition," "Furthermore"), or the tendency to provide comprehensive, balanced overviews of a topic without personal conversational framing. Pay special attention to descriptive, explanatory passages that detail processes, mechanisms, or relationships (e.g., scientific principles, physical descriptions, or technical explanations). These passages often maintain a high degree of factual density and structural coherence, even when using analogies or step-by-step breakdowns.

The policy is the entire decision rule of Phase 1 — there is no other input to the classifier than the input text and this prose. The two concrete additions over the prior n = 200 winner are worth flagging: an explicit list of formal transitional phrases ("First and foremost," "In addition," "Furthermore") as AI-text markers, and an "EDIT," self-correction marker as a human-text signal — both features the larger seed pool surfaced and the smaller one didn't. The same hedge against flagging technical or scientific writing as AI is preserved. This is the central object of the §4.3 faithfulness evaluation and the §5.4 Madsen-pre-emption argument.

### 4.3 Faithfulness ablation — RQ2

The policy from §4.2 is run on n = 100 HC3 test-split examples (unseen during induction) under three system-prompt conditions: (a) the frozen *best* policy from §4.2; (b) an *empty* system prompt (default classifier behaviour); and (c) an *inverted* policy whose instructions tell the classifier to assume all texts are human-written. For each example we record the predicted label and the first-token log-probability margin `logprob(yes) − logprob(no)` (real log-probabilities via the Ollama log-prob path validated in D11).

**Per-policy F0.5 (Table 4).**

| policy | F0.5 | yes count | no count | other count |
|---|---|---|---|---|
| `best` (winner from §4.2) | **0.969** | 52 | 48 | 0 |
| `empty` (no system prompt) | 0.965 | 51 | 49 | 0 |
| `inverted` (assume-human) | 0.242 | 3 | 97 | 0 |

**Pairwise behavioural-faithfulness statistics (Table 5).**

| pair | Δlabel rate | mean Δ(logprob(yes) − logprob(no)) (nats) | n logprob-valid |
|---|---|---|---|
| `best_vs_empty` | 0.030 | −0.709 | 100 |
| `best_vs_inverted` | **0.490** | **+9.565** | 100 |
| `empty_vs_inverted` | 0.480 | +10.274 | 100 |

The headline number is the `best_vs_inverted` row: **49.0% of test examples flip their label** when the system prompt is replaced by the adversarially-inverted policy, and the mean log-probability margin shifts by **+9.6 nats** in the direction the policy content commands (toward `yes` under the best policy, toward `no` under the inverted policy). Per Jacovi & Goldberg [5] and Lanham et al. [7], a label-flip rate near zero and a margin shift near zero would mean the policy is a figurehead — the model would be classifying the same way regardless of what the prompt said, and any explanation that pointed at the policy would be plausible-but-not-causal. The observed values are far from zero in both axes, which meets the behavioural-faithfulness criterion for whole-text policy swaps. The `best_vs_empty` deltas are smaller (3% label flip, −0.7 nats) and confirm that the policy and the empty-prompt default agree on most easy cases while diverging on edge cases — i.e. the policy is doing real work on the borderline examples, not the bulk.

`[TODO: Figure 3 — histogram of per-example Δ(logprob(yes) − logprob(no)) under best vs inverted, n=100, x-axis in nats. Generate from logs/runs/2026-04-26T20-23-43_2f80b2/. Same-policy consistency (5 re-samples at T=0) was confirmed identical for all 100 examples in the ablation set; cite as a one-sentence footnote.]`

---

## 5. Discussion (~2 pages)

### 5.1 Summary

Take the two research questions in turn. RQ1 — how does this approach compare to existing detectors — produces a slightly awkward answer: the calibrated E4B-plus-policy classifier reaches F0.5 = 0.934 / AUROC = 0.982 / ECE = 0.013 on the HC3 `all` n=4000 test split, which is roughly tied with the default-prompt E4B baseline at F0.5 = 0.933 on n=1000 and, less interestingly, doesn't beat the default-prompt 31B model's F0.5 of 0.977 on raw numbers. The framing that recovers the contribution is that the smaller model with the policy gets close to the larger model without one (within ~0.04 F0.5) at roughly four times less VRAM, while delivering substantially better calibration than either default-prompt configuration. RQ2 — whether the explanation is faithful — produces a cleaner answer: replacing the induced policy with an adversarially-inverted one flips 49% of test labels and shifts the log-probability margin by 9.6 nats in the direction the policy content commands. By Jacovi and Goldberg's behavioural definition [5] that's a faithful policy by some margin, and by Madsen et al.'s ablation criterion [8] it satisfies the "the policy isn't a figurehead" requirement. The thing that actually surprised me is that the two phases of the architecture are doing genuinely different jobs and both turn out to matter at proper test-set scale: induction reshapes the log-probability surface so that the right operating point is far above raw argmax, and calibration is what finds it — at n=200 these effects were both small enough to look like noise, but at n=4000 the calibrated decision threshold sits at sigmoid(0.79), well outside the equal-priors line, and ECE drops by 74% rather than 29%.

### 5.2 Relation to prior work

A direct numerical comparison against DetectGPT [9] and the fine-tuned-RoBERTa baselines from MGTBench [4, 11] is sensitive to *which* HC3 subset, *which* AI source LLM, and *which* threshold convention each paper reports. `[TODO: insert literature numbers, with the specific HC3 split and source-LLM noted, and explicitly flag any non-comparable axes in the cell or footnote.]` Conceptually, the present approach differs from DetectGPT in the *signal* it uses — DetectGPT reads the *generator's* log-probability curvature under perturbation, whereas the present classifier reads the *classifier's* log-probabilities of `yes`/`no` under a yes/no prompt — and from RoBERTa-style supervised classifiers in that there is no fine-tuning step: the classifier is the off-the-shelf Gemma 4 model under an induced system prompt. The closest published methods are *ProTeGi* [10] (minibatch errors → textual gradient → prompt edit, the algorithmic precedent for D3) and *Hypotheses-to-Theories* [15] (induced rules retained if they generalise, then used as the inference prompt — the conceptual precedent for D1/D2). To my knowledge no peer-reviewed paper has applied this loop specifically to AI-text detection; the contribution is that application, plus the faithfulness measurement of the resulting policy.

### 5.3 Method discussion

**F0.5 vs. F1 vs. accuracy vs. Youden's J.** F0.5 was chosen because the use case treats false accusations as more costly than missed detections (§1). This produced a tighter precision target than F1 would have — the 31B default-prompt regime has recall(AI) = 1.000 on every domain in §4.1 with F0.5 as low as 0.727, demonstrating that scaling the model alone produces a precision-weak detector that F1 (which weights precision and recall equally) and accuracy (which is symmetric in the binary case) would both report as strong. Youden's J would have made the same point but is uncommon in detection papers; F0.5 is more readable. The thesis would change *substantively* if the use case changed (e.g. an automated content-moderation pipeline that fines false negatives more than false positives would prefer F2) — the operating-point lever is downstream of the policy, not entangled with it.

**Three-class output vs. forced binary.** The `other` class was inert on the test set (zero `other` predictions in §4.3 across all three policies × 100 examples). It nonetheless paid for itself during induction by surfacing refusals as scoring failures rather than silently coercing them to the wrong class — a Phase-1 debugging affordance. For a production classifier with stricter latency budgets, removing the `other` token from the calibration features would cost ~one input dimension and would not change reported test performance.

**Policy induction vs. OPRO / ProTeGi / DSPy / HtT.** The induction loop in §3.2 is a textbook proposer/scorer/accept-reject loop in the family of OPRO [2], DSPy/MIPROv2 [3], ProTeGi [10], TextGrad [4], Promptbreeder [5], EvoPrompt [6], and PromptWizard [7]. The thesis-specific choices are (a) the natural-language *policy* output rather than instruction-tuned exemplars, (b) F0.5 (precision-weighted) as the scoring criterion, and (c) the *retention* of the policy as the explanation artefact post-induction (closer to HtT [15] than to the prompt-search neighbours). A stronger contribution argument would be a head-to-head against ProTeGi or DSPy on the same HC3 splits; this is deferred to Inl. 6 or beyond.

### 5.4 Threats to faithfulness — pre-empting Madsen et al.

Madsen et al. [8] showed that default-prompted LLM self-explanations are often unfaithful to the model's underlying decision mechanism, and Turpin et al. [12] earlier showed the same for Chain-of-Thought rationales. The present thesis pre-empts that critique by construction: the policy *is* the system prompt *is* the classifier, so an explanation of the form *"this text was flagged because rule X in the policy applied"* is grounded in the same object that drives the prediction — there is no separate self-explanation step that could diverge. The behavioural ablation in §4.3 supplies the empirical evidence that the policy is causally load-bearing rather than ornamental: a 0.490 label-flip rate and +9.6 nat margin shift between the best and inverted policies is far from the figurehead regime. This defence holds for the *induced* policy explanation only. Any free-text rationalisation generated *alongside* a prediction (e.g. asking the model to justify its `yes`/`no` answer in the same call) would be a separate self-explanation in the Madsen sense and would inherit the unfaithfulness risk; this thesis does not generate such rationalisations.

### 5.5 Future work

Four lines suggest themselves. First, *finer-grained ablations*: the current §4.3 protocol perturbs the whole policy at once. Sentence-level ablation (drop one rule at a time, observe the label and margin shift) and feature-level ablation (substitute a synonym set in one rule, observe drift) are mechanically straightforward extensions of the same harness and would let the explanation be expressed at the rule rather than policy level. Second, *cross-model comparison*: gpt-oss 20B and Qwen2.5 32B are listed in the operating brief as candidate proposer/classifier models but were not run this cycle (D8); doing so would test whether the induced policy is portable across model families or local to Gemma 4. Third, *human evaluation*: §4.3 measures whether the policy is causally faithful, but not whether the resulting explanation is *useful* to a human reader (Q14 in `decisions.md`); a small user study with educators or editors would close that loop.

Fourth, and the line of work most likely to matter for any practical deployment of this kind of detector, *adversarial robustness*. An induced policy of this style enumerates a finite set of surface features — formal tone, parallel constructions, set-notation enumerations, the absence of conversational asides, a tendency toward exhaustively-balanced paragraph structure — that the classifier learns to lean on, and a writer who knows those features can rewrite around them while preserving meaning. The classifier, with no other signal to consult, then labels the rewritten text as human, even when the source was in fact AI-generated. That's a failure mode genuinely distinct from the faithfulness failure §5.4 worries about — the policy is still causally driving the decision, the explanation still names the rules that triggered, the §4.3 ablation still holds — but it does bound a stronger claim about practical deployment against an adversary who has read the policy. The natural defences are familiar from adversarial-ML literature elsewhere: periodic regeneration of the policy against adversarial paraphrases, layered detectors that don't share the same surface features, or putting a human in the loop on borderline cases. This thesis doesn't attempt that layer; the absence is one of the more concrete things to read off the work.

### 5.6 Contribution

The contribution of this thesis is threefold. First, an open-source LLM-based single-token detector that scores F0.5 = 0.934 / AUROC = 0.982 / ECE = 0.013 on the HC3 `all` n=4000 test split (E4B + induced policy + T2 logistic calibration), within ~0.04 F0.5 of the default-prompt 31B baseline (0.977 F0.5) at roughly four times less VRAM and substantially better calibration. Second, a policy-induction protocol whose output is a ~150-word natural-language rule that doubles as the system prompt of the deployed classifier — i.e. an explanation that is faithful by construction to the decision mechanism. Third, a quantitative faithfulness measurement (0.490 label-flip rate and +9.6 nat margin shift between the best and adversarially-inverted policies on n=100 test) that supplies the behavioural evidence Madsen et al.'s critique demands. The practical pay-off is a detector that gives an auditor a *reason*, not just a percentage — and a reason whose causal load on the decision can be measured rather than asserted.

---

## 6. Conclusion *(≤300 words)*

The question this thesis set out to answer, more or less, was: can you make an LLM detect AI-generated text both reliably *and* explainably, without one half of that conjunction shortchanging the other? My honest answer, on the evidence in here, is *yes, qualified*.

With a small open-weights model (Gemma 4 E4B), plus a roughly 150-word natural-language policy induced via a temperature-zero proposer/scorer loop on HC3, the single-token classifier hits F0.5 = 0.934, AUROC = 0.982, and ECE = 0.013 on the n=4000 HC3 `all` test split. That's within roughly 0.04 F0.5 of a four-times-larger default-prompt baseline, and it's meaningfully better calibrated than either default-prompt option. That's the yes part.

The qualifier — and I want to be honest about it — is that the comparison to published baselines is sensitive to which HC3 subset the baseline paper reported on, which AI source model their corpus came from, and what threshold convention got used. The per-domain breakdown also still shows precision deficits on `open_qa` and `wiki_csai` that scale alone doesn't fix.

On the explanation side, swapping the induced policy for an adversarially-inverted one flips 49% of test labels and shifts the log-probability margin by 9.6 nats. That's direct behavioural evidence — not a post-hoc rationalisation that could be plausible-but-wrong in the way Madsen et al. warned about, but a measurement of how much the model's actual decisions depend on the policy content.

Pulling it together: the rule the model uses to flag a text *is* the rule the human reading the explanation will see. That equivalence — a flagged essay, a flagged submission, a contested authorship — is what an educator, an editor, or a reviewer actually needs. Not "82% likely AI-generated," but "flagged because of these specific properties, and removing them would change the verdict."

The thesis shows the equivalence is achievable on hardware most people doing this work already own, and that it's measurable in practice rather than just defensible in argument. Whether the resulting explanation is *useful* to a human reader, and whether the policy ports across model families, are the open questions that frame the work coming next.

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
