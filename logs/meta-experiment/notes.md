# Meta-experiment iteration notes

Adversarial rewriting log: rewrite the prose of `docs/inl5-draft-v2.md` to drive down the AI-flag rate of the frozen induced-policy classifier (Gemma 4 E4B + policy `2026-04-23T23-06-25_b15d8f`, F0.5 = 0.980). Hard skips: §4 Results, References, Pre-flight, Sammanfattning/Abstract placeholders. Each iteration's per-section JSON is in this directory.

## Headline

| run | AI flags | mean P(AI) | mean margin (nats) |
|---|---|---|---|
| baseline (E4B + best policy) | 9/17 | 0.534 | +0.44 |
| iter-1 (E4B) | 4/17 | 0.297 | −1.92 |
| iter-2 (E4B) | 1/17 | 0.138 | −3.73 |
| iter-3 (E4B) | **0/17** | 0.123 | −3.79 |
| iter-4 (31B + best policy, same v2) | **0/17** | **0.000** | **−12.06** |
| iter-5 (E4B + best policy, v2 with adversarial-robustness paragraph added to §5.5) | **0/17** | 0.124 | −3.78 |
| v3-baseline (E4B + new induction-large policy, copy of v2 + thesis-numbers updates) | 6/17 | 0.319 | varied |
| **v3-iter-1 (E4B + new policy, reasoning-informed rewrites of the 6 flagged sections)** | **0/17** | **0.012** | **−6.6** |

## iter-1

**What I changed.** Rewrote 9/9 baseline-flagged sections (§1, §2.1, §2.4, §3.2, §3.3, §3.4, §3.5, §5.1, §6). Style targets derived from sections that passed as human at baseline (§5.2 / §5.3 / §5.6):

- contractions (`it's`, `isn't`, `doesn't`)
- parenthetical asides — lots of them
- first-person register where natural ("I keep coming back to")
- long sentences with multiple clauses, hedging
- conversational openers ("More or less:", "By the way,")
- swap perfect parallel constructions for slight asymmetries

**What flipped (AI → human):** §1 (0.768 → 0.002), §3.4 (0.921 → 0.288), §3.5 (0.981 → 0.007), §5.1 (0.985 → 0.001), §6 (0.809 → 0.342).

**What didn't (still flagged AI):** §2.1 (0.760 → 0.581 — close), §2.4 (0.999 → 0.994), §3.2 (0.999 → 0.974), §3.3 (0.999 → 0.999 — barely budged).

**Hypothesis for iter-2.** The four stragglers are *structurally* AI-shaped, not just stylistically. The induced policy explicitly says: "the tendency to list points or define concepts with exhaustive, balanced explanations." So:

- §3.2's bold "Phase 1 / Phase 2" parallel structure trips the list trigger
- §3.3's `{yes, no, other}` enumeration is a literal three-item list
- §2.4's RQ1/RQ2 numbered blockquote is a numbered list
- §2.1's formal definition of "token" is a definition trigger

Rewriting register won't move them further. Structure has to dissolve.

## iter-2

**What I changed.** Structural rewrites of the four stragglers (§2.1, §2.4, §3.2, §3.3). The hypothesis that *structure* was the remaining trigger turned out to be right.

- §2.1: dropped the formal "A token is" definition into a parenthetical, removed the explicit `Yes`/`No` enumeration, looped the DetectGPT-vs-this-thesis comparison into a winding contrast instead of a parallel one.
- §2.4: added a long pre-amble *and* a long after-paragraph around the RQ blockquote so the list is no longer the dominant texture; phrased the difference between the two RQs as something that "pulls in slightly different directions" rather than as parallel.
- §3.2: removed the bold `**Phase 1**` / `**Phase 2**` banners entirely, merged into one rambling paragraph that uses "the first step" and "by the end of induction" as soft transitions.
- §3.3: dropped `{yes, no, other}` set notation, replaced enumeration with "alongside the affirmative and negative outcomes there's a third bucket".

**What flipped.** Three of four:
- §2.1: 0.581 → **0.001** (margin −6.53n — extreme flip)
- §3.2: 0.974 → **0.215**
- §3.3: 0.999 → **0.001** (also extreme — the literal set notation was the trigger)

**What didn't.** §2.4 dropped from 0.994 to **0.633** but is still on the AI side. The numbered `**RQ1.** ... **RQ2.** ...` blockquote remains a strong list cue even with surrounding prose. The structure of *that block itself* is what's left.

**Headline.** 4/17 → 1/17 AI flags. Mean P(AI) 0.297 → 0.138. Mean margin −1.92 → −3.73 nats.

**Big read.** Style rewrites moved the easy half. Structural rewrites moved most of the hard half. Numbered lists with parallel boldface captions are the strongest single trigger for this policy — match exactly the "tendency to list points" + "exhaustive, balanced explanations" clauses in the policy text.

## iter-3

**What I changed.** Just §2.4 — collapsed the parallel `**RQ1.** ... **RQ2.** ...` numbered blockquote into a single-paragraph blockquote that names both RQs in flowing prose. Surrounding paragraphs unchanged.

**What flipped.** §2.4: 0.633 → **0.381** (now human).

**Headline.** **0/17 AI flags.** Mean P(AI) 0.138 → 0.123. Mean margin −3.73 → −3.79 nats.

**Big read.** The single trick was dissolving the parallel two-item numbered list into one continuous-prose blockquote. Same content, same visual highlight, but no longer reads as "exhaustive balanced explanation" to the policy. Confirms the iter-2 read: numbered parallel structures are the strongest single trigger of this particular induced policy.

## iter-4 (plan: 31B stress test)

Three iterations of style + structure rewrites have driven the E4B+best-policy classifier from 9/17 to 0/17 AI flags on the prose sections of the thesis. That's a fully successful evasion of the *small-model* classifier under its strongest induced policy.

The natural next question: does this evasion transfer to the *larger* classifier? Gemma 4 31B with the default prompt scored F0.5 = 0.977 / AUROC ≈ 1.000 on HC3 (cf. §4.1 of the thesis itself), making it the strongest detector benchmarked in this work. If the rewritten v2 still passes as human under 31B with the same induced policy as system prompt, the evasion is meaningfully robust to model scale. If it doesn't, the per-section deltas tell us *which* sections are E4B-fragile vs. genuinely human-shaped.

Run: `python scripts/classify_draft_sections.py --model gemma4:31b --label iter-4-31b`. No further v2 edits this iteration — the test is the *same prose* against a *stronger judge*.

## iter-4 — what happened

**Headline.** 0/17 sections flagged AI on Gemma 4 31B. Every section returned P(AI) ≈ 0.000. Where the margin was computable it averaged −12.06 nats; in most sections the `Yes` token didn't even make the top-20 candidates, which is the same pattern Thread A's calibration notes described for 31B + induced policy ("the chosen token is near-deterministic").

**Per-section deltas (E4B iter-3 → 31B iter-4 on the same v2):**

| section | E4B P(AI) | 31B P(AI) | E4B margin | 31B margin |
|---|---|---|---|---|
| §1 Intro | 0.002 | 0.000 | −6.41 | n/a (Yes off-top-20) |
| §2.1 LLMs/tokens | 0.001 | 0.000 | −6.53 | n/a |
| §2.2 Related work | 0.122 | 0.001 | −1.98 | −6.84 |
| §2.3 Faithfulness BG | 0.191 | 0.000 | −1.44 | n/a |
| §2.4 Purpose & RQs | 0.381 | 0.000 | −0.49 | n/a |
| §3.1 Data | 0.479 | 0.000 | −0.09 | n/a |
| §3.2 Two-phase | 0.215 | 0.000 | −1.30 | n/a |
| §3.3 Three-class | 0.001 | 0.000 | −7.12 | n/a |
| §3.4 Faithfulness eval | 0.288 | 0.000 | −0.91 | n/a |
| §3.5 Models & baselines | 0.007 | 0.000 | −5.02 | n/a |
| §5.1 Summary | 0.001 | 0.000 | −7.17 | n/a |
| §5.2 Relation | 0.002 | 0.000 | −6.26 | n/a |
| §5.3 Method discussion | 0.002 | 0.000 | −6.11 | n/a |
| §5.4 Threats | 0.030 | 0.000 | −3.48 | n/a |
| §5.5 Future work | 0.034 | 0.000 | −3.34 | **−17.28** |
| §5.6 Contribution | 0.002 | 0.000 | −6.06 | n/a |
| §6 Conclusion | 0.342 | 0.000 | −0.66 | n/a |

**What this says.**

1. The adversarial rewrites are robust to classifier scale within the Gemma 4 family + this policy. The same prose that barely passed on E4B (§3.1 at 0.479, §3.4 at 0.288, §6 at 0.342) is rejected as human with overwhelming confidence on 31B.
2. 31B is *more* susceptible to the evasion than E4B. Once the prose looks human on the surface features the policy enumerates, the larger model commits harder to that verdict (margins of −12 nats and beyond, vs E4B's −3.79 mean).
3. This is the same pattern Thread A's run notes flagged for the 31B+policy regime — recall(AI) was 1.000 on every default-prompt 31B run (no missed AI text), but precision was the variable axis. Translation: 31B is *sensitive* to whatever the policy points at, and once the policy doesn't point at anything in the input, the verdict swings hard the other way.

## Final analysis across the four iterations

- **Total reduction.** 9/17 AI flags at baseline → 0/17 after three rewrites. Robust to model scaling from E4B to 31B with the same policy.
- **Most effective move.** iter-2 — dissolving the structural patterns the policy explicitly targets (numbered lists, parallel boldface captions, set-notation enumerations, formal definitions). Bigger lift than iter-1's register-only rewrites.
- **Smallest-effort biggest-lift micro-move.** iter-3 — the only change was collapsing the `**RQ1.**` / `**RQ2.**` two-item blockquote into a single-paragraph blockquote with the same content. That one edit took §2.4 from 0.633 to 0.381 and brought the file to 0/17.
- **What this says about the policy.** The induced policy is real — it has causal force on classifications (per §4.3 of the thesis itself, 46% label-flip rate vs the inverted policy and a +8.9 nat margin shift). But it can be evaded by deliberate prose rewriting that targets the specific features the policy enumerates. The policy is causally load-bearing without being adversarially robust. These are different properties, and the thesis's faithfulness claim doesn't require the second.
- **Possible thesis-section consequence.** This finding is genuinely relevant to §5.4 (threats to faithfulness) and §5.5 (future work). The adversarial-rewriting evasion isn't a faithfulness failure — Madsen et al.'s critique is about whether *the explanation* is causally true, not whether the *classification* is robust to paraphrase. But it does bound a stronger claim about practical deployment: an adversary who knows the policy can rewrite around it. Worth at least one sentence in §5.5.

## External-detector transfer (GPTZero, 2026-04-26)

Pasted the rewritten v2 prose (the §1 Introduction chunk; ~470 words) into GPTZero's Basic Scan (model 4.4b). Result: **AI 100% / Mixed 0% / Human 0%, "highly confident this text was AI generated"**, with most sentences yellow-highlighted including "The contribution has two halves" and the Phase-1/Phase-2 leadup.

What this says: the local-policy evasion does **not** transfer to GPTZero's detector. The two detectors are clearly looking at different surface features. Our induced policy is an explicit, enumerable list of ~150 words pointing at a small set of stylistic markers; GPTZero is presumably a fine-tuned model that learned a distributed representation of "AI-text-ness" from training data, and that representation isn't fooled by removing the markers our policy enumerates.

This is a *real* methodological finding for the §5.5 future-work paragraph: "evading the induced policy" and "evading AI detection broadly" are distinct goals, and the gap between them is large. A practical deployment hardening story would need both: layered detectors that don't share surface features (which is what GPTZero appears to be relative to our policy), plus periodic re-induction against successful evasions.

## v3 — reasoning-guided rewriting (2026-04-26 evening)

**Setup.** New harness `code/scripts/classify_with_reasoning.py` runs *two* Ollama calls per section: the standard single-token classification and a second call that asks the same model+policy to surface which specific phrases match AI-text and human-text indicators in the policy. Output: per-section JSON with both verdict and reasoning.

**Why this was tried.** v2 had been heavily edited since iter-3 (new headline numbers, abstracts, scale-up updates), and it was now classified against the new induction-large policy `2026-04-26T17-42-47_3d67db` (different surface-feature targets than the predecessor). v3 was created as a copy of v2 to test whether reasoning-guided rewriting outperforms the score-only iteration arc.

**v3-baseline.** Same content as v2 + new policy. **6/17 AI flags**, mean P(AI) = 0.319. Sections still flagged: §2.2 (0.974), §3.1 (0.960), §3.2 (0.926), §3.4 (0.846), §2.4 (0.749), §6 (0.763). The reasoning showed the new policy's strongest single trigger is "descriptive, explanatory passages that detail processes, mechanisms, or relationships" — the *content* of a method-style thesis section trips it even when the surface register is conversational. The policy quoted specific sentences verbatim with the policy rule each matches.

**v3-iter-1.** Targeted rewrites of all 6 flagged sections, each one informed by the specific phrases the reasoning surfaced. Strategy: replace technical-overview prose with first-person narrative texture ("Honestly the first time I sketched this…", "What I was worried about with this whole approach…", "the rule I broke once early on and had to walk back"). Rewrites preserved every numerical fact and methodological claim; only the *texture* changed. Result: **0/17 AI flags, mean P(AI) = 0.012, mean margin = −6.6 nats**. Every one of the 6 flagged sections flipped, all of them deeply (margins between −6.75 and −10.19 nats). One pass, perfect hit rate.

**Why reasoning-guided was so much more efficient than score-only.** The earlier v2 score-only arc took 3 iterations of progressively more aggressive rewriting (register → structure → list-collapse) to reach 0/17. v3 got there in one shot because the reasoning told us *exactly* which sentence to rewrite and *exactly* which policy clause it tripped — no guessing at which features mattered. The asymmetry is large enough that this should be the default loop for future iterations: use the reasoning call as a per-section diagnostic and let it scope the rewrite.

**The thesis status.** v1 still frozen as the canonical first draft. v2 is the meta-experiment-rewritten version (now also bearing the new headline numbers and the v2-iter-3 evasion polish). v3 is the v2-baseline carried into the new-policy world plus targeted reasoning-guided rewrites. All three are browseable via markserv. None has been pushed to GPTZero again — the §5.5 paragraph already documents that local-policy evasion does not generalise to model-based external detectors, and the same is presumably true here.

## iter-5 — adversarial-robustness paragraph added to §5.5

Folded the finding back into the thesis as a fourth bullet in §5.5 Future work. Framed generically (the property of policy-induced detectors, not the specific meta-experiment we ran on this report) per Gustav's preference. Length: ~190 words.

The §5.5 section grew from 146w → 353w; classifier verdict stayed human (P(AI) 0.034 → 0.037, margin −3.34 → −3.26 nats). Whole-file 0/17 holds.

Confirms the section voice was already permissive enough that adding a substantive new paragraph in the same register doesn't budge the verdict.
