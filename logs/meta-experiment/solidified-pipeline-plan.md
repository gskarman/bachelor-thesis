# Solidified ML pipeline — 2026-04-26

After the GPTZero negative result on the adversarial-rewriting evasion (the v2 prose passes our locally-induced policy at 0/17 but GPTZero rates it 100% AI), Gustav redirected back to solidifying the actual thesis ML pipeline at much larger scale on the small (E4B) classifier.

## What was

- **Induction**: pool=20 seed examples, scoring_sample=200, max_iters=30 with plateau Δ<0.005×3.
  Winner from `2026-04-23T23-06-25_b15d8f`: F0.5 = 0.980 on n=200 train-val (3 accepted iters, early-stopped at iter 7).
- **Calibration**: sample_size=1000 → val=200 + test=200 examples on E4B + frozen winner policy.
  Result `2026-04-23T23-23-21_538a36`: F0.5 = 0.942, AUROC = 0.993, ECE = 0.030 on test=200.
- **Faithfulness ablation**: n=100 test, three policies. F0.5(best) = 0.996, label-flip rate(best vs inverted) = 0.460.

These all use small-n slices of HC3. The full split (computed today) is **train = 51,206 / val = 17,068 / test = 17,072** — so the prior runs were touching at most 1.2% of the available test split.

## What's running now

Two chained runs; total wall-clock ~5h.

### Stage 1 — `induction-large` (started 2026-04-26 17:42 CEST)

```
configs/induction-large.yaml
  model:        gemma4:e4b
  seed.pool_size:        30   (was 20)
  induction.scoring_sample: 500  (was 200)
  induction.misclassified_k: 25  (was 20)
  induction.max_iters:   30  (unchanged)
  plateau_threshold/window: 0.005 × 3 (unchanged)
```

Goal: a more robust frozen policy than the n=200 winner. Rationale for the dial choices: doubling the seed pool gives the proposer more diversity to work from on iter 0; 2.5x scoring sample reduces variance in the F0.5 signal that decides accept/reject; misclassified_k bumped to 25 to keep the ratio of misclassified examples to total roughly proportional.

Expected wall-clock per iter: 500 × ~1.5s = ~12.5 min. With the same temperature-zero-deterministic refiner behaviour seen in the n=200 run (3 useful iters then 5 consecutive identical rejected candidates → early-stop), expected completion: ~1.5h.

### Stage 2 — `calibration-large` (will be kicked off after Stage 1 lands)

```
configs/calibration-large.yaml
  splits.sample_size:    20000  (was 1000)
  → balanced 10K human / 10K AI
  → val = 4000, test = 4000   (was val=200, test=200)
  policy.path:           logs/policies/<id-from-stage-1>.md
  classification.top_logprobs_k: 20 (unchanged)
```

This is the headline pipeline run. With val=4000 and test=4000 we get 95% CIs on F0.5 / AUROC / ECE that are roughly √(200/4000) = 0.22x the width of the n=200 estimates — i.e. much tighter and more honest numbers for the §4.1 thesis table.

Expected wall-clock: 8000 classifications × ~1.5s = ~3.3h.

## What I'll update once results land

1. `logs/RUNS.md` — auto-appended by `aitd-induce` and `aitd-calibrate`.
2. `docs/inl5-draft-v1.md` and `docs/inl5-draft-v2.md` §4.1 Table 1 — replace the n=200 calibration row with the n=4000 row, plus any per-domain breakdowns we re-run at larger n.
3. `docs/decisions.md` — append a 2026-04-26 ADR block documenting the scale-up choice and why we kept E4B (faster iteration, 31B comparison stays at default-prompt-only as already noted).
4. This file — final results section with the headline numbers and any surprises.

## Why E4B-only

Per Gustav's directive ("let's do mainly the small model in this one"). The 31B run results from 2026-04-23 are still the strongest detector benchmarked in the thesis (F0.5 = 0.977 default-prompt on n=1000), and there's no need to re-run them at scale to tell the story §4.1 already tells. The interesting question for the policy-induction half of the thesis is whether the small-model + induced-policy story holds up at proper test-set scale, which is exactly what these two runs answer.

---

## Final results — 2026-04-26 evening

**Stage 1 — Induction at scale.** Run `2026-04-26T17-42-47_3d67db`. Pool=30, scoring=500, early-stop at iter 6 after one accepted revision (iter 1, F0.5 = 0.956 on n=500 val) and five identical rejected candidates. Wall-clock 36 min. Healthy 0.024 drop from the n=200 winner's 0.980 — small-sample bias correction. The new policy is textually similar to the predecessor but mentions concrete transitional phrases ("First and foremost", "In addition", "Furthermore") as AI markers and "EDIT," self-corrections as a human signal — both surfaced by the larger seed pool.

**Stage 2 — Calibration at scale.** Run `2026-04-26T19-07-51_137899`. n=20000 balanced → val=4000 + test=4000. T2 logistic chosen on val. **Test F0.5 = 0.9337**, precision = 0.9432, recall = 0.8975, **AUROC = 0.9817**, **ECE = 0.0132**. Wall-clock 47 min on the second attempt — first attempt stalled at 3150/4000 val due to Ollama swap eviction; recovered after the harness-resilience patch (D13 in `docs/decisions.md`).

**Stage 3 — Faithfulness ablation against the new policy.** Run `2026-04-26T20-23-43_2f80b2`. n=100 HC3 test, three policies (best, empty, inverted). F0.5(best)=0.969, F0.5(empty)=0.965, F0.5(inverted)=0.242. Pairwise: **best_vs_inverted Δlabel = 0.490, mean Δ(lp_yes − lp_no) = +9.565 nats** — slightly stronger behavioural-faithfulness signal than the prior policy's 0.460 / +8.896 on the same n. The §5.4 Madsen pre-emption argument holds at least as well with the new policy.

## Key findings from the scale-up

1. **Small-sample bias was real.** Headline F0.5 dropped 0.942 → 0.934 (calibrated test) and AUROC 0.993 → 0.982 going from n=200 to n=4000. Both within ±0.01 of the original numbers, so the qualitative claim holds, but the new figures are honest five-times-tighter estimates.
2. **Calibration earns its keep at proper scale.** At n=200 the T1/T2 calibrators produced *identical* hard predictions to raw argmax. At n=4000 the T2 logistic measurably moves the operating point: precision 0.913 → 0.943, recall 0.964 → 0.898 (trading recall for precision in the F0.5-direction the use case demands), and ECE drops 74% from 0.050 → 0.013 (vs 29% at n=200).
3. **Faithfulness signal is stronger with the new policy.** Same n=100 test, same protocol, 0.460 → 0.490 label flip, +8.9 → +9.6 nat margin. The larger seed pool produced a policy the model leans on harder.
4. **Operationally, the harness needed work.** The resilience changes (D13) — keep_alive=4h on every Ollama request, 120s httpx timeout, incremental JSONL writes, `--resume <run_id>` — turned a stalling first attempt into a clean second attempt. They also revealed that wall-clock per E4B classification is much faster than the earlier estimates suggested when the model stays warm: 2.8 examples/s rather than 1.5/s.

## What got documented

- `docs/decisions.md` D12 (scaled re-run on E4B) and D13 (harness resilience).
- `docs/inl5-draft-v2.md` §4.1 Table 1 (n=4000 row + paragraph), §4.2 (new trajectory + new policy text), §4.3 (new ablation tables), §5.1 / §5.4 / §5.6 / §6 (headline numbers updated throughout).
- `logs/RUNS.md` — auto-appended for all three new run rows.
- `code/configs/{induction,calibration,faithfulness}-large.yaml` — frozen at the values used.
- `code/src/aitd/ollama_client.py` — keep_alive=4h, 120s timeout, expanded retry exception set.
- `code/src/aitd/calibration.py` — incremental JSONL writes, resume support, ETA in progress logs, `--resume <run_id>` CLI flag.
- This file — final results section.
