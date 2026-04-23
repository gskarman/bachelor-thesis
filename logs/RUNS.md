# Experiment runs

Append-only human log of every experiment run. One row per `aitd-run` invocation.
Artifacts for each `run_id` live under `logs/runs/<run_id>/` (gitignored — regenerate from frozen `config.yaml`).

**How to use:**
- Every run auto-appends a row here via `aitd.run._append_runs_log`.
- After a run, add a short note below in the Notes section if there's something worth remembering (surprise, bug found, hypothesis change).
- Never edit existing rows — correct by adding a new note.

## Runs

| run_id | model | split / n | metrics | notes |
|--------|-------|-----------|---------|-------|

## Notes / changelog

- **2026-04-19** — Scaffolding ready. Stack: Python 3.11+, Ollama client, HuggingFace `datasets`, tenacity retry, sklearn metrics. Dev default switched to Gemma 4 E4B (`gemma4:e4b`) for fast iteration; `gemma4:31b` (already pulled locally) reserved for full-quality runs. First planned run: HC3/all n=100 seed=42, v1 yes/no prompt, no logprobs yet. Noted: Gemma 4 lineup is E2B/E4B/26B-MoE/31B — no 34B variant.
| `2026-04-19T15-20-59_2818c6` | gemma4:e4b | all / n=10 | acc=0.000 f1=0.000 | n=10 smoke test to validate end-to-end pipeline before real runs |
| `2026-04-19T15-23-31_2dc781` | gemma4:e4b | all / n=10 | acc=0.900 f1=0.909 | n=10 smoke test to validate end-to-end pipeline before real runs |
| `2026-04-19T15-25-26_50dd11` | gemma4:e4b | all / n=100 | acc=0.910 f1=0.917 | v1 yes/no classifier via Ollama (Gemma 4 E4B) on HC3 subset |
| `2026-04-19T15-41-39_50dd11` | gemma4:e4b | all / n=100 | acc=0.910 f1=0.917 | v1 yes/no classifier via Ollama (Gemma 4 E4B) on HC3 subset |
| `2026-04-23T21-10-04_6bedd0` | gemma4:e4b | induction / iters=6 | F0.5=0.978 P=1.000 R=0.900 | Thread B smoke: 5 iters, 10 seed + 20 val — validate policy-induction loop mechanics |
| `2026-04-23T21-29-57_eb3317` | gemma4:e4b | all / n=10 | f0.5=0.862 f1=0.909 acc=0.900 auroc=1.000 | n=10 smoke test to validate end-to-end pipeline before real runs |
| `2026-04-23T21-32-55_767540` | gemma4:e4b | all / n=1000 | f0.5=0.933 f1=0.951 acc=0.949 auroc=0.992 | D8 baseline — Gemma 4 E4B, HC3 all, n=1000, return_logprobs on for D6/D9 |
| `2026-04-23T21-39-50_10e2fb` | gemma4:e4b | finance / n=200 | f0.5=0.890 f1=0.925 acc=0.920 auroc=0.990 | D8 per-domain — Gemma 4 E4B, HC3 finance, n=200, return_logprobs for D9 AUROC/ECE |
| `2026-04-23T21-41-40_51da7c` | gemma4:e4b | medicine / n=200 | f0.5=0.952 f1=0.966 acc=0.965 auroc=0.996 | D8 per-domain — Gemma 4 E4B, HC3 medicine, n=200, return_logprobs for D9 AUROC/ECE |
| `2026-04-23T21-43-34_073bb2` | gemma4:e4b | open_qa / n=200 | f0.5=0.615 f1=0.695 acc=0.610 auroc=0.750 | D8 per-domain — Gemma 4 E4B, HC3 open_qa, n=200, return_logprobs for D9 AUROC/ECE |
| `2026-04-23T21-45-26_3aa623` | gemma4:e4b | reddit_eli5 / n=200 | f0.5=0.917 f1=0.943 acc=0.940 auroc=0.993 | D8 per-domain — Gemma 4 E4B, HC3 reddit_eli5, n=200, return_logprobs for D9 AUROC/ECE |
| `2026-04-23T21-47-33_ab07f5` | gemma4:e4b | wiki_csai / n=200 | f0.5=0.625 f1=0.713 acc=0.625 auroc=0.896 | D8 per-domain — Gemma 4 E4B, HC3 wiki_csai, n=200, return_logprobs for D9 AUROC/ECE |

- **2026-04-23 — Thread A E4B baseline batch (D8+D9)** — first real numbers. HC3 `all` n=1000: F0.5 **0.933**, AUROC 0.992, ECE 0.035. Per-domain n=200 on E4B spans a wide range: medicine 0.952 / reddit_eli5 0.917 / finance 0.890 are strong; open_qa **0.615** and wiki_csai **0.625** are much weaker with high ECE (0.34 each). Consistent pattern across every split: recall(AI) ≈ 0.89-0.99 but precision(AI) only 0.57-0.94 — i.e. the default prompt over-predicts "Yes" (AI). That's exactly why F0.5 < F1 and why D6 calibration matters: the threshold currently sits at the raw argmax, which is too permissive for an F0.5-optimizing classifier. wiki_csai is a small-n domain (many reddit answers are long, wiki-csai often short computer-science Q&A) — check once we have 31B numbers whether this is a domain effect or a small-model effect. Raw rows above, artifacts under `logs/runs/` (gitignored).
| `2026-04-23T21-52-51_8bb847` | gemma4:31b | all / n=10 | f0.5=1.000 f1=1.000 acc=1.000 auroc=1.000 | n=10 31B parity check — confirm logprob surface + wall-clock before scaling |
| `2026-04-23T21-55-32_8aee83` | gemma4:31b | all / n=1000 | f0.5=0.977 f1=0.984 acc=0.984 auroc=1.000 | D8 full-quality baseline — Gemma 4 31B, HC3 all, n=1000, return_logprobs for D6/D9 |
| `2026-04-23T22-40-12_9dc98f` | gemma4:31b | finance / n=200 | f0.5=0.992 f1=0.995 acc=0.995 auroc=1.000 | D8 per-domain — Gemma 4 31B, HC3 finance, n=200, return_logprobs for D9 AUROC/ECE |
| `2026-04-23T22-45-49_3d6df7` | gemma4:31b | medicine / n=200 | f0.5=0.984 f1=0.990 acc=0.990 auroc=0.995 | D8 per-domain — Gemma 4 31B, HC3 medicine, n=200, return_logprobs for D9 AUROC/ECE |
| `2026-04-23T22-50-07_657936` | gemma4:31b | open_qa / n=200 | f0.5=0.727 f1=0.810 acc=0.765 auroc=0.997 | D8 per-domain — Gemma 4 31B, HC3 open_qa, n=200, return_logprobs for D9 AUROC/ECE |
| `2026-04-23T22-53-20_2fd6a5` | gemma4:31b | reddit_eli5 / n=200 | f0.5=0.992 f1=0.995 acc=0.995 auroc=1.000 | D8 per-domain — Gemma 4 31B, HC3 reddit_eli5, n=200, return_logprobs for D9 AUROC/ECE |
| `2026-04-23T22-58-01_f8699f` | gemma4:31b | wiki_csai / n=200 | f0.5=0.868 f1=0.913 acc=0.905 auroc=0.995 | D8 per-domain — Gemma 4 31B, HC3 wiki_csai, n=200, return_logprobs for D9 AUROC/ECE |

- **2026-04-23 — Thread A 31B baseline batch (D8 full-quality)** — HC3 `all` n=1000: F0.5 **0.977**, AUROC **0.9998**, ECE 0.015. Uniformly dominates E4B on every split (Δ F0.5 +0.03 to +0.24). Biggest gains on the E4B weak spots: wiki_csai 0.625→0.868 (+0.243), open_qa 0.615→0.727 (+0.112). Two observations load-bearing for the Results section: (a) **recall(AI)=1.000 on every 31B run** — 31B never misses an AI text across 1800 examples, so the over-predict-AI bias is model-scale-invariant and entirely a prompt/calibration issue. (b) **AUROC is near-ceiling (0.995-1.000) even where F0.5 is weak** (open_qa 0.727 F0.5 / 0.997 AUROC, wiki_csai 0.868 / 0.995) — the logprob *ranking* is essentially perfect, so D6 calibration should convert the huge AUROC headroom into F0.5. That's the thesis story: the policy-induction layer decides *what* the classifier sees, and the calibration layer picks the operating point; both are needed. Wall clock: run 1 was contended with Thread B for ~20 min (evicting E4B), dropped to ~1.2-1.5 s/call after B paused — exactly the pattern A-166 (OLLAMA_MAX_LOADED_MODELS=2 + keep_alive) addresses. Also noted for D6 wiring: on 31B the runner-up token (Yes or No) often falls outside top_logprobs_k=10 because the chosen token is near-deterministic; `yes_no_prob_ai()` short-circuits correctly (1.0 or 0.0), but a calibration model that leans on `lp(yes) - lp(no)` will want k≥20 on 31B for a continuous margin feature.
