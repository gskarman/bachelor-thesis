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
