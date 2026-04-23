# Bachelor thesis — agent guide

Repo: explainable AI text detection (KTH DM128X). Companion to `docs/operating-brief.md` (course/deadline spec — read that first if you need context on why the thesis exists).

This file is the **operational** guide: how to run experiments, log them, and keep the repo reproducible.

---

## 1. Repo layout

```
bachelor-thesis/
├── code/
│   ├── pyproject.toml          # deps + entry point (`aitd-run`)
│   ├── configs/
│   │   └── base.yaml           # default experiment config
│   └── src/aitd/
│       ├── data.py             # HC3 loader
│       ├── ollama_client.py    # Ollama wrapper + tenacity retry
│       ├── classifier.py       # prompt + yes/no parser
│       ├── evaluation.py       # F1 / AUROC / ECE
│       └── run.py              # CLI entry
├── docs/
│   ├── operating-brief.md      # course, deadlines, Canvas/Notion/Drive links
│   ├── decisions.md            # method & design ADRs — authoritative; read before proposing method changes
│   └── outline.md              # thesis report structure
├── logs/
│   ├── RUNS.md                 # append-only changelog of every run
│   └── runs/<run_id>/          # gitignored per-run artifacts
├── CLAUDE.md                   # you are here
└── README.md
```

---

## 2. Setup (one-time)

```bash
cd code
uv venv                         # or python -m venv .venv
source .venv/bin/activate
uv pip install -e .             # or pip install -e .
ollama serve &                  # if not already running
ollama pull gemma4:e4b          # dev default; pull `gemma4:31b` for full-quality runs
```

---

## 3. Running an experiment

```bash
cd code
aitd-run --config configs/base.yaml
```

Every run:
1. Reads the YAML config.
2. Mints a `run_id` = `<ISO-timestamp>_<sha1[:6] of config>`.
3. Creates `logs/runs/<run_id>/` with:
   - `config.yaml` — frozen copy of the config used.
   - `predictions.jsonl` — one row per example: `{idx, label, pred, source, raw, error}`.
   - `metrics.json` — aggregate metrics.
   - `stdout.log` — full stdout/stderr.
4. Appends a summary row to `logs/RUNS.md`.

**Reproducibility rule:** never mutate a run directory after it's written. To re-run, copy the config to a new file, adjust, and run again — this produces a new `run_id`.

---

## 4. Adding a new experiment variant

1. Copy `configs/base.yaml` → `configs/<name>.yaml`.
2. Edit the config (model tag, prompt version, data split, sample size, etc.).
3. Run it.
4. If the change was interesting, add a short note under `## Notes / changelog` in `logs/RUNS.md`.

**Configurable knobs today:**

| key | meaning | default |
|---|---|---|
| `model.name` | Ollama model tag | `gemma4:e4b` (dev default; swap to `gemma4:31b` for final runs) |
| `model.host` | Ollama URL | `http://localhost:11434` |
| `data.split` | HC3 subset (`all`, `finance`, `medicine`, `open_qa`, `reddit_eli5`, `wiki_csai`) | `all` |
| `data.sample_size` | subset size (null = full) | `100` |
| `data.seed` | sampling seed | `42` |
| `classification.num_predict` | Ollama `num_predict` | `4` |
| `classification.temperature` | generation temperature | `0.0` |
| `classification.max_retries` | tenacity retry attempts | `3` |

---

## 5. Commit conventions

- One logical change per commit.
- Subject: `<area>: <imperative>` — e.g. `classifier: tighten yes/no parsing`, `data: add balanced sampling`, `runs: log n=500 gemma3 baseline`.
- **Never commit** raw HC3 data, Ollama models, or `logs/runs/<run_id>/` contents. They're regenerable.
- **Do commit** configs under `code/configs/` and updates to `logs/RUNS.md`.
- Reference run IDs in commit messages when the change was motivated by a specific run.

---

## 6. Story-of-changes — where to put what

| thing | where |
|---|---|
| Config of a run | `logs/runs/<run_id>/config.yaml` (auto) |
| Aggregate metrics of a run | `logs/runs/<run_id>/metrics.json` (auto) |
| Per-example predictions of a run | `logs/runs/<run_id>/predictions.jsonl` (auto) |
| One-line summary of a run | `logs/RUNS.md` (auto row + optional manual note) |
| Why I chose config X | commit message referencing the run_id |
| Method / design / architecture decisions (ADR-style) | `docs/decisions.md` |
| Thesis report structure (Intro, Background, …) | `docs/outline.md` |
| Course / deadline facts | `docs/operating-brief.md` |

---

## 7. What NOT to do

- Don't commit dataset files, model weights, or `logs/runs/` artifacts.
- Don't run with `temperature > 0` without also recording the seed and noting it in RUNS.md — reproducibility matters for the thesis.
- Don't edit a past run's `config.yaml` or `predictions.jsonl`. Make a new run.
- Don't push to main with failing imports. Minimum bar before commit: `python -c "import aitd.run"`.
- Don't hit the live HuggingFace Hub from a test — cache locally or mock.

---

## 8. Open questions / TODOs

Authoritative tracker: `docs/decisions.md` (§ Open questions + each ADR's own open issues).

---

## 9. Agent operating rules

- Read `docs/operating-brief.md` before taking action on deadlines or Canvas/Notion.
- Prefer editing existing files over adding new ones. Ask before creating new top-level directories.
- Prefer small, testable modules; keep `aitd/*` files under ~200 lines.
- If a module grows past that, propose a split — don't just sprawl.
- When unsure about a model tag, dataset split, or prompt variant — **write a note in RUNS.md and ask**, don't guess silently.
