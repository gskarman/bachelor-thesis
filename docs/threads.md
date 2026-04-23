# Parallel work threads — coordination

Two Claude Code sessions work in parallel on the 7-day run to Inlämning 5 (2026-04-30).

- **Thread A — Calibration & baselines.** See `docs/thread-a-calibration.md`.
- **Thread B — Policy induction & faithfulness.** See `docs/thread-b-policy-induction.md`.

Authoritative method reference: **`docs/decisions.md`**. Do not modify without surfacing to the other thread via the `#thesis-design` channel.

---

## Shared invariants

- **Method is frozen** at v1 (see `decisions.md`). Any method change = new ADR block + ping the other thread.
- **Split is frozen.** HC3 60/20/20 train/val/test, seed=42. Test split is never seen during induction or calibration tuning.
- **Models are frozen for this cycle.** Gemma 4 E4B (iterate) + Gemma 4 31B (full). Do not reach for gpt-oss / Qwen without sync.
- **Primary metric is F0.5.** Both threads score against F0.5 unless ADR D9 changes.

---

## Shared artifacts — write-once, read-many

| Artifact | First-writer | Consumers | Notes |
|---|---|---|---|
| `code/configs/splits.yaml` | whichever thread needs it first | both | Seed + indices for HC3 60/20/20. Write once, never regenerate with a different seed mid-cycle. |
| Train split indices | first-writer (via `aitd.data.make_splits` helper) | both | Cache on disk under `logs/splits/<hash>.json`. |
| Best induced policy string | Thread B | Thread A (for calibration over frozen policy) | Saved to `logs/policies/<run_id>.md`. Thread A reads this when running calibration on Day 5. |
| Log-prob extraction path | Thread A | Thread B (optional, Day 5+) | Documented in `docs/logprob-spike.md`. |

---

## Shared files — coordinate edits

| File | Thread A changes | Thread B changes | Protocol |
|---|---|---|---|
| `code/src/aitd/classifier.py` | Add `return_logprobs: bool` flag; extend `Prediction` with `logprobs: dict[str, float] \| None` | Add `system_prompt: str \| None` param; route into Ollama `generate` as system message | Both additions are orthogonal. Whichever thread touches the file first adds **both** signatures (stub the one it doesn't need with `raise NotImplementedError`). The second thread fills in its side. |
| `code/src/aitd/ollama_client.py` | Owned by A | Read-only for B | A expands to surface log-probs. B consumes the existing `generate` signature. |
| `logs/RUNS.md` | Both append rows | Both append rows | Append-only; never rewrite existing rows. |

---

## Branching and merging

- Branch naming: `thread-a/<slug>`, `thread-b/<slug>` (e.g. `thread-a/logprob-spike`, `thread-b/policy-scaffold`).
- Short-lived branches. Open PR + merge as soon as a milestone lands. Rebase on main daily to minimize drift.
- Main is protected; every change goes through a PR (see `CLAUDE.md` git-guard).
- If both threads need to touch `classifier.py` on the same day: whichever PR opens first merges first; the second rebases.

---

## Communication

- `#thesis-design` — design questions, cross-thread decisions, "is this ADR-shaped?".
- `#thesis-runs` (new) — run announcements, "I'm about to hit 31B for 30 min, don't steal the GPU."
- End-of-day post in `#thesis-design` from each thread: 2–3 lines on what shipped and what's next. A running log across sessions.

---

## Failure modes to watch

- **Interface drift.** If A adds `return_logprobs` but B forgets `system_prompt`, both runs break. Enforce with the "add both signatures on first touch" rule.
- **Split regeneration.** Two threads generating splits with slightly different seeds = incomparable numbers. Write-once, hash-verified.
- **Policy thrash.** Thread B must **freeze** one best policy before Thread A runs calibration. Policy string is a shared artifact, not a moving target.
- **Decision drift.** If an ADR stops fitting reality, append a dated supersession block to `decisions.md` — don't edit past text.
