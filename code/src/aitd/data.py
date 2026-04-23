from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Iterable

from huggingface_hub import hf_hub_download


HC3_REPO = "Hello-SimpleAI/HC3"
HC3_SPLITS = ("all", "finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")
LABEL_HUMAN = 0
LABEL_AI = 1


@dataclass(frozen=True)
class Example:
    text: str
    label: int
    source: str
    question_id: str | int | None = None


def _download_hc3_jsonl(split: str) -> str:
    return hf_hub_download(
        repo_id=HC3_REPO,
        filename=f"{split}.jsonl",
        repo_type="dataset",
    )


def load_hc3(
    split: str = "all",
    sample_size: int | None = None,
    seed: int = 42,
    min_chars: int = 32,
) -> list[Example]:
    if split not in HC3_SPLITS:
        raise ValueError(f"Unknown HC3 split {split!r}. Allowed: {HC3_SPLITS}")

    path = _download_hc3_jsonl(split)

    pairs: list[Example] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            source = ex.get("source") or split
            qid = ex.get("id")
            for text in ex.get("human_answers") or []:
                if text and len(text) >= min_chars:
                    pairs.append(Example(text=text, label=LABEL_HUMAN, source=source, question_id=qid))
            for text in ex.get("chatgpt_answers") or []:
                if text and len(text) >= min_chars:
                    pairs.append(Example(text=text, label=LABEL_AI, source=source, question_id=qid))

    if sample_size is not None and sample_size < len(pairs):
        rng = random.Random(seed)
        pairs = _balanced_sample(pairs, sample_size, rng)

    return pairs


def _balanced_sample(pairs: list[Example], n: int, rng: random.Random) -> list[Example]:
    humans = [p for p in pairs if p.label == LABEL_HUMAN]
    ais = [p for p in pairs if p.label == LABEL_AI]
    half = n // 2
    h = rng.sample(humans, min(half, len(humans)))
    a = rng.sample(ais, min(n - len(h), len(ais)))
    out = h + a
    rng.shuffle(out)
    return out


def iter_batches(items: Iterable[Example], size: int) -> Iterable[list[Example]]:
    batch: list[Example] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def make_splits(
    examples: list[Example],
    train: float = 0.60,
    val: float = 0.20,
    test: float = 0.20,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Deterministic balanced train/val/test split on `examples` indices.

    Shuffles per-label with a seeded RNG so every split is class-balanced.
    The returned indices refer to positions in `examples` as given.
    """
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        by_label.setdefault(ex.label, []).append(i)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idxs in by_label.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train)
        n_val = int(n * val)
        out["train"].extend(idxs[:n_train])
        out["val"].extend(idxs[n_train : n_train + n_val])
        out["test"].extend(idxs[n_train + n_val :])
    for k in out:
        out[k].sort()
    return out


def splits_sha256(splits: dict[str, list[int]]) -> str:
    """SHA256 of the canonical JSON of a splits dict. Use for cross-thread hash-verification."""
    payload = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
