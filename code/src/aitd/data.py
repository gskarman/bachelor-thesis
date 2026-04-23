from __future__ import annotations

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
