"""Meta-experiment harness: classify each prose section of a thesis draft against
the frozen induced policy, to measure how AI-y the writing reads section-by-section.

Skip rules (sections never sent to the classifier):

- §4 Results            — contains real run numbers, not Claude prose
- References            — citation list
- Pre-flight checklist  — checklist, not prose
- Title block           — metadata
- Sammanfattning / Abstract — empty placeholders for now

Sections classified: §1 Intro, §2.1–§2.4 Background, §3.1–§3.6 Method,
§5.1–§5.6 Discussion, §6 Conclusion. Each section is sent as a single passage
to the classifier (Gemma 4 E4B + frozen induced policy as system prompt).

Usage:
    python scripts/classify_draft_sections.py                       # default v2 draft
    python scripts/classify_draft_sections.py --draft docs/inl5-draft-v1.md
    python scripts/classify_draft_sections.py --label baseline      # custom label
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from aitd.classifier import classify, yes_no_prob_ai
from aitd.ollama_client import OllamaClient

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DRAFT = REPO / "docs" / "inl5-draft-v2.md"
POLICY_PATH = REPO / "logs" / "policies" / "2026-04-23T23-06-25_b15d8f.md"
OUT_DIR = REPO / "logs" / "meta-experiment"

YES_TOKENS = ("Yes", "yes", " Yes", " yes", "YES")
NO_TOKENS = ("No", "no", " No", " no", "NO")

SKIP_PARENT_PREFIXES = (
    "4. results",
    "results",
    "references",
    "pre-flight",
    "[title",
    "title",
    "sammanfattning",
    "abstract",
)


def _normalise(heading: str) -> str:
    s = heading.lower().strip()
    s = re.sub(r"^[#\s]*", "", s)
    return s


def is_skip_parent(h2_heading: str) -> bool:
    h = _normalise(h2_heading)
    return any(h.startswith(prefix) for prefix in SKIP_PARENT_PREFIXES)


def load_policy() -> str:
    text = POLICY_PATH.read_text(encoding="utf-8")
    return text[text.index("## Policy text") + len("## Policy text"):].strip()


def split_by_headings(md: str) -> list[tuple[int, str, str]]:
    """Walk the doc linearly, returning [(level, heading, body)] for every H2/H3."""
    pattern = re.compile(r"(?m)^(#{2,3})\s+(.+)$")
    matches = list(pattern.finditer(md))
    out = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        out.append((level, heading, body))
    return out


def build_classifiable_chunks(md: str) -> list[dict]:
    """Return a list of {parent, heading, body} for sections worth classifying.

    Strategy:
      - Walk H2/H3 in order, tracking the current H2 parent.
      - If the current H2 parent matches a skip prefix, drop the section.
      - For an H2 whose body has no H3 children (e.g. §1 Intro, §6 Conclusion),
        classify the H2 body directly.
      - For an H2 with H3 children, classify each H3 separately and skip the H2
        body (which is usually empty or a short stub).
    """
    sections = split_by_headings(md)
    # Annotate each entry with its H2 parent.
    annotated: list[tuple[int, str, str, str]] = []  # (level, heading, body, h2_parent)
    current_h2 = ""
    for level, heading, body in sections:
        if level == 2:
            current_h2 = heading
        annotated.append((level, heading, body, current_h2))

    # Decide which entries to classify.
    h2_has_h3: dict[str, bool] = {}
    for level, _heading, _body, h2 in annotated:
        if level == 3:
            h2_has_h3[h2] = True

    chunks: list[dict] = []
    for level, heading, body, h2 in annotated:
        if is_skip_parent(h2):
            continue
        if level == 2 and h2_has_h3.get(h2, False):
            # H3 children will be classified individually; skip the H2 stub
            continue
        chunks.append({"parent": h2, "heading": heading, "body": body})
    return chunks


def strip_markdown(text: str) -> str:
    """Light markdown → plain text, plus thesis-flag annotation removal."""
    # Remove [FROM V2 ...], [UPDATED ...], [TODO ...] meta-flags (with optional bold + tick wrappers)
    text = re.sub(r"\*\*`?\[(?:FROM V2|UPDATED|TODO)[^\]]*\]`?\*\*", "", text)
    text = re.sub(r"`\[(?:FROM V2|UPDATED|TODO)[^\]]*\]`", "", text)
    text = re.sub(r"\[(?:FROM V2|UPDATED|TODO)[^\]]*\]", "", text)
    # Drop block-quote markers `> `
    text = re.sub(r"(?m)^>\s?", "", text)
    # Drop horizontal rules
    text = re.sub(r"(?m)^---+\s*$", "", text)
    # Drop heading markers (### etc.) — keep the text
    text = re.sub(r"(?m)^#+\s+", "", text)
    # Bold/italic markers (keep inner text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # Links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Inline code `x`
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Tables: drop pipe-delimited rows entirely (we want prose only)
    text = re.sub(r"(?m)^\|.*\|$\n?", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def yes_no_margin(logprobs: dict[str, float] | None) -> float | None:
    if not logprobs:
        return None
    lp_yes = max((logprobs[k] for k in YES_TOKENS if k in logprobs), default=None)
    lp_no = max((logprobs[k] for k in NO_TOKENS if k in logprobs), default=None)
    if lp_yes is None or lp_no is None:
        return None
    return lp_yes - lp_no


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--label", type=str, default=None,
                        help="Run label (defaults to ISO timestamp)")
    parser.add_argument("--model", type=str, default="gemma4:e4b")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-words", type=int, default=25,
                        help="Skip sections shorter than this many words")
    args = parser.parse_args()

    draft_path = args.draft.resolve()
    label = args.label or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    md = draft_path.read_text(encoding="utf-8")
    policy = load_policy()
    client = OllamaClient(model=args.model)

    chunks = build_classifiable_chunks(md)
    print(f"[meta-experiment] draft : {draft_path.relative_to(REPO)}")
    print(f"[meta-experiment] policy: {POLICY_PATH.name} ({len(policy)} chars)")
    print(f"[meta-experiment] model : {args.model}")
    print(f"[meta-experiment] label : {label}\n")

    results = []
    for chunk in chunks:
        passage = strip_markdown(chunk["body"])
        word_count = len(passage.split())
        heading = chunk["heading"]
        parent = chunk["parent"]
        if word_count < args.min_words:
            print(f"  [skip — {word_count}w] {heading}")
            continue
        pred = classify(
            client, passage, system_prompt=policy,
            return_logprobs=True, top_logprobs_k=args.top_k, temperature=0.0,
        )
        prob_ai = yes_no_prob_ai(pred.logprobs) if pred.logprobs else None
        margin = yes_no_margin(pred.logprobs)
        label_str = {1: "AI", 0: "human", -1: "other"}.get(pred.label, str(pred.label))
        results.append({
            "parent": parent,
            "heading": heading,
            "words": word_count,
            "raw": pred.raw_response,
            "label": label_str,
            "p_ai": prob_ai,
            "margin_nats": margin,
        })
        margin_str = f"{margin:+5.2f}" if margin is not None else " n/a "
        p_str = f"{prob_ai:.3f}" if prob_ai is not None else "n/a"
        flag = "[AI]" if pred.label == 1 else "[hum]"
        print(f"  {flag} p(AI)={p_str} margin={margin_str}n  ({word_count:>4}w)  {heading}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{label}.json"
    out_path.write_text(json.dumps({
        "draft": str(draft_path.relative_to(REPO)),
        "label": label,
        "policy": str(POLICY_PATH.relative_to(REPO)),
        "model": args.model,
        "results": results,
    }, indent=2), encoding="utf-8")

    n_ai = sum(1 for r in results if r["label"] == "AI")
    n_total = len(results)
    avg_p_ai = (sum(r["p_ai"] for r in results if r["p_ai"] is not None)
                / max(1, sum(1 for r in results if r["p_ai"] is not None)))
    avg_margin = (sum(r["margin_nats"] for r in results if r["margin_nats"] is not None)
                  / max(1, sum(1 for r in results if r["margin_nats"] is not None)))
    print(f"\n[meta-experiment] {n_ai}/{n_total} sections flagged AI")
    print(f"[meta-experiment] mean P(AI)        = {avg_p_ai:.3f}")
    print(f"[meta-experiment] mean margin (nats)= {avg_margin:+.2f}")
    print(f"[meta-experiment] saved → {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
