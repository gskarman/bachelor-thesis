"""Per-section meta-experiment with reasoning.

For each prose section of a thesis draft, do TWO Ollama calls:

  1. Standard classification (single-token yes/no with logprobs) — same as
     classify_draft_sections.py.
  2. Reasoning call — ask the same model with the same policy as system prompt
     to identify which specific phrases in the passage match AI-text and
     human-text indicators in the policy, and which side dominated.

The reasoning text is what informs targeted rewrites: instead of guessing
which structural patterns trip the policy, we let the model surface them
explicitly. Output JSON with both verdict and reasoning per section.

Usage:
    python scripts/classify_with_reasoning.py --draft docs/inl5-draft-v3.md --label v3-baseline
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aitd.classifier import classify, yes_no_prob_ai
from aitd.ollama_client import OllamaClient

REPO = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO / "logs" / "policies" / "2026-04-26T17-42-47_3d67db.md"
OUT_DIR = REPO / "logs" / "meta-experiment"

YES_TOKENS = ("Yes", "yes", " Yes", " yes", "YES")
NO_TOKENS = ("No", "no", " No", " no", "NO")

SKIP_PARENT_PREFIXES = (
    "4. results", "results",
    "references",
    "pre-flight",
    "[title", "title",
    "sammanfattning",
    "abstract",
)

REASONING_PROMPT = """You are an expert text-origin classifier. Below is your decision policy and a passage to evaluate.

Policy:
\"\"\"
{policy}
\"\"\"

Passage:
\"\"\"
{text}
\"\"\"

Identify which features of the passage match the policy. Respond in this exact format with no preamble:

AI INDICATORS:
- "<short verbatim quote>" — <which policy rule>
- "<short verbatim quote>" — <which policy rule>
- "<short verbatim quote>" — <which policy rule>

HUMAN INDICATORS:
- "<short verbatim quote>" — <which policy rule>
- "<short verbatim quote>" — <which policy rule>
- "<short verbatim quote>" — <which policy rule>

VERDICT: <AI or human>
PRIMARY DRIVER: <one sentence on which side dominated and why>
"""


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


def build_chunks(md: str) -> list[dict]:
    sections = split_by_headings(md)
    annotated = []
    current_h2 = ""
    for level, heading, body in sections:
        if level == 2:
            current_h2 = heading
        annotated.append((level, heading, body, current_h2))
    h2_has_h3 = {}
    for level, _, _, h2 in annotated:
        if level == 3:
            h2_has_h3[h2] = True
    chunks = []
    for level, heading, body, h2 in annotated:
        if is_skip_parent(h2):
            continue
        if level == 2 and h2_has_h3.get(h2, False):
            continue
        chunks.append({"parent": h2, "heading": heading, "body": body})
    return chunks


def strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*`?\[(?:FROM V2|UPDATED|TODO)[^\]]*\]`?\*\*", "", text)
    text = re.sub(r"`\[(?:FROM V2|UPDATED|TODO)[^\]]*\]`", "", text)
    text = re.sub(r"\[(?:FROM V2|UPDATED|TODO)[^\]]*\]", "", text)
    text = re.sub(r"(?m)^>\s?", "", text)
    text = re.sub(r"(?m)^---+\s*$", "", text)
    text = re.sub(r"(?m)^#+\s+", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"(?m)^\|.*\|$\n?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def yes_no_margin(logprobs):
    if not logprobs:
        return None
    lp_yes = max((logprobs[k] for k in YES_TOKENS if k in logprobs), default=None)
    lp_no = max((logprobs[k] for k in NO_TOKENS if k in logprobs), default=None)
    if lp_yes is None or lp_no is None:
        return None
    return lp_yes - lp_no


def reasoning_call(client: OllamaClient, policy: str, text: str) -> str:
    prompt = REASONING_PROMPT.format(policy=policy, text=text)
    result = client.generate(prompt, num_predict=400, temperature=0.0)
    return result.text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--model", type=str, default="gemma4:e4b")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-words", type=int, default=25)
    args = parser.parse_args()

    draft_path = args.draft.resolve()
    label = args.label or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    md = draft_path.read_text(encoding="utf-8")
    policy = load_policy()
    client = OllamaClient(model=args.model)

    chunks = build_chunks(md)
    print(f"[reasoning-experiment] draft  : {draft_path.relative_to(REPO)}")
    print(f"[reasoning-experiment] policy : {POLICY_PATH.name} ({len(policy)} chars)")
    print(f"[reasoning-experiment] model  : {args.model}")
    print(f"[reasoning-experiment] label  : {label}\n")

    results = []
    for chunk in chunks:
        heading = chunk["heading"]
        passage = strip_markdown(chunk["body"])
        words = len(passage.split())
        if words < args.min_words:
            print(f"  [skip — {words}w] {heading}")
            continue
        pred = classify(
            client, passage, system_prompt=policy,
            return_logprobs=True, top_logprobs_k=args.top_k, temperature=0.0,
        )
        prob_ai = yes_no_prob_ai(pred.logprobs) if pred.logprobs else None
        margin = yes_no_margin(pred.logprobs)
        label_str = {1: "AI", 0: "human", -1: "other"}.get(pred.label, str(pred.label))

        try:
            reasoning = reasoning_call(client, policy, passage)
        except Exception as e:
            reasoning = f"<reasoning call failed: {type(e).__name__}: {e}>"

        results.append({
            "parent": chunk["parent"],
            "heading": heading,
            "words": words,
            "label": label_str,
            "p_ai": prob_ai,
            "margin_nats": margin,
            "reasoning": reasoning,
        })
        flag = "[AI]" if pred.label == 1 else "[hum]"
        p_str = f"{prob_ai:.3f}" if prob_ai is not None else "n/a"
        m_str = f"{margin:+5.2f}" if margin is not None else " n/a "
        print(f"  {flag} p={p_str} m={m_str}n  ({words:>4}w)  {heading}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{label}-with-reasoning.json"
    out_path.write_text(json.dumps({
        "draft": str(draft_path.relative_to(REPO)),
        "label": label,
        "policy": str(POLICY_PATH.relative_to(REPO)),
        "model": args.model,
        "results": results,
    }, indent=2), encoding="utf-8")

    n_ai = sum(1 for r in results if r["label"] == "AI")
    avg_p = sum(r["p_ai"] for r in results if r["p_ai"] is not None) / max(1, sum(1 for r in results if r["p_ai"] is not None))
    print(f"\n[reasoning-experiment] {n_ai}/{len(results)} sections flagged AI")
    print(f"[reasoning-experiment] mean P(AI) = {avg_p:.3f}")
    print(f"[reasoning-experiment] saved → {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
