from __future__ import annotations

import json
import pathlib
from typing import Any


def _trim(text: str, n: int = 500) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _label_name(label: int) -> str:
    return {0: "human", 1: "ai", -1: "unparseable"}.get(label, f"?{label}")


def write_insights(
    run_dir: pathlib.Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    max_examples: int = 15,
) -> pathlib.Path:
    pred_path = run_dir / "predictions.jsonl"
    rows: list[dict[str, Any]] = []
    with pred_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    misclassified = [r for r in rows if r["pred"] != -1 and r["pred"] != r["label"]]
    unparseable = [r for r in rows if r["pred"] == -1]
    false_pos = [r for r in misclassified if r["label"] == 0 and r["pred"] == 1]  # human called AI
    false_neg = [r for r in misclassified if r["label"] == 1 and r["pred"] == 0]  # AI called human

    run_id = run_dir.name
    lines = [
        f"# Run insights — `{run_id}`",
        "",
        f"**Config:** {config.get('run', {}).get('name', '—')} · model `{config['model']['name']}` · split `{config['data']['split']}` · n={config['data'].get('sample_size', 'all')} · seed={config['data'].get('seed', 42)}",
        f"**Notes:** {config.get('run', {}).get('notes', '—')}",
        "",
        "## Metrics",
        "",
        f"- n / n_valid: {metrics['n']} / {metrics['n_valid']}",
        f"- accuracy: **{metrics['accuracy']:.3f}**",
        f"- F1: **{metrics['f1']:.3f}**",
        f"- precision (AI): {metrics['precision_ai']:.3f}",
        f"- recall (AI): {metrics['recall_ai']:.3f}",
    ]
    if metrics.get("auroc") is not None:
        lines.append(f"- AUROC: {metrics['auroc']:.3f}")
    if metrics.get("ece") is not None:
        lines.append(f"- ECE: {metrics['ece']:.3f}")

    lines += [
        "",
        "## Error breakdown",
        "",
        f"- False positives (human → AI): **{len(false_pos)}**",
        f"- False negatives (AI → human): **{len(false_neg)}**",
        f"- Unparseable: **{len(unparseable)}**",
        "",
    ]

    if false_pos:
        lines += ["## False positives — humans the model flagged as AI", ""]
        for r in false_pos[:max_examples]:
            src = r.get("source", "—")
            text = _trim(r.get("text", ""))
            raw = r.get("raw", "")
            lines += [f"### idx={r['idx']} · source={src} · raw=`{raw}`", "", f"> {text}", ""]

    if false_neg:
        lines += ["## False negatives — AI the model called human", ""]
        for r in false_neg[:max_examples]:
            src = r.get("source", "—")
            text = _trim(r.get("text", ""))
            raw = r.get("raw", "")
            lines += [f"### idx={r['idx']} · source={src} · raw=`{raw}`", "", f"> {text}", ""]

    if unparseable:
        lines += ["## Unparseable responses", ""]
        for r in unparseable[:max_examples]:
            lines += [f"- idx={r['idx']} error=`{r.get('error', '')}` raw=`{r.get('raw', '')}`"]
        lines += [""]

    by_source: dict[str, dict[str, int]] = {}
    for r in rows:
        src = r.get("source", "—")
        by_source.setdefault(src, {"n": 0, "correct": 0, "fp": 0, "fn": 0, "bad": 0})
        by_source[src]["n"] += 1
        if r["pred"] == -1:
            by_source[src]["bad"] += 1
        elif r["pred"] == r["label"]:
            by_source[src]["correct"] += 1
        elif r["label"] == 0:
            by_source[src]["fp"] += 1
        else:
            by_source[src]["fn"] += 1

    lines += ["## Per-source accuracy", "", "| source | n | correct | FP (human→AI) | FN (AI→human) | invalid |", "|---|---|---|---|---|---|"]
    for src, s in sorted(by_source.items()):
        lines.append(f"| {src} | {s['n']} | {s['correct']} | {s['fp']} | {s['fn']} | {s['bad']} |")
    lines.append("")

    out = run_dir / "insights.md"
    out.write_text("\n".join(lines))
    return out
