from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import pathlib
import re
import sys
from dataclasses import asdict, dataclass, field

import yaml
from rich.console import Console

from .classifier import classify
from .data import LABEL_AI, LABEL_HUMAN, Example, load_hc3, make_splits, splits_sha256
from .ollama_client import OllamaClient, UnparseableResponse
from .run import RUNS_DIR, _mint_run_id, _setup_logger

POLICIES_DIR = pathlib.Path(__file__).resolve().parents[3] / "logs" / "policies"
_POLICY_MD_SECTION = re.compile(r"##\s*Policy text\s*\n+(.*)", re.DOTALL)

INVERTED_POLICY = (
    "Every passage you see was written by a human. "
    "Only answer 'Yes' (AI-generated) when the text is overwhelmingly, unmistakably machine-generated — "
    "hallucinated facts, boilerplate LLM phrases, refusal language, or blatant templating. "
    "When in doubt, answer 'No' (human). Be strongly biased toward 'No'."
)

@dataclass
class PerExample:
    idx: int
    label: int
    preds: dict[str, int]
    diffs: dict[str, float | None]

@dataclass
class PairwiseDelta:
    label_change_rate: float
    mean_delta_logprob: float | None
    n_both_valid: int

@dataclass
class FaithfulnessReport:
    policy_names: list[str]
    examples_n: int
    logprobs_available: bool
    per_policy_counts: dict[str, dict[str, int]]
    per_policy_f0_5: dict[str, float]
    pairwise: dict[str, PairwiseDelta]
    rows: list[PerExample] = field(default_factory=list)

def load_policy_md(path: pathlib.Path) -> str:
    m = _POLICY_MD_SECTION.search(path.read_text())
    if not m or not m.group(1).strip():
        raise ValueError(f"Policy md missing '## Policy text' section: {path}")
    return m.group(1).strip()

def _extract_yes_no_diff(logprobs: dict[str, float] | None) -> float | None:
    if not logprobs:
        return None
    yes = no = None
    for k, v in logprobs.items():
        kl = k.strip().lower()
        if kl.startswith("yes"):
            yes = v if yes is None else max(yes, v)
        elif kl.startswith("no"):
            no = v if no is None else max(no, v)
    return None if yes is None or no is None else yes - no

def _f_beta(labels: list[int], preds: list[int], beta: float = 0.5) -> float:
    tp = fp = fn = 0
    for y, p in zip(labels, preds):
        if p == -1: continue
        if p == LABEL_AI and y == LABEL_AI: tp += 1
        elif p == LABEL_AI and y == LABEL_HUMAN: fp += 1
        elif p == LABEL_HUMAN and y == LABEL_AI: fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    denom = (beta * beta * prec) + rec
    return ((1 + beta * beta) * prec * rec / denom) if denom else 0.0

def _probe_logprobs_available(client: OllamaClient) -> bool:
    try:
        classify(client, "probe", num_predict=1, return_logprobs=True)
        return True
    except NotImplementedError:
        return False
    except Exception:
        return True

def _classify_one(client: OllamaClient, text: str, policy: str | None, *, num_predict: int, temperature: float, try_logprobs: bool, top_logprobs_k: int) -> tuple[int, float | None]:
    system_prompt = policy if policy else None
    if try_logprobs:
        try:
            p = classify(client, text, num_predict=num_predict, temperature=temperature, system_prompt=system_prompt, return_logprobs=True, top_logprobs_k=top_logprobs_k)
            return p.label, _extract_yes_no_diff(p.logprobs)
        except NotImplementedError:
            pass
        except UnparseableResponse:
            return -1, None
    try:
        p = classify(client, text, num_predict=num_predict, temperature=temperature, system_prompt=system_prompt)
        return p.label, None
    except UnparseableResponse:
        return -1, None

def run_faithfulness(client: OllamaClient, policies: dict[str, str | None], examples: list[Example], *, num_predict: int = 16, temperature: float = 0.0, top_logprobs_k: int = 10, logger=None) -> FaithfulnessReport:
    log = (lambda m: logger.info(m)) if logger is not None else (lambda m: None)
    names = list(policies.keys())
    try_logprobs = _probe_logprobs_available(client)
    log(f"logprobs_available={try_logprobs}")
    rows: list[PerExample] = []
    for i, ex in enumerate(examples, 1):
        preds: dict[str, int] = {}
        diffs: dict[str, float | None] = {}
        for name in names:
            lbl, diff = _classify_one(client, ex.text, policies[name], num_predict=num_predict, temperature=temperature, try_logprobs=try_logprobs, top_logprobs_k=top_logprobs_k)
            preds[name] = lbl
            diffs[name] = diff
        rows.append(PerExample(idx=i - 1, label=ex.label, preds=preds, diffs=diffs))
        if i % 25 == 0 or i == len(examples):
            log(f"  {i}/{len(examples)} done")
    per_counts: dict[str, dict[str, int]] = {}
    per_f05: dict[str, float] = {}
    labels = [r.label for r in rows]
    for name in names:
        col = [r.preds[name] for r in rows]
        per_counts[name] = {"yes": sum(1 for p in col if p == LABEL_AI), "no": sum(1 for p in col if p == LABEL_HUMAN), "other": sum(1 for p in col if p == -1)}
        per_f05[name] = _f_beta(labels, col, beta=0.5)
    pairwise: dict[str, PairwiseDelta] = {}
    for a, b in itertools.combinations(names, 2):
        changes = sum(1 for r in rows if r.preds[a] != r.preds[b])
        rate = changes / len(rows) if rows else 0.0
        deltas = [r.diffs[a] - r.diffs[b] for r in rows if r.diffs[a] is not None and r.diffs[b] is not None]
        mean_d = (sum(deltas) / len(deltas)) if deltas else None
        pairwise[f"{a}_vs_{b}"] = PairwiseDelta(label_change_rate=rate, mean_delta_logprob=mean_d, n_both_valid=len(deltas))
    return FaithfulnessReport(policy_names=names, examples_n=len(rows), logprobs_available=try_logprobs, per_policy_counts=per_counts, per_policy_f0_5=per_f05, pairwise=pairwise, rows=rows)

def _render_markdown(report: FaithfulnessReport, run_id: str, config: dict) -> str:
    lines: list[str] = [
        f"# Faithfulness ablation {run_id}",
        "",
        f"- Examples: {report.examples_n}",
        f"- Logprobs available: **{report.logprobs_available}**",
        f"- Policies: {', '.join(report.policy_names)}",
        f"- Model: {config['model']['name']}",
        f"- Date: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Per-policy breakdown",
        "",
        "| policy | F0.5 | yes | no | other |",
        "|---|---|---|---|---|",
    ]
    for name in report.policy_names:
        c = report.per_policy_counts[name]
        lines.append(f"| `{name}` | {report.per_policy_f0_5[name]:.3f} | {c['yes']} | {c['no']} | {c['other']} |")
    lines += ["", "## Pairwise faithfulness", "", "| pair | Δlabel rate | mean Δ(logp(yes) − logp(no)) | n logp-valid |", "|---|---|---|---|"]
    for pair, d in report.pairwise.items():
        md = f"{d.mean_delta_logprob:+.3f}" if d.mean_delta_logprob is not None else "n/a"
        lines.append(f"| `{pair}` | {d.label_change_rate:.3f} | {md} | {d.n_both_valid} |")
    lines += ["", "## Interpretation", "", "A policy is **behaviorally faithful** if label decisions and logprob margins track policy content. If `best_vs_inverted` has Δlabel rate ≈ 0 and mean Δlogp ≈ 0, the policy is a figurehead — the model ignores it. Per Jacovi & Goldberg 2020 / Lanham et al. 2023."]
    return "\n".join(lines) + "\n"

def run_ablation(config_path: pathlib.Path) -> pathlib.Path:
    config = yaml.safe_load(config_path.read_text())
    run_id = _mint_run_id(config)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    logger = _setup_logger(run_dir)
    logger.info(f"[bold]Faithfulness[/bold] {run_id}")
    client = OllamaClient(model=config["model"]["name"], host=config["model"].get("host", "http://localhost:11434"), max_attempts=config.get("classification", {}).get("max_retries", 3))
    if not client.health_check():
        logger.error("Ollama not reachable."); sys.exit(2)
    best_src = config["faithfulness"]["best_policy"]
    best_path = pathlib.Path(best_src) if pathlib.Path(best_src).is_absolute() or "/" in best_src else POLICIES_DIR / best_src
    if not best_path.exists() and not best_path.suffix:
        best_path = POLICIES_DIR / f"{best_src}.md"
    best_policy = load_policy_md(best_path)
    policies: dict[str, str | None] = {"best": best_policy, "empty": None, "inverted": INVERTED_POLICY}
    source = config["data"].get("source", "test")
    n = config["data"].get("sample_size", 100)
    all_ex = load_hc3(split=config["data"]["split"], sample_size=None, seed=config["data"].get("seed", 42), min_chars=config["data"].get("min_chars", 32))
    sp = config.get("splits", {})
    splits = make_splits(all_ex, train=sp.get("train", 0.60), val=sp.get("val", 0.20), test=sp.get("test", 0.20), seed=sp.get("seed", 42))
    logger.info(f"splits sha256={splits_sha256(splits)[:12]}... | using source={source}")
    pool = [all_ex[i] for i in splits[source]]
    humans = [e for e in pool if e.label == LABEL_HUMAN]
    ais = [e for e in pool if e.label == LABEL_AI]
    half = n // 2
    examples = humans[:half] + ais[: n - half]
    logger.info(f"ablation examples: {len(examples)} (target {n}) from {source} split")
    cls_cfg = config.get("classification", {})
    report = run_faithfulness(client, policies, examples, num_predict=cls_cfg.get("num_predict", 16), temperature=cls_cfg.get("temperature", 0.0), top_logprobs_k=cls_cfg.get("top_logprobs_k", 10), logger=logger)
    (run_dir / "faithfulness.jsonl").write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in report.rows))
    payload = asdict(report); payload.pop("rows", None)
    (run_dir / "faithfulness.json").write_text(json.dumps(payload, indent=2))
    md_path = run_dir / "faithfulness.md"
    md_path.write_text(_render_markdown(report, run_id, config))
    logger.info(f"[green]Ablation[/green] → {md_path}")
    return run_dir

def main() -> None:
    p = argparse.ArgumentParser(prog="aitd-ablate")
    p.add_argument("--config", required=True, type=pathlib.Path)
    args = p.parse_args()
    if not args.config.exists():
        Console().print(f"[red]Config not found:[/red] {args.config}"); sys.exit(1)
    run_ablation(args.config)

if __name__ == "__main__":
    main()
