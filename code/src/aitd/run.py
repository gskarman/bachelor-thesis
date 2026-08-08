from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import pathlib
import sys
import time
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler

from .classifier import classify, yes_no_prob_ai
from .data import load_hc3
from .evaluation import evaluate
from .insights import write_insights
from .ollama_client import OllamaClient, UnparseableResponse


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "logs" / "runs"
RUNS_LOG = REPO_ROOT / "logs" / "RUNS.md"

console = Console()


def _mint_run_id(config: dict[str, Any]) -> str:
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:6]
    return f"{stamp}_{digest}"


def _setup_logger(run_dir: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger("aitd.run")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(RichHandler(console=console, show_path=False, markup=True))
    fh = logging.FileHandler(run_dir / "stdout.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


def _append_runs_log(run_id: str, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    auroc = metrics.get("auroc")
    tail = f" auroc={auroc:.3f}" if auroc is not None else ""
    line = (
        f"| `{run_id}` | {config['model']['name']} | "
        f"{config['data']['split']} / n={config['data'].get('sample_size', 'all')} | "
        f"f0.5={metrics['f0_5']:.3f} f1={metrics['f1']:.3f} acc={metrics['accuracy']:.3f}{tail} | "
        f"{config.get('run', {}).get('notes', '').strip() or '—'} |\n"
    )
    with RUNS_LOG.open("a") as f:
        f.write(line)


def run_experiment(config_path: pathlib.Path) -> pathlib.Path:
    config = yaml.safe_load(config_path.read_text())
    run_id = _mint_run_id(config)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    logger = _setup_logger(run_dir)
    logger.info(f"[bold]Run[/bold] {run_id}")
    logger.info(f"Model: {config['model']['name']} @ {config['model'].get('host', 'default')}")

    client = OllamaClient(
        model=config["model"]["name"],
        host=config["model"].get("host", "http://localhost:11434"),
        max_attempts=config.get("classification", {}).get("max_retries", 3),
    )
    if not client.health_check():
        logger.error("Ollama not reachable. Is `ollama serve` running?")
        sys.exit(2)

    data = load_hc3(
        split=config["data"]["split"],
        sample_size=config["data"].get("sample_size"),
        seed=config["data"].get("seed", 42),
    )
    logger.info(f"Loaded {len(data)} examples")

    cls_cfg = config.get("classification", {})
    return_logprobs = bool(cls_cfg.get("return_logprobs", False))
    top_logprobs_k = int(cls_cfg.get("top_logprobs_k", 10))
    keep_alive = cls_cfg.get("keep_alive", "1h")
    num_ctx = cls_cfg.get("num_ctx")
    num_batch = cls_cfg.get("num_batch")

    preds_file = run_dir / "predictions.jsonl"
    labels: list[int] = []
    preds: list[int] = []
    probs_ai: list[float | None] = []
    start = time.time()
    with preds_file.open("w") as f:
        for i, ex in enumerate(data, 1):
            try:
                pred = classify(
                    client,
                    ex.text,
                    num_predict=cls_cfg.get("num_predict", 16),
                    temperature=cls_cfg.get("temperature", 0.0),
                    think=cls_cfg.get("think", False),
                    return_logprobs=return_logprobs,
                    top_logprobs_k=top_logprobs_k,
                    keep_alive=keep_alive,
                    num_ctx=num_ctx,
                    num_batch=num_batch,
                )
                pred_label = pred.label
                raw = pred.raw_response
                logprobs = pred.logprobs
                err = None
            except UnparseableResponse as e:
                pred_label = -1
                raw = ""
                logprobs = None
                err = str(e)
            except Exception as e:
                pred_label = -1
                raw = ""
                logprobs = None
                err = f"{type(e).__name__}: {e}"
                logger.warning(f"[red]{i}/{len(data)}[/red] {err}")

            labels.append(ex.label)
            preds.append(pred_label)
            probs_ai.append(yes_no_prob_ai(logprobs) if logprobs else None)
            row = {
                "idx": i - 1,
                "label": ex.label,
                "pred": pred_label,
                "source": ex.source,
                "text": ex.text,
                "raw": raw,
                "error": err,
            }
            if logprobs is not None:
                row["logprobs"] = logprobs
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(data):
                logger.info(f"  {i}/{len(data)} done ({(time.time() - start):.1f}s)")

    probs_for_eval: list[float] | None = None
    if any(p is not None for p in probs_ai):
        probs_for_eval = [p if p is not None else 0.5 for p in probs_ai]
    metrics = evaluate(labels, preds, probs=probs_for_eval)
    metrics_dict = metrics.as_dict()
    (run_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))
    logger.info(f"[green]Metrics[/green] {metrics_dict}")
    insights_path = write_insights(run_dir, config, metrics_dict)
    logger.info(f"Insights: {insights_path.name}")
    _append_runs_log(run_id, config, metrics_dict)
    logger.info(f"Artifacts: {run_dir}")
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(prog="aitd-run")
    p.add_argument("--config", required=True, type=pathlib.Path, help="Path to YAML config")
    args = p.parse_args()
    if not args.config.exists():
        console.print(f"[red]Config not found:[/red] {args.config}")
        sys.exit(1)
    run_experiment(args.config)


if __name__ == "__main__":
    main()
