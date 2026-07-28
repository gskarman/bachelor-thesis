"""D6 calibration — fit a threshold / logistic classifier over first-token logprobs.

Loads a frozen policy (produced by aitd.policy.run_induction), runs the classifier
over val+test with `system_prompt=policy` and `return_logprobs=True`, fits two
candidate calibrators on val, picks the one with higher F0.5 on val, and reports
on test: F0.5 at the chosen operating point, precision/recall, AUROC, ECE, and
reliability-diagram bins.

Two calibrators (per `docs/decisions.md` D6):
- T1 — single threshold on `margin = lp(yes) - lp(no)`. Grid-search margin; pick
  the value that maximizes F0.5 on val.
- T2 — logistic regression on `{lp(yes), lp(no), margin}`. Score probability is
  the positive-class sigmoid output; threshold on that probability.

Outputs `logs/runs/<run_id>/calibration.json` and appends a row to `logs/RUNS.md`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import pathlib
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import yaml
from rich.console import Console
from rich.logging import RichHandler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, roc_auc_score

from .classifier import NO_TOKENS, YES_TOKENS, _best_logprob, classify
from .data import LABEL_AI, LABEL_HUMAN, Example, load_hc3, make_splits
from .ollama_client import OllamaClient, UnparseableResponse


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "logs" / "runs"
RUNS_LOG = REPO_ROOT / "logs" / "RUNS.md"

console = Console()


# --- Policy parsing --------------------------------------------------------------

_POLICY_BODY = re.compile(r"##\s*Policy text\s*\n+(.*)$", re.DOTALL | re.IGNORECASE)


def parse_policy_md(path: pathlib.Path) -> str:
    """Extract the policy text from a `logs/policies/<id>.md` file.

    Matches `aitd.policy.run_induction`'s writer: a markdown header block
    followed by `## Policy text` and the policy body.
    """
    text = path.read_text()
    m = _POLICY_BODY.search(text)
    if not m:
        raise ValueError(f"No '## Policy text' section found in {path}")
    body = m.group(1).strip()
    if not body:
        raise ValueError(f"Empty policy body in {path}")
    return body


# --- Feature extraction ----------------------------------------------------------


@dataclass
class FeatureRow:
    idx: int
    label: int
    source: str
    lp_yes: float | None
    lp_no: float | None
    margin: float | None  # lp_yes - lp_no when both present
    pred_hard: int  # the model's raw argmax label (from classify())
    error: str | None


def _make_feature_row(
    idx: int, ex: Example, pred_label: int, logprobs: dict[str, float] | None, error: str | None
) -> FeatureRow:
    lp_yes = _best_logprob(logprobs, YES_TOKENS) if logprobs else None
    lp_no = _best_logprob(logprobs, NO_TOKENS) if logprobs else None
    margin = (lp_yes - lp_no) if (lp_yes is not None and lp_no is not None) else None
    return FeatureRow(
        idx=idx, label=ex.label, source=ex.source,
        lp_yes=lp_yes, lp_no=lp_no, margin=margin,
        pred_hard=pred_label, error=error,
    )


def extract_features(
    client: OllamaClient,
    policy: str,
    examples: list[Example],
    *,
    top_logprobs_k: int,
    num_predict: int,
    temperature: float,
    think: bool,
    logger: logging.Logger,
    out_path: pathlib.Path | None = None,
) -> list[FeatureRow]:
    """Classify each example and (optionally) write each result to `out_path` as a
    JSONL line as it lands. If `out_path` already exists, existing rows are loaded
    and their indices skipped — a crashed/killed run can resume without losing work.
    """
    rows: list[FeatureRow] = []
    done_idxs: set[int] = set()

    if out_path is not None and out_path.exists():
        with out_path.open() as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rows.append(FeatureRow(**d))
                    done_idxs.add(d["idx"])
                except Exception as e:
                    logger.warning(f"  bad jsonl row in {out_path.name}: {e}")
        if done_idxs:
            logger.info(f"  resume: loaded {len(done_idxs)} existing rows from {out_path.name}")

    total_to_classify = len(examples) - len(done_idxs)
    if total_to_classify == 0:
        logger.info(f"  all {len(examples)} examples already classified — skipping")
        rows.sort(key=lambda r: r.idx)
        return rows

    f_out = out_path.open("a") if out_path is not None else None
    new_done = 0
    start = time.time()
    try:
        for i, ex in enumerate(examples, 1):
            idx = i - 1
            if idx in done_idxs:
                continue
            try:
                pred = classify(
                    client, ex.text,
                    num_predict=num_predict, temperature=temperature, think=think,
                    system_prompt=policy,
                    return_logprobs=True, top_logprobs_k=top_logprobs_k,
                )
                row = _make_feature_row(idx, ex, pred.label, pred.logprobs, None)
            except UnparseableResponse as e:
                row = _make_feature_row(idx, ex, -1, None, str(e))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                logger.warning(f"[red]{i}/{len(examples)}[/red] {err}")
                row = _make_feature_row(idx, ex, -1, None, err)
            new_done += 1
            rows.append(row)
            if f_out is not None:
                f_out.write(json.dumps(asdict(row)) + "\n")
                f_out.flush()
            if new_done % 50 == 0 or new_done == total_to_classify:
                elapsed = time.time() - start
                rate = new_done / max(elapsed, 1.0)
                eta_min = (total_to_classify - new_done) / max(rate, 1e-6) / 60.0
                logger.info(
                    f"  {new_done}/{total_to_classify} new ({elapsed:.0f}s, {rate:.1f}/s, ETA {eta_min:.1f}min)"
                )
    finally:
        if f_out is not None:
            f_out.close()

    rows.sort(key=lambda r: r.idx)
    return rows


# --- Calibrators -----------------------------------------------------------------


def _impute_margin(rows: list[FeatureRow]) -> np.ndarray:
    """Margin as a usable continuous feature. Falls back when lp is missing:
    - both present → lp_yes - lp_no
    - only yes → +LARGE
    - only no  → -LARGE
    - neither  → 0 (no info)
    """
    LARGE = 20.0
    out = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        if r.margin is not None:
            out[i] = r.margin
        elif r.lp_yes is not None and r.lp_no is None:
            out[i] = +LARGE
        elif r.lp_yes is None and r.lp_no is not None:
            out[i] = -LARGE
        else:
            out[i] = 0.0
    return out


def _impute_lp(rows: list[FeatureRow], attr: str) -> np.ndarray:
    """lp_yes/lp_no as features; missing → floor (e.g. -LARGE)."""
    FLOOR = -20.0
    return np.asarray(
        [getattr(r, attr) if getattr(r, attr) is not None else FLOOR for r in rows],
        dtype=float,
    )


@dataclass
class ThresholdModel:
    threshold: float  # decide AI if margin >= threshold
    f0_5_val: float

    def predict(self, margin: np.ndarray) -> np.ndarray:
        return (margin >= self.threshold).astype(int)

    def score(self, margin: np.ndarray) -> np.ndarray:
        """Score usable for AUROC — the raw margin itself (monotonic in p(AI))."""
        return margin.astype(float)


def fit_threshold(margin_val: np.ndarray, labels_val: np.ndarray) -> ThresholdModel:
    """Grid-search margin threshold to maximize F0.5 on val."""
    # candidate thresholds: the midpoints between consecutive sorted margins,
    # plus sentinels below and above the range.
    sorted_m = np.sort(np.unique(margin_val))
    if len(sorted_m) == 1:
        candidates = np.array([sorted_m[0] - 1e-6, sorted_m[0] + 1e-6])
    else:
        midpoints = (sorted_m[:-1] + sorted_m[1:]) / 2.0
        candidates = np.concatenate([[sorted_m[0] - 1.0], midpoints, [sorted_m[-1] + 1.0]])
    best_f = -1.0
    best_t = 0.0
    for t in candidates:
        preds = (margin_val >= t).astype(int)
        f = float(fbeta_score(labels_val, preds, beta=0.5, zero_division=0))
        if f > best_f:
            best_f = f
            best_t = float(t)
    return ThresholdModel(threshold=best_t, f0_5_val=best_f)


@dataclass
class LogisticModel:
    coef: list[float]
    intercept: float
    feature_names: list[str]
    decision_threshold: float  # on sigmoid(probability); default 0.5, tuned to maximize F0.5
    f0_5_val: float
    _model: Any = None  # the fitted sklearn estimator; not serialized

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.score(X)
        return (proba >= self.decision_threshold).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """P(AI) for each row; used for AUROC/ECE and the decision."""
        if self._model is None:
            # fallback: reconstruct from coef/intercept (sigmoid)
            z = X @ np.array(self.coef) + self.intercept
            return 1.0 / (1.0 + np.exp(-z))
        return self._model.predict_proba(X)[:, 1]


def fit_logistic(X_val: np.ndarray, labels_val: np.ndarray, feature_names: list[str]) -> LogisticModel:
    """Logistic regression on val features; decision threshold tuned on val to max F0.5."""
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(X_val, labels_val)
    proba = lr.predict_proba(X_val)[:, 1]
    # sweep decision threshold on the sigmoid
    grid = np.unique(np.concatenate([[0.0, 1.0], proba, (proba[:-1] + proba[1:]) / 2.0 if len(proba) > 1 else proba]))
    best_f = -1.0
    best_t = 0.5
    for t in grid:
        preds = (proba >= t).astype(int)
        f = float(fbeta_score(labels_val, preds, beta=0.5, zero_division=0))
        if f > best_f:
            best_f = f
            best_t = float(t)
    return LogisticModel(
        coef=lr.coef_[0].tolist(),
        intercept=float(lr.intercept_[0]),
        feature_names=feature_names,
        decision_threshold=best_t,
        f0_5_val=best_f,
        _model=lr,
    )


# --- Metrics on test -------------------------------------------------------------


def _expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins[1:-1])
    ece = 0.0
    reliability: list[dict] = []
    for b in range(n_bins):
        in_bin = idx == b
        if not in_bin.any():
            reliability.append({"bin": b, "low": float(bins[b]), "high": float(bins[b + 1]), "n": 0})
            continue
        acc = float((y[in_bin] == LABEL_AI).mean())
        conf = float(probs[in_bin].mean())
        fraction = float(in_bin.mean())
        ece += fraction * abs(acc - conf)
        reliability.append({
            "bin": b, "low": float(bins[b]), "high": float(bins[b + 1]),
            "n": int(in_bin.sum()), "conf": conf, "acc": acc,
        })
    return float(ece), reliability


def report_on_test(
    model_name: str, scores: np.ndarray, preds: np.ndarray, labels: np.ndarray,
) -> dict[str, Any]:
    mask = preds != -1
    y = labels[mask]
    p = preds[mask]
    s = scores[mask]
    if len(y) == 0:
        return {
            "model": model_name,
            "n": int(len(labels)),
            "n_valid": 0,
            "f0_5": 0.0, "precision_ai": 0.0, "recall_ai": 0.0, "accuracy": 0.0,
            "auroc": None, "ece": 0.0, "reliability": [],
        }
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f0_5 = float(fbeta_score(y, p, beta=0.5, zero_division=0))
    # AUROC uses the raw monotonic score (margin or sigmoid prob)
    auroc = float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else None
    # ECE needs probabilities in [0, 1]; for threshold model we map margin via sigmoid
    if s.min() < 0 or s.max() > 1:
        probs = 1.0 / (1.0 + np.exp(-s))
    else:
        probs = s
    ece, reliability = _expected_calibration_error(y, probs)
    return {
        "model": model_name,
        "n": int(len(labels)),
        "n_valid": int(mask.sum()),
        "f0_5": f0_5,
        "precision_ai": float(precision),
        "recall_ai": float(recall),
        "accuracy": float((p == y).mean()) if len(p) else 0.0,
        "auroc": auroc,
        "ece": ece,
        "reliability": reliability,
    }


# --- Run driver ------------------------------------------------------------------


def _mint_run_id(config: dict[str, Any]) -> str:
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:6]
    return f"{stamp}_{digest}"


def _setup_logger(run_dir: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger("aitd.calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(RichHandler(console=console, show_path=False, markup=True))
    fh = logging.FileHandler(run_dir / "stdout.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


def _append_runs_log(run_id: str, config: dict[str, Any], best: dict[str, Any]) -> None:
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    auroc = best.get("auroc")
    tail = f" auroc={auroc:.3f}" if auroc is not None else ""
    line = (
        f"| `{run_id}` | {config['model']['name']} | "
        f"calibration / test n={best['n']} | "
        f"f0.5={best['f0_5']:.3f} ece={best['ece']:.3f}{tail} (model={best['model']}) | "
        f"{config.get('run', {}).get('notes', '').strip() or 'D6 calibration'} |\n"
    )
    with RUNS_LOG.open("a") as f:
        f.write(line)


def run_calibration(config_path: pathlib.Path, *, resume_run_id: str | None = None) -> pathlib.Path:
    config = yaml.safe_load(config_path.read_text())
    if resume_run_id is not None:
        run_id = resume_run_id
        run_dir = RUNS_DIR / run_id
    else:
        run_id = _mint_run_id(config)
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    logger = _setup_logger(run_dir)
    logger.info(f"[bold]Calibration[/bold] {run_id}{' (resumed)' if resume_run_id else ''}")

    # --- Resolve policy
    policy_path = pathlib.Path(config["policy"]["path"])
    if not policy_path.is_absolute():
        policy_path = REPO_ROOT / policy_path
    policy = parse_policy_md(policy_path)
    logger.info(f"Policy: {policy_path.name} ({len(policy)} chars)")

    # --- Client
    client = OllamaClient(
        model=config["model"]["name"],
        host=config["model"].get("host", "http://localhost:11434"),
        max_attempts=config.get("classification", {}).get("max_retries", 3),
    )
    if not client.health_check():
        logger.error("Ollama not reachable. Is `ollama serve` running?")
        sys.exit(2)

    # --- Load data + make splits
    splits_cfg = config["splits"]
    data = load_hc3(
        split=splits_cfg.get("split", "all"),
        sample_size=splits_cfg.get("sample_size"),
        seed=splits_cfg.get("seed", 42),
        min_chars=splits_cfg.get("min_chars", 32),
    )
    s = make_splits(
        data,
        train=splits_cfg.get("train", 0.60),
        val=splits_cfg.get("val", 0.20),
        test=splits_cfg.get("test", 0.20),
        seed=splits_cfg.get("seed", 42),
    )
    logger.info(f"Data: n={len(data)}, val={len(s['val'])}, test={len(s['test'])}")

    # --- Extract features on val + test
    cls_cfg = config.get("classification", {})
    val_ex = [data[i] for i in s["val"]]
    test_ex = [data[i] for i in s["test"]]
    logger.info("Extracting logprobs on val…")
    val_rows = extract_features(
        client, policy, val_ex,
        top_logprobs_k=cls_cfg.get("top_logprobs_k", 25),
        num_predict=cls_cfg.get("num_predict", 1),
        temperature=cls_cfg.get("temperature", 0.0),
        think=cls_cfg.get("think", False),
        logger=logger,
        out_path=run_dir / "features_val.jsonl",
    )
    logger.info("Extracting logprobs on test…")
    test_rows = extract_features(
        client, policy, test_ex,
        top_logprobs_k=cls_cfg.get("top_logprobs_k", 25),
        num_predict=cls_cfg.get("num_predict", 1),
        temperature=cls_cfg.get("temperature", 0.0),
        think=cls_cfg.get("think", False),
        logger=logger,
        out_path=run_dir / "features_test.jsonl",
    )

    # --- Build feature matrices
    y_val = np.asarray([r.label for r in val_rows])
    y_test = np.asarray([r.label for r in test_rows])
    m_val = _impute_margin(val_rows)
    m_test = _impute_margin(test_rows)
    lpy_val, lpn_val = _impute_lp(val_rows, "lp_yes"), _impute_lp(val_rows, "lp_no")
    lpy_test, lpn_test = _impute_lp(test_rows, "lp_yes"), _impute_lp(test_rows, "lp_no")
    X_val = np.column_stack([lpy_val, lpn_val, m_val])
    X_test = np.column_stack([lpy_test, lpn_test, m_test])
    feat_names = ["lp_yes", "lp_no", "margin"]

    # --- Fit calibrators
    t_model = fit_threshold(m_val, y_val)
    l_model = fit_logistic(X_val, y_val, feat_names)
    logger.info(f"T1 threshold={t_model.threshold:.3f} val_F0.5={t_model.f0_5_val:.3f}")
    logger.info(f"T2 logistic decision_threshold={l_model.decision_threshold:.3f} val_F0.5={l_model.f0_5_val:.3f}")

    # --- Pick winner on val, evaluate on test
    chosen_name = "T2_logistic" if l_model.f0_5_val >= t_model.f0_5_val else "T1_threshold"

    t_report = report_on_test(
        "T1_threshold",
        scores=t_model.score(m_test),
        preds=t_model.predict(m_test),
        labels=y_test,
    )
    l_report = report_on_test(
        "T2_logistic",
        scores=l_model.score(X_test),
        preds=l_model.predict(X_test),
        labels=y_test,
    )
    # Raw (argmax, uncalibrated) baseline for reference
    raw_preds = np.asarray([r.pred_hard for r in test_rows])
    raw_scores = 1.0 / (1.0 + np.exp(-m_test))
    raw_report = report_on_test("raw_argmax", scores=raw_scores, preds=raw_preds, labels=y_test)

    # --- Persist
    out = {
        "run_id": run_id,
        "policy_path": str(policy_path),
        "policy_len": len(policy),
        "model": config["model"]["name"],
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "chosen": chosen_name,
        "t1_threshold": {
            "threshold": t_model.threshold,
            "f0_5_val": t_model.f0_5_val,
            "test": t_report,
        },
        "t2_logistic": {
            "coef": l_model.coef,
            "intercept": l_model.intercept,
            "feature_names": l_model.feature_names,
            "decision_threshold": l_model.decision_threshold,
            "f0_5_val": l_model.f0_5_val,
            "test": l_report,
        },
        "raw_argmax_test": raw_report,
    }
    (run_dir / "calibration.json").write_text(json.dumps(out, indent=2))
    # features_val.jsonl and features_test.jsonl are written incrementally in
    # extract_features (each row flushed as it's produced) — see resume logic there.

    best = t_report if chosen_name == "T1_threshold" else l_report
    _au = f"{best['auroc']:.3f}" if best['auroc'] is not None else "n/a"
    _raw_au = f"{raw_report['auroc']:.3f}" if raw_report['auroc'] is not None else "n/a"
    logger.info(
        f"[green]Chosen[/green] {chosen_name}: "
        f"test F0.5={best['f0_5']:.3f} P={best['precision_ai']:.3f} R={best['recall_ai']:.3f} "
        f"AUROC={_au} ECE={best['ece']:.3f}"
    )
    logger.info(
        f"[dim]Raw baseline[/dim] test F0.5={raw_report['f0_5']:.3f} "
        f"AUROC={_raw_au} ECE={raw_report['ece']:.3f}"
    )
    _append_runs_log(run_id, config, best)
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(prog="aitd-calibrate")
    p.add_argument("--config", type=pathlib.Path, help="YAML config (required for a new run)")
    p.add_argument("--resume", type=str, default=None, help="Run ID to resume — reuses run_dir + its config.yaml")
    args = p.parse_args()

    if args.resume:
        run_dir = RUNS_DIR / args.resume
        if not run_dir.exists():
            console.print(f"[red]Resume run dir not found:[/red] {run_dir}")
            sys.exit(1)
        cfg = run_dir / "config.yaml"
        if not cfg.exists():
            console.print(f"[red]No config.yaml in {run_dir}[/red]")
            sys.exit(1)
        run_calibration(cfg, resume_run_id=args.resume)
    else:
        if args.config is None:
            console.print("[red]--config is required for a new run (or pass --resume <run_id>)[/red]")
            sys.exit(1)
        if not args.config.exists():
            console.print(f"[red]Config not found:[/red] {args.config}")
            sys.exit(1)
        run_calibration(args.config)


if __name__ == "__main__":
    main()
