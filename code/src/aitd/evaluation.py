from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class Metrics:
    n: int
    n_valid: int
    accuracy: float
    f1: float
    precision_ai: float
    recall_ai: float
    auroc: float | None = None
    ece: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(
    labels: Sequence[int],
    preds: Sequence[int],
    probs: Sequence[float] | None = None,
) -> Metrics:
    labels_arr = np.asarray(labels)
    preds_arr = np.asarray(preds)
    mask = preds_arr != -1
    n = len(labels_arr)
    n_valid = int(mask.sum())

    if n_valid == 0:
        return Metrics(n=n, n_valid=0, accuracy=0.0, f1=0.0, precision_ai=0.0, recall_ai=0.0)

    y, p = labels_arr[mask], preds_arr[mask]
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    auroc = None
    if probs is not None:
        probs_arr = np.asarray(probs)[mask]
        if len(set(y.tolist())) == 2:
            auroc = float(roc_auc_score(y, probs_arr))

    ece = _expected_calibration_error(y, np.asarray(probs)[mask]) if probs is not None else None

    return Metrics(
        n=n,
        n_valid=n_valid,
        accuracy=float((p == y).mean()),
        f1=float(f1_score(y, p, zero_division=0)),
        precision_ai=float(precision),
        recall_ai=float(recall),
        auroc=auroc,
        ece=ece,
    )


def _expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        in_bin = idx == b
        if not in_bin.any():
            continue
        acc = (y[in_bin] == 1).mean()
        conf = probs[in_bin].mean()
        ece += (in_bin.mean()) * abs(acc - conf)
    return float(ece)
