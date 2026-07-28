"""Generate the §4 result figures for Inlämning 5.

All figures use a clean, minimal matplotlib style suitable for a bachelor
thesis: a single accent colour per chart, no decorative gridlines beyond
the axis ticks, sans-serif, 300 DPI, sized to fit a single column or a
half-page. Section-prefixed filenames so they slot directly into the
draft's `[TODO: Figure N]` placeholders.

Outputs:
  docs/figures/fig-4-1-roc.png
  docs/figures/fig-4-1-reliability.png
  docs/figures/fig-4-1-per-domain.png
  docs/figures/fig-4-2-induction-trajectory.png
  docs/figures/fig-4-3-margin-histogram.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

REPO = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "docs" / "figures"
CALIB_RUN = REPO / "logs" / "runs" / "2026-04-26T19-07-51_137899"
ABLATION_RUN = REPO / "logs" / "runs" / "2026-04-26T20-23-43_2f80b2"

# A spare, readable palette: one accent per chart + a neutral gray for context.
ACCENT = "#1f4e79"   # KTH-ish navy
ACCENT_2 = "#c46c2c" # warm contrast for the second series
NEUTRAL = "#777777"
LIGHT_BG = "#f6f6f6"


def _style():
    """Apply a clean baseline rcParams configuration to all figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --- Figure 4.1 (a): ROC curve --------------------------------------------------


def fig_roc():
    rows = load_jsonl(CALIB_RUN / "features_test.jsonl")
    valid = [r for r in rows if r.get("margin") is not None]
    y = np.array([r["label"] for r in valid])
    margin = np.array([r["margin"] for r in valid])

    fpr, tpr, thresholds = roc_curve(y, margin)
    auroc = float(np.trapezoid(tpr, fpr))

    # Pick the F0.5-optimal threshold on val (T1 reports 4.736 in calibration.json).
    calib = json.loads((CALIB_RUN / "calibration.json").read_text())
    chosen_threshold = calib["t1_threshold"]["threshold"]
    # Find the (fpr, tpr) closest to the chosen threshold.
    op_idx = int(np.argmin(np.abs(thresholds - chosen_threshold)))
    op_fpr, op_tpr = fpr[op_idx], tpr[op_idx]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(fpr, tpr, color=ACCENT, linewidth=1.8, label=f"E4B + policy (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=0.8, linestyle="--", label="random")
    ax.scatter([op_fpr], [op_tpr], s=70, color=ACCENT_2, zorder=5,
               label=f"F0.5-optimal threshold (margin = {chosen_threshold:.2f})")
    ax.annotate(f"  P = {calib['t1_threshold']['test']['precision_ai']:.3f}\n  R = {calib['t1_threshold']['test']['recall_ai']:.3f}",
                xy=(op_fpr, op_tpr), xytext=(op_fpr + 0.06, op_tpr - 0.10),
                fontsize=8.5, color=ACCENT_2, va="top")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — E4B with induced policy on HC3 test (n = 4 000)")
    ax.legend(loc="lower right", fontsize=8.5)

    out = FIG_DIR / "fig-4-1-roc.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}  AUROC = {auroc:.4f}, op @ ({op_fpr:.3f}, {op_tpr:.3f})")


# --- Figure 4.1 (b): Reliability diagram ---------------------------------------


def fig_reliability():
    calib = json.loads((CALIB_RUN / "calibration.json").read_text())
    bins = calib["t2_logistic"]["test"]["reliability"]
    populated = [b for b in bins if b["n"] > 0 and "conf" in b]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5.5, 4.6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Top: predicted vs observed accuracy with bar widths
    centres = [(b["low"] + b["high"]) / 2 for b in populated]
    confs = [b["conf"] for b in populated]
    accs = [b["acc"] for b in populated]
    counts = [b["n"] for b in populated]

    bar_widths = [b["high"] - b["low"] for b in populated]
    ax_top.bar(centres, accs, width=[w * 0.9 for w in bar_widths],
               color=ACCENT, alpha=0.78, edgecolor="white", linewidth=0.8,
               label="observed accuracy in bin")
    ax_top.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=0.8, linestyle="--",
                label="perfect calibration")
    ax_top.scatter(confs, accs, s=14, color=ACCENT_2, zorder=5, label="bin centroid")
    ax_top.set_ylabel("Empirical accuracy")
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1.04)
    ax_top.legend(loc="upper left", fontsize=8.5)
    ax_top.set_title(f"Reliability — E4B + policy + T2 calibration  (ECE = {calib['t2_logistic']['test']['ece']:.3f})")

    # Bottom: bin counts on a log scale so small bins are visible
    ax_bot.bar(centres, counts, width=[w * 0.9 for w in bar_widths],
               color=ACCENT, alpha=0.78, edgecolor="white", linewidth=0.8)
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("Predicted P(AI)")
    ax_bot.set_ylabel("Bin count")
    ax_bot.set_xlim(0, 1)

    fig.tight_layout()
    out = FIG_DIR / "fig-4-1-reliability.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}  ECE = {calib['t2_logistic']['test']['ece']:.4f}")


# --- Figure 4.1 (c): Per-domain F0.5 (E4B vs 31B) --------------------------------


PER_DOMAIN = [
    # (subset, e4b_f05, 31b_f05) — n=200 each, default-prompt baselines from logs/RUNS.md
    ("finance",     0.890, 0.992),
    ("medicine",    0.952, 0.984),
    ("open_qa",     0.615, 0.727),
    ("reddit_eli5", 0.917, 0.992),
    ("wiki_csai",   0.625, 0.868),
]


def fig_per_domain():
    domains = [d[0] for d in PER_DOMAIN]
    e4b = np.array([d[1] for d in PER_DOMAIN])
    b31 = np.array([d[2] for d in PER_DOMAIN])

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(domains))
    w = 0.36
    ax.bar(x - w/2, e4b, width=w, color=ACCENT, label="Gemma 4 E4B", edgecolor="white", linewidth=0.7)
    ax.bar(x + w/2, b31, width=w, color=ACCENT_2, label="Gemma 4 31B", edgecolor="white", linewidth=0.7)

    for i, (e, b) in enumerate(zip(e4b, b31)):
        ax.text(i - w/2, e + 0.012, f"{e:.3f}", ha="center", fontsize=7.5, color=ACCENT)
        ax.text(i + w/2, b + 0.012, f"{b:.3f}", ha="center", fontsize=7.5, color=ACCENT_2)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=0, fontsize=9)
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("F0.5")
    ax.set_title("Per-domain F0.5 — default-prompt baselines (n = 200 each)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0)
    out = FIG_DIR / "fig-4-1-per-domain.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# --- Figure 4.2: Induction trajectory ------------------------------------------


TRAJECTORY = [
    # (iter, F0.5, decision)
    (0, 0.936, "initial"),
    (1, 0.956, "accepted"),
    (2, 0.941, "rejected"),
    (3, 0.941, "rejected"),
    (4, 0.941, "rejected"),
    (5, 0.941, "rejected"),
    (6, 0.941, "rejected"),
]


def fig_trajectory():
    iters = np.array([t[0] for t in TRAJECTORY])
    f05 = np.array([t[1] for t in TRAJECTORY])
    decisions = [t[2] for t in TRAJECTORY]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    # Connecting line for the best-so-far cumulative trace
    best_so_far = np.maximum.accumulate(np.where(np.array(decisions) != "rejected", f05, 0))
    best_so_far[best_so_far == 0] = np.nan
    # Carry forward the best across rejections so the line is monotone non-decreasing
    cur = -np.inf
    line = []
    for v in best_so_far:
        if not np.isnan(v):
            cur = v
        line.append(cur)
    ax.plot(iters, line, color=NEUTRAL, linewidth=0.9, linestyle="--", label="best so far")

    # Per-iter points (use first-occurrence legend tracking)
    seen = set()

    def _label(key: str) -> str:
        if key in seen:
            return "_nolegend_"
        seen.add(key)
        return key

    for it, val, dec in zip(iters, f05, decisions):
        if dec == "accepted":
            ax.scatter([it], [val], s=110, color=ACCENT_2, zorder=5,
                       label=_label("accepted (winner)"))
            ax.annotate(f"  iter {it} winner\n  F0.5 = {val:.3f}",
                        xy=(it, val), xytext=(it + 0.25, val + 0.005),
                        fontsize=9, color=ACCENT_2, va="bottom")
        elif dec == "initial":
            ax.scatter([it], [val], s=70, color=ACCENT, zorder=5,
                       label=_label("initial"))
        else:
            ax.scatter([it], [val], s=50, color=NEUTRAL, marker="x", zorder=5,
                       label=_label("rejected (refiner deadlock)"))

    ax.set_xticks(iters)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("F0.5  (n = 500 val)")
    ax.set_ylim(0.92, 0.972)
    ax.set_title("Policy induction trajectory  (run 2026-04-26T17-42-47_3d67db)")

    # Early-stop annotation, placed above the legend zone
    ax.axvspan(1.5, 6.5, alpha=0.08, color=NEUTRAL)
    ax.text(4, 0.965, "early-stop region\n(5 consecutive rejections)",
            ha="center", fontsize=8.5, color=NEUTRAL, style="italic")
    # Legend at lower-left to keep clear of the early-stop band on the right
    ax.legend(loc="lower left", fontsize=8.5)

    out = FIG_DIR / "fig-4-2-induction-trajectory.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# --- Figure 4.3: Faithfulness margin histogram ---------------------------------


def fig_margin_histogram():
    rows = load_jsonl(ABLATION_RUN / "faithfulness.jsonl")
    best = np.array([r["diffs"]["best"] for r in rows if r["diffs"]["best"] is not None])
    inverted = np.array([r["diffs"]["inverted"] for r in rows if r["diffs"]["inverted"] is not None])

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    # Common bin range
    lo = float(min(best.min(), inverted.min()))
    hi = float(max(best.max(), inverted.max()))
    bins = np.linspace(lo, hi, 24)

    ax.hist(inverted, bins=bins, color=NEUTRAL, alpha=0.55, edgecolor="white", linewidth=0.7,
            label=f"inverted policy (mean = {inverted.mean():+.2f} nats)")
    ax.hist(best, bins=bins, color=ACCENT, alpha=0.78, edgecolor="white", linewidth=0.7,
            label=f"best policy (mean = {best.mean():+.2f} nats)")

    ax.axvline(best.mean(), color=ACCENT, linewidth=1.0, linestyle="--")
    ax.axvline(inverted.mean(), color=NEUTRAL, linewidth=1.0, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.6)

    shift = best.mean() - inverted.mean()
    # Place annotation in upper-right so it doesn't fight the upper-left legend
    y_top = ax.get_ylim()[1]
    ax.annotate(f"mean shift = {shift:+.2f} nats",
                xy=(best.mean() + 1.5, y_top * 0.7),
                fontsize=9, ha="left", color=ACCENT_2,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT_2, lw=0.8))

    ax.set_xlabel("logp(yes) − logp(no)  (nats)")
    ax.set_ylabel("Count  (n = 100 test)")
    ax.set_title("Faithfulness ablation — per-example log-probability margin")
    ax.legend(loc="upper left", fontsize=8.5)

    out = FIG_DIR / "fig-4-3-margin-histogram.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}  shift = {shift:+.3f} nats")


# --- Figure 3.2: Pipeline architecture --------------------------------------------


def fig_pipeline():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, ec=ACCENT, fontsize=9.5):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                           fc=fc, ec=ec, linewidth=1.3)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=ACCENT)

    def arrow_v(x, y_top, y_bot):
        a = FancyArrowPatch((x, y_top), (x, y_bot),
                            arrowstyle="->,head_length=8,head_width=5",
                            linewidth=1.2, color=NEUTRAL)
        ax.add_patch(a)

    def arrow_h(x_left, x_right, y, label=None, color=NEUTRAL):
        a = FancyArrowPatch((x_left, y), (x_right, y),
                            arrowstyle="->,head_length=9,head_width=6",
                            linewidth=1.4, color=color)
        ax.add_patch(a)
        if label:
            ax.text((x_left + x_right) / 2, y + 0.35, label,
                    ha="center", fontsize=9, color=color, style="italic")

    # Phase 1 — left column
    P1_X, P1_W = 0.5, 6.0
    ax.text(P1_X + P1_W / 2, 10.4, "Phase 1 — Policy induction",
            fontsize=11.5, weight="bold", ha="center", color=ACCENT)
    box(P1_X, 8.50, P1_W, 1.20, "Seed pool\n(10–20 labelled HC3 examples)", LIGHT_BG)
    box(P1_X, 6.50, P1_W, 1.20, "Proposer LLM\n(writes ~150-word policy)", "#e8eef5")
    box(P1_X, 4.50, P1_W, 1.20, "Scorer\n(F0.5 on n = 500 val subset)", LIGHT_BG)
    box(P1_X, 2.50, P1_W, 1.20, "Accept iff F0.5 improves\n(plateau Δ < 0.005 × 3, max 30 iters)",
        "#e8eef5")
    cx_p1 = P1_X + P1_W / 2
    arrow_v(cx_p1, 8.50, 7.75)
    arrow_v(cx_p1, 6.50, 5.75)
    arrow_v(cx_p1, 4.50, 3.75)

    # Phase 1 feedback loop on the far left
    fb_x = P1_X - 0.45
    ax.plot([P1_X, fb_x], [2.95, 2.95], color=NEUTRAL, linewidth=1.0)
    ax.plot([fb_x, fb_x], [2.95, 7.10], color=NEUTRAL, linewidth=1.0)
    ax.add_patch(FancyArrowPatch((fb_x, 7.10), (P1_X, 7.10),
                 arrowstyle="->,head_length=8,head_width=5",
                 linewidth=1.0, color=NEUTRAL))
    ax.text(fb_x - 0.20, 5.0, "misclassified\nexamples", ha="right", va="center",
            fontsize=8.5, color=NEUTRAL, style="italic", rotation=90)

    # Frozen-policy bridge — centre
    BR_X, BR_W = 8.5, 5.0
    box(BR_X, 5.10, BR_W, 1.80,
        "Frozen policy\n(~150 words of natural language)",
        "#fff4e8", ec=ACCENT_2, fontsize=10)
    ax.text(BR_X + BR_W / 2, 7.30, "the policy is the artefact",
            ha="center", fontsize=9.5, color=ACCENT_2, style="italic", weight="bold")
    ax.text(BR_X + BR_W / 2, 4.55, "(re-used as the system prompt of the deployed classifier)",
            ha="center", fontsize=8.5, color=ACCENT_2, style="italic")

    arrow_h(P1_X + P1_W + 0.15, BR_X - 0.15, 6.00, label="freeze", color=ACCENT_2)

    # Phase 2 — right column
    P2_X, P2_W = 15.5, 6.0
    ax.text(P2_X + P2_W / 2, 10.4, "Phase 2 — Calibration",
            fontsize=11.5, weight="bold", ha="center", color=ACCENT)
    box(P2_X, 8.50, P2_W, 1.20, "Classifier\n(E4B with frozen policy as system prompt)",
        LIGHT_BG)
    box(P2_X, 6.50, P2_W, 1.20, "log-probabilities\n{ logp(yes), logp(no), logp(other) }",
        "#e8eef5")
    box(P2_X, 4.50, P2_W, 1.20, "Calibrator\n(T1 threshold or T2 logistic, fit on val)",
        LIGHT_BG)
    box(P2_X, 2.50, P2_W, 1.20, "Test split  →  F0.5 / AUROC / ECE",
        "#fff4e8", ec=ACCENT_2)
    cx_p2 = P2_X + P2_W / 2
    arrow_v(cx_p2, 8.50, 7.75)
    arrow_v(cx_p2, 6.50, 5.75)
    arrow_v(cx_p2, 4.50, 3.75)

    arrow_h(BR_X + BR_W + 0.15, P2_X - 0.15, 6.00, label="system prompt", color=ACCENT_2)

    out = FIG_DIR / "fig-3-2-pipeline.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# --- Figure 4.1: Confusion matrix on n=4000 test ----------------------------------


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _apply_t2(rows: list[dict], calib: dict) -> np.ndarray:
    coef = np.array(calib["t2_logistic"]["coef"])
    intercept = float(calib["t2_logistic"]["intercept"])
    threshold = float(calib["t2_logistic"]["decision_threshold"])
    LARGE, FLOOR = 20.0, -20.0

    def feat(r):
        lp_yes = r["lp_yes"] if r["lp_yes"] is not None else FLOOR
        lp_no = r["lp_no"] if r["lp_no"] is not None else FLOOR
        m = r["margin"]
        if m is None:
            if r["lp_yes"] is not None and r["lp_no"] is None:
                m = LARGE
            elif r["lp_yes"] is None and r["lp_no"] is not None:
                m = -LARGE
            else:
                m = 0.0
        return np.array([lp_yes, lp_no, m])

    X = np.stack([feat(r) for r in rows])
    z = X @ coef + intercept
    p = _sigmoid(z)
    return (p >= threshold).astype(int)


def fig_confusion_matrix():
    rows = load_jsonl(CALIB_RUN / "features_test.jsonl")
    calib = json.loads((CALIB_RUN / "calibration.json").read_text())
    y_true = np.array([r["label"] for r in rows])
    y_pred = _apply_t2(rows, calib)

    # 2x2 confusion matrix
    cm = np.array([
        [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
        [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
    ])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=cm.max() * 1.05)

    labels = ["human (0)", "AI (1)"]
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / total * 100
            txt_color = "white" if count > cm.max() * 0.55 else ACCENT
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12, color=txt_color, weight="bold")

    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix — E4B + policy + T2 calibration  (n = 4 000)")
    ax.grid(False)

    # Annotate aggregate metrics under the matrix
    P = calib["t2_logistic"]["test"]["precision_ai"]
    R = calib["t2_logistic"]["test"]["recall_ai"]
    F = calib["t2_logistic"]["test"]["f0_5"]
    A = calib["t2_logistic"]["test"]["accuracy"]
    fig.text(0.5, -0.02,
             f"precision = {P:.3f}    recall = {R:.3f}    F0.5 = {F:.3f}    accuracy = {A:.3f}",
             ha="center", fontsize=9.5, color=ACCENT)

    out = FIG_DIR / "fig-4-1-confusion-matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}  cm = {cm.tolist()}")


# --- Figure 4.1: Calibration before/after -----------------------------------------


def fig_calibration_effect():
    calib = json.loads((CALIB_RUN / "calibration.json").read_text())
    raw = calib["raw_argmax_test"]
    t2 = calib["t2_logistic"]["test"]

    metrics = ["F0.5", "precision", "recall", "ECE"]
    raw_vals = [raw["f0_5"], raw["precision_ai"], raw["recall_ai"], raw["ece"]]
    t2_vals = [t2["f0_5"], t2["precision_ai"], t2["recall_ai"], t2["ece"]]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w/2, raw_vals, width=w, color=NEUTRAL, label="raw argmax", edgecolor="white", linewidth=0.7)
    ax.bar(x + w/2, t2_vals, width=w, color=ACCENT, label="T2 logistic calibration", edgecolor="white", linewidth=0.7)

    for i, (rv, tv) in enumerate(zip(raw_vals, t2_vals)):
        ax.text(i - w/2, rv + 0.012, f"{rv:.3f}", ha="center", fontsize=8, color=NEUTRAL)
        ax.text(i + w/2, tv + 0.012, f"{tv:.3f}", ha="center", fontsize=8, color=ACCENT)
        delta = tv - rv
        sign = "+" if delta > 0 else ""
        ax.text(i, max(rv, tv) + 0.06, f"Δ {sign}{delta:.3f}", ha="center",
                fontsize=8, color=ACCENT_2, style="italic")

    ax.set_xticks(x, metrics, fontsize=9.5)
    ax.set_ylim(0, max(max(raw_vals), max(t2_vals)) * 1.18)
    ax.set_ylabel("value")
    ax.set_title("Calibration effect — raw argmax vs T2 logistic on n = 4 000 test")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", alpha=0)
    out = FIG_DIR / "fig-4-1-calibration-effect.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# --- Figure 4.3: Three-policy margin histogram (best / empty / inverted) ---------


def fig_three_policy_histogram():
    rows = load_jsonl(ABLATION_RUN / "faithfulness.jsonl")
    best = np.array([r["diffs"]["best"] for r in rows if r["diffs"]["best"] is not None])
    empty = np.array([r["diffs"]["empty"] for r in rows if r["diffs"]["empty"] is not None])
    inverted = np.array([r["diffs"]["inverted"] for r in rows if r["diffs"]["inverted"] is not None])

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.2), sharex=True)

    bins = np.linspace(min(best.min(), empty.min(), inverted.min()),
                       max(best.max(), empty.max(), inverted.max()), 24)

    for ax, data, name, color in [
        (axes[0], best,     f"best policy (mean = {best.mean():+.2f} nats)",     ACCENT),
        (axes[1], empty,    f"empty system prompt (mean = {empty.mean():+.2f} nats)", ACCENT_2),
        (axes[2], inverted, f"inverted policy (mean = {inverted.mean():+.2f} nats)",  NEUTRAL),
    ]:
        ax.hist(data, bins=bins, color=color, alpha=0.78, edgecolor="white", linewidth=0.7)
        ax.axvline(data.mean(), color=color, linewidth=1.0, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.6)
        ax.set_title(name, fontsize=9.5, color=color, loc="left", weight="normal")
        ax.set_ylabel("count")
        ax.set_ylim(0, 25)

    axes[-1].set_xlabel("logp(yes) − logp(no)  (nats)")
    fig.suptitle("Faithfulness — per-example margin under three policies (n = 100)",
                 fontsize=11, weight="bold", y=0.995)
    fig.tight_layout()

    out = FIG_DIR / "fig-4-3-three-policies.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}  best/empty/inverted means: "
          f"{best.mean():+.2f}/{empty.mean():+.2f}/{inverted.mean():+.2f}")


# --- Figure 4.3: Pairwise label-flip + margin-shift ------------------------------


def fig_pairwise_summary():
    report = json.loads((ABLATION_RUN / "faithfulness.json").read_text())
    pairs = ["best_vs_empty", "best_vs_inverted", "empty_vs_inverted"]
    label_rates = [report["pairwise"][p]["label_change_rate"] for p in pairs]
    margin_shifts = [report["pairwise"][p]["mean_delta_logprob"] for p in pairs]

    fig, ax_left = plt.subplots(figsize=(6.0, 3.8))
    ax_right = ax_left.twinx()

    x = np.arange(len(pairs))
    w = 0.36
    bars1 = ax_left.bar(x - w/2, label_rates, width=w, color=ACCENT,
                        label="Δlabel rate", edgecolor="white", linewidth=0.7)
    bars2 = ax_right.bar(x + w/2, margin_shifts, width=w, color=ACCENT_2,
                         label="mean Δ(lp(yes) − lp(no))", edgecolor="white", linewidth=0.7)

    for b, v in zip(bars1, label_rates):
        ax_left.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                     ha="center", fontsize=8, color=ACCENT)
    for b, v in zip(bars2, margin_shifts):
        ax_right.text(b.get_x() + b.get_width() / 2,
                      v + (0.4 if v >= 0 else -0.6), f"{v:+.2f}",
                      ha="center", fontsize=8, color=ACCENT_2)

    ax_left.set_xticks(x, [p.replace("_", " ") for p in pairs], fontsize=9)
    ax_left.set_ylabel("Δlabel rate", color=ACCENT)
    ax_right.set_ylabel("mean Δ margin (nats)", color=ACCENT_2)
    ax_left.set_ylim(0, max(label_rates) * 1.25)
    ax_right.set_ylim(min(margin_shifts) - 1, max(margin_shifts) * 1.18)
    ax_left.set_title("Pairwise faithfulness — label flips + margin shifts (n = 100)")
    ax_left.tick_params(axis="y", colors=ACCENT)
    ax_right.tick_params(axis="y", colors=ACCENT_2)
    ax_left.grid(axis="x", alpha=0)
    ax_right.grid(False)

    # Combined legend
    lines = [bars1, bars2]
    labels = ["Δlabel rate", "mean Δ margin (nats)"]
    ax_left.legend(lines, labels, loc="upper left", fontsize=8.5)

    out = FIG_DIR / "fig-4-3-pairwise.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# --- Figure 5.5: Meta-experiment evolution heatmap -------------------------------


META = REPO / "logs" / "meta-experiment"


def fig_meta_heatmap():
    runs = [
        ("baseline-v2.json",          "v2 baseline"),
        ("iter-1.json",               "v2 iter-1"),
        ("iter-2.json",               "v2 iter-2"),
        ("iter-3.json",               "v2 iter-3"),
        ("iter-4-31b.json",           "v2 iter-4 (31B)"),
        ("v3-baseline-with-reasoning.json", "v3 baseline"),
        ("v3-iter-1-with-reasoning.json",   "v3 iter-1"),
    ]
    data = []
    for fname, label in runs:
        d = json.loads((META / fname).read_text())
        data.append((label, {r["heading"]: r.get("p_ai") for r in d["results"]}))

    # Use the union of headings across all runs, ordered by their order in the latest run
    latest_order = [r["heading"] for r in json.loads((META / runs[-1][0]).read_text())["results"]]
    all_headings = []
    seen = set()
    for h in latest_order:
        if h not in seen:
            all_headings.append(h)
            seen.add(h)
    for _, hd in data:
        for h in hd.keys():
            if h not in seen:
                all_headings.append(h)
                seen.add(h)

    matrix = np.full((len(all_headings), len(runs)), np.nan)
    for j, (_, hd) in enumerate(data):
        for i, h in enumerate(all_headings):
            v = hd.get(h)
            if v is not None:
                matrix[i, j] = float(v)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(runs)))
    ax.set_xticklabels([d[0] for d in data], rotation=22, ha="right", fontsize=8.5)
    short_labels = [h.split("(")[0].rstrip(" *").strip() for h in all_headings]
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)

    # Annotate each cell with P(AI) and a faint AI-flag star
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#888")
                continue
            text_color = "white" if (v < 0.25 or v > 0.7) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=text_color)

    ax.set_title("Meta-experiment — per-section P(AI) across iterations")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("P(AI)  (E4B + induced policy)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    out = FIG_DIR / "fig-5-5-meta-experiment-heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}  shape = {matrix.shape}")


# --- Figure 5.5: Reasoning-vs-score iteration counts -----------------------------


def fig_reasoning_vs_score():
    # AI-flag count per iteration, two arcs
    v2 = [("baseline", 9), ("iter-1", 4), ("iter-2", 1), ("iter-3", 0)]
    v3 = [("baseline", 6), ("iter-1", 0)]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot([s[0] for s in v2], [s[1] for s in v2], color=ACCENT, marker="o",
            linewidth=1.6, markersize=8, label="score-only (v2)")
    ax.plot([s[0] for s in v3], [s[1] for s in v3], color=ACCENT_2, marker="s",
            linewidth=1.6, markersize=8, label="reasoning-guided (v3)")

    for x, y in v2:
        ax.text(x, y + 0.3, str(y), ha="center", fontsize=8.5, color=ACCENT)
    for x, y in v3:
        ax.text(x, y + 0.3, str(y), ha="center", fontsize=8.5, color=ACCENT_2)

    ax.axhline(0, color=NEUTRAL, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sections flagged AI (out of 17)")
    ax.set_ylim(-0.7, 10.5)
    ax.set_xlim(-0.3, 3.3)
    ax.set_title("Score-only vs reasoning-guided rewriting — convergence to 0/17 AI flags")
    ax.legend(loc="upper right", fontsize=9.5)

    out = FIG_DIR / "fig-5-5-reasoning-vs-score.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# =================================================================================
# v3 series — figures that visualise the inl5-draft-v3.md prose itself, not the
# underlying detection experiments. Saved to docs/figures/v3/. Together they show
# how the policy classifier sees the final v3 draft section-by-section, and the
# baseline-vs-iter-1 effect of reasoning-guided rewriting.
# =================================================================================


V3_DIR = FIG_DIR / "v3"


def _short_heading(h: str) -> str:
    return h.split("(")[0].rstrip(" *").strip()


def _load_v3(name: str) -> list[dict]:
    p = META / f"v3-{name}-with-reasoning.json"
    return json.loads(p.read_text())["results"]


def v3_fig_section_pai():
    """Per-section P(AI) at v3-iter-1, sorted by section order."""
    rows = _load_v3("iter-1")
    headings = [_short_heading(r["heading"]) for r in rows]
    p_ai = np.array([r.get("p_ai") if r.get("p_ai") is not None else 0.0 for r in rows])

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    y = np.arange(len(headings))
    colors = [ACCENT_2 if v >= 0.5 else ACCENT for v in p_ai]
    ax.barh(y, p_ai, color=colors, edgecolor="white", linewidth=0.7)
    ax.axvline(0.5, color=NEUTRAL, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(0.51, len(headings) - 0.4, "AI threshold", color=NEUTRAL, fontsize=8.5,
            style="italic", va="top")
    for i, v in enumerate(p_ai):
        ax.text(min(v + 0.02, 0.96), i, f"{v:.3f}", va="center", fontsize=8,
                color=ACCENT_2 if v >= 0.5 else ACCENT)

    ax.set_yticks(y, headings, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("P(AI)  —  E4B + induced policy")
    ax.set_title("v3 final draft — per-section P(AI)  (0/17 sections flagged)")
    ax.grid(axis="y", alpha=0)

    out = V3_DIR / "v3-fig-1-section-pai.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


def v3_fig_section_margin():
    """Per-section log-probability margin at v3-iter-1.  Negative = human, positive = AI."""
    rows = _load_v3("iter-1")
    headings = [_short_heading(r["heading"]) for r in rows]
    margins = np.array([r.get("margin_nats") if r.get("margin_nats") is not None else np.nan
                        for r in rows])

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    y = np.arange(len(headings))
    # Color by sign: cool for negative (human), warm for positive (AI)
    colors = [ACCENT if (not np.isnan(v) and v < 0) else ACCENT_2 if (not np.isnan(v) and v > 0) else NEUTRAL
              for v in margins]
    ax.barh(y, margins, color=colors, edgecolor="white", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.6)
    for i, v in enumerate(margins):
        if np.isnan(v):
            continue
        offset = -0.6 if v < 0 else 0.3
        ha = "right" if v < 0 else "left"
        ax.text(v + offset, i, f"{v:+.2f}", va="center", ha=ha, fontsize=8,
                color=ACCENT if v < 0 else ACCENT_2)

    ax.set_yticks(y, headings, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(min(margins[~np.isnan(margins)]) - 2.0, max(margins[~np.isnan(margins)]) + 2.5)
    ax.set_xlabel("logp(yes) − logp(no)  (nats; negative = human)")
    ax.set_title("v3 final draft — per-section log-probability margin")
    ax.grid(axis="y", alpha=0)

    out = V3_DIR / "v3-fig-2-section-margin.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


def v3_fig_baseline_vs_iter1():
    """Slope chart — P(AI) at v3-baseline vs v3-iter-1 per section."""
    base = {_short_heading(r["heading"]): r.get("p_ai") for r in _load_v3("baseline")}
    iter1 = {_short_heading(r["heading"]): r.get("p_ai") for r in _load_v3("iter-1")}

    sections = [h for h in base.keys() if h in iter1
                and base[h] is not None and iter1[h] is not None]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    x_base, x_iter1 = 0, 1
    for h in sections:
        b = base[h]
        i = iter1[h]
        was_flagged = b >= 0.5
        flipped = was_flagged and i < 0.5
        color = ACCENT_2 if flipped else NEUTRAL if was_flagged else ACCENT
        alpha = 1.0 if flipped else 0.45 if was_flagged else 0.35
        lw = 2.0 if flipped else 1.0
        ax.plot([x_base, x_iter1], [b, i], color=color, alpha=alpha, linewidth=lw,
                marker="o", markersize=5)
        if flipped:
            ax.text(x_base - 0.04, b, h, ha="right", va="center", fontsize=8.5,
                    color=ACCENT_2)

    ax.axhline(0.5, color=NEUTRAL, linewidth=0.7, linestyle="--", alpha=0.7)
    ax.text(1.02, 0.51, "AI threshold", color=NEUTRAL, fontsize=8.5, style="italic")
    ax.set_xticks([x_base, x_iter1], ["v3 baseline", "v3 iter-1"], fontsize=10)
    ax.set_ylabel("P(AI)")
    ax.set_xlim(-0.55, 1.30)
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Reasoning-guided rewriting — per-section P(AI), v3 baseline → iter-1")
    ax.grid(axis="x", alpha=0)

    # Footer summary
    n_flipped = sum(1 for h in sections if base[h] >= 0.5 and iter1[h] < 0.5)
    n_flagged_baseline = sum(1 for h in sections if base[h] >= 0.5)
    fig.text(0.5, -0.01,
             f"{n_flipped}/{n_flagged_baseline} flagged sections flipped to human in one iteration",
             ha="center", fontsize=9.5, color=ACCENT_2, style="italic")

    out = V3_DIR / "v3-fig-3-baseline-vs-iter1.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}  flipped {n_flipped}/{n_flagged_baseline}")


def v3_fig_pai_distribution():
    """Overlapping histograms of section P(AI) at baseline vs iter-1."""
    base = np.array([r.get("p_ai") for r in _load_v3("baseline") if r.get("p_ai") is not None])
    iter1 = np.array([r.get("p_ai") for r in _load_v3("iter-1") if r.get("p_ai") is not None])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(base, bins=bins, color=NEUTRAL, alpha=0.65, edgecolor="white", linewidth=0.7,
            label=f"v3 baseline (mean = {base.mean():.3f})")
    ax.hist(iter1, bins=bins, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.7,
            label=f"v3 iter-1 (mean = {iter1.mean():.3f})")
    ax.axvline(0.5, color=ACCENT_2, linewidth=0.8, linestyle="--", alpha=0.8)
    ax.text(0.51, ax.get_ylim()[1] * 0.92, "AI threshold", color=ACCENT_2, fontsize=8.5,
            style="italic")

    ax.set_xlabel("P(AI)")
    ax.set_ylabel("number of sections")
    ax.set_title("Per-section P(AI) distribution — v3 baseline vs iter-1  (n = 17 sections)")
    ax.legend(loc="upper right", fontsize=9, bbox_to_anchor=(0.98, 0.85))
    ax.set_xlim(0, 1)

    out = V3_DIR / "v3-fig-4-pai-distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


def v3_fig_word_count_vs_pai():
    """Scatter — section word count vs P(AI) at v3-iter-1, with baseline overlaid."""
    base = _load_v3("baseline")
    iter1 = _load_v3("iter-1")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, rows, color, marker in [
        ("v3 baseline", base, NEUTRAL, "o"),
        ("v3 iter-1",   iter1, ACCENT,  "s"),
    ]:
        words = np.array([r["words"] for r in rows])
        p_ai = np.array([r.get("p_ai") if r.get("p_ai") is not None else 0.0 for r in rows])
        ax.scatter(words, p_ai, color=color, marker=marker, s=55, alpha=0.75,
                   edgecolor="white", linewidth=0.7, label=label)

    ax.axhline(0.5, color=ACCENT_2, linewidth=0.7, linestyle="--", alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.97, 0.52, "AI threshold", color=ACCENT_2, fontsize=8.5,
            style="italic", ha="right")

    ax.set_xlabel("section length  (words)")
    ax.set_ylabel("P(AI)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Section length vs classifier verdict — v3 baseline and iter-1")
    ax.legend(loc="upper right", fontsize=9)

    out = V3_DIR / "v3-fig-5-length-vs-pai.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# =================================================================================
# Aliases that match the v3 draft's "Figure 1 / Figure 2 / Figure 3" references.
# These are the drop-in canonical filenames Gustav uses when assembling the report.
# =================================================================================


def figure_1_combined():
    """Figure 1 (per v3 §4.1) — ROC curve + reliability diagram in one image,
    stacked vertically so both panels read at the same width."""
    rows = load_jsonl(CALIB_RUN / "features_test.jsonl")
    valid = [r for r in rows if r.get("margin") is not None]
    y = np.array([r["label"] for r in valid])
    margin = np.array([r["margin"] for r in valid])

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, margin)
    auroc = float(np.trapezoid(tpr, fpr))

    calib = json.loads((CALIB_RUN / "calibration.json").read_text())
    chosen_threshold = calib["t1_threshold"]["threshold"]
    op_idx = int(np.argmin(np.abs(thresholds - chosen_threshold)))
    op_fpr, op_tpr = fpr[op_idx], tpr[op_idx]

    bins = calib["t2_logistic"]["test"]["reliability"]
    populated = [b for b in bins if b["n"] > 0 and "conf" in b]

    fig = plt.figure(figsize=(7.0, 8.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 2.4, 1.0], hspace=0.42)
    ax_roc = fig.add_subplot(gs[0])
    ax_rel = fig.add_subplot(gs[1])
    ax_cnt = fig.add_subplot(gs[2], sharex=ax_rel)

    # Top — ROC
    ax_roc.plot(fpr, tpr, color=ACCENT, linewidth=1.8, label=f"E4B + policy (AUROC = {auroc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=0.8, linestyle="--", label="random")
    ax_roc.scatter([op_fpr], [op_tpr], s=70, color=ACCENT_2, zorder=5,
                   label=f"F0.5-optimal threshold (margin = {chosen_threshold:.2f})")
    ax_roc.annotate(f"  P = {calib['t1_threshold']['test']['precision_ai']:.3f}\n"
                    f"  R = {calib['t1_threshold']['test']['recall_ai']:.3f}",
                    xy=(op_fpr, op_tpr), xytext=(op_fpr + 0.06, op_tpr - 0.10),
                    fontsize=8.5, color=ACCENT_2, va="top")
    ax_roc.set_xlim(-0.01, 1.01)
    ax_roc.set_ylim(-0.01, 1.01)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("(a) ROC — E4B with induced policy on HC3 test (n = 4 000)",
                     loc="left", fontsize=10.5)
    ax_roc.legend(loc="lower right", fontsize=8.5)

    # Middle — Reliability diagram (top panel only — accuracy vs confidence)
    centres = [(b["low"] + b["high"]) / 2 for b in populated]
    confs = [b["conf"] for b in populated]
    accs = [b["acc"] for b in populated]
    bar_widths = [b["high"] - b["low"] for b in populated]
    ax_rel.bar(centres, accs, width=[w * 0.9 for w in bar_widths],
               color=ACCENT, alpha=0.78, edgecolor="white", linewidth=0.8,
               label="observed accuracy in bin")
    ax_rel.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=0.8, linestyle="--",
                label="perfect calibration")
    ax_rel.scatter(confs, accs, s=14, color=ACCENT_2, zorder=5, label="bin centroid")
    ax_rel.set_ylabel("Empirical accuracy")
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1.04)
    ax_rel.legend(loc="upper left", fontsize=8.5)
    ax_rel.set_title(f"(b) Reliability — E4B + policy + T2 calibration  (ECE = "
                     f"{calib['t2_logistic']['test']['ece']:.3f})",
                     loc="left", fontsize=10.5)
    plt.setp(ax_rel.get_xticklabels(), visible=False)

    # Bottom — bin counts
    counts = [b["n"] for b in populated]
    ax_cnt.bar(centres, counts, width=[w * 0.9 for w in bar_widths],
               color=ACCENT, alpha=0.78, edgecolor="white", linewidth=0.8)
    ax_cnt.set_yscale("log")
    ax_cnt.set_xlabel("Predicted P(AI)")
    ax_cnt.set_ylabel("Bin count")
    ax_cnt.set_xlim(0, 1)

    out = FIG_DIR / "figure-1.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}  AUROC = {auroc:.4f}, ECE = {calib['t2_logistic']['test']['ece']:.4f}")


def figure_2_alias():
    """Figure 2 (per v3 §4.2) — trajectory plot. Same content as fig-4-2-induction-trajectory.png."""
    import shutil
    src = FIG_DIR / "fig-4-2-induction-trajectory.png"
    dst = FIG_DIR / "figure-2.png"
    shutil.copyfile(src, dst)
    print(f"  ✓ {dst.name}  (alias of {src.name})")


def figure_3_alias():
    """Figure 3 (per v3 §4.3) — margin histogram. Same content as fig-4-3-margin-histogram.png."""
    import shutil
    src = FIG_DIR / "fig-4-3-margin-histogram.png"
    dst = FIG_DIR / "figure-3.png"
    shutil.copyfile(src, dst)
    print(f"  ✓ {dst.name}  (alias of {src.name})")


def main():
    _style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    V3_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[figures] writing to {FIG_DIR.relative_to(REPO)}")
    fig_pipeline()
    fig_roc()
    fig_reliability()
    fig_per_domain()
    fig_confusion_matrix()
    fig_calibration_effect()
    fig_trajectory()
    fig_margin_histogram()
    fig_three_policy_histogram()
    fig_pairwise_summary()
    fig_meta_heatmap()
    fig_reasoning_vs_score()
    print(f"[figures] v3-aligned canonicals (drop-in for the draft)")
    figure_1_combined()
    figure_2_alias()
    figure_3_alias()
    print(f"[figures/v3] writing to {V3_DIR.relative_to(REPO)}")
    v3_fig_section_pai()
    v3_fig_section_margin()
    v3_fig_baseline_vs_iter1()
    v3_fig_pai_distribution()
    v3_fig_word_count_vs_pai()


if __name__ == "__main__":
    main()
