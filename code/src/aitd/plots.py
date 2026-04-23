from __future__ import annotations

import json
import pathlib


def plot_trajectory(trajectory_jsonl: pathlib.Path, output_path: pathlib.Path) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters: list[int] = []
    f0_5s: list[float] = []
    accepted: list[bool] = []
    with trajectory_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iters.append(row["iter"])
            f0_5s.append(row["f0_5"])
            accepted.append(row["accepted"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, f0_5s, color="#444", linewidth=1.0, zorder=1)
    acc_x = [i for i, a in zip(iters, accepted) if a]
    acc_y = [v for v, a in zip(f0_5s, accepted) if a]
    rej_x = [i for i, a in zip(iters, accepted) if not a]
    rej_y = [v for v, a in zip(f0_5s, accepted) if not a]
    if acc_x:
        ax.scatter(acc_x, acc_y, marker="o", color="#2a9d8f", s=60, label="accepted", zorder=3)
    if rej_x:
        ax.scatter(rej_x, rej_y, marker="x", color="#e76f51", s=60, linewidths=2.0, label="rejected", zorder=3)

    best_f05 = max(f0_5s) if f0_5s else 0.0
    best_iter = iters[f0_5s.index(best_f05)] if f0_5s else 0
    ax.axhline(best_f05, linestyle="--", color="#888", linewidth=0.8, zorder=0)
    ax.annotate(f"best={best_f05:.3f} @ iter={best_iter}", xy=(best_iter, best_f05),
                xytext=(5, -12), textcoords="offset points", fontsize=9, color="#444")

    ax.set_xlabel("iteration")
    ax.set_ylabel("F0.5 (val)")
    ax.set_title(f"Policy-induction trajectory — {trajectory_jsonl.parent.name}")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
