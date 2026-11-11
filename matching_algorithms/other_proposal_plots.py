"""Figures for proposal 3's experiments, made from the CSVs other_proposal_sim.py writes.

    python matching_algorithms/other_proposal_plots.py exports/other_proposal
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette (light mode), in fixed order, and text/surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED_MARK = "#b9b8b1"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"

PANELS = [
    ("spearman", "Spearman (taste)", True),
    ("ndcg10", "NDCG@10 (taste)", True),
    ("precision_at_k", "Precision@10, true matches", True),
    ("cum_regret_match", "Cumulative match regret", False),
    ("exposure_gini", "Exposure Gini", False),
]
DEFAULT_LABELS = {"proposal (default)", "15", "d-optimal", "thompson", "with prior", "shrunk_intercept", "harmonic", "with cap"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ablation_figure(path: Path, out: Path) -> None:
    """One row of small multiples: each metric's mean ± SD across seeds, per config."""
    rows = _read(path)
    name = path.stem.removeprefix("summary_")
    labels = [r["label"] for r in rows]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.0 * len(PANELS), 0.45 * len(rows) + 1.3), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (metric, title, higher) in zip(axes, PANELS):
        mean = np.array([float(r[metric]) for r in rows])
        sd = np.array([float(r[f"{metric}_sd"]) for r in rows])
        colors = [SERIES[0] if l in DEFAULT_LABELS else MUTED_MARK for l in labels]
        ax.barh(y, mean, height=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(mean, y, xerr=sd, fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=2)
        for yi, m, s in zip(y, mean, sd):
            ax.text(max(m, 0) + s, yi, f"  {m:.3f}", va="center", ha="left", fontsize=8, color=TEXT)
        ax.set_title(title + ("  ↑" if higher else "  ↓"))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(min(0, mean.min()), max((mean + sd).max() * 1.4, 1e-6))
    axes[0].set_yticks(y, labels)
    fig.suptitle(f"Ablation: {name.replace('_', ' ')}  (blue = default config; mean ± SD over seeds)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"ablation_{name}.png", dpi=150)
    plt.close(fig)


def regret_figure(curves_path: Path, out: Path, ablation: str) -> None:
    """Cumulative match regret per user over bandit rounds, one line per config."""
    curves = json.loads(curves_path.read_text())
    by_label: dict[str, list[list[float]]] = {}
    for key, value in curves.items():
        a, label, _ = key.split("|")
        if a == ablation:
            by_label.setdefault(label, []).append(value["match"])
    if not by_label:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    # The legend lists configs in the order of their final regret, worst first, so it
    # reads top to bottom like the lines do. End labels would collide where lines meet.
    lines = []
    for color, (label, runs) in zip(SERIES, by_label.items()):
        n = min(len(r) for r in runs)
        mean = np.mean([r[:n] for r in runs], axis=0)
        (line,) = ax.plot(np.arange(1, n + 1), mean, color=color, linewidth=2, label=f"{label} ({mean[-1]:.2f})")
        lines.append((mean[-1], line))
    lines.sort(key=lambda t: -t[0])
    ax.set_xlabel("Bandit round")
    ax.set_ylabel("Cumulative regret per user\n(true match probability)")
    ax.set_title(f"Match regret: {ablation.replace('_', ' ')} (lower is better)")
    ax.legend([l for _, l in lines], [l.get_label() for _, l in lines], frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"regret_{ablation}.png", dpi=150)
    plt.close(fig)


def audit_figure(out: Path) -> None:
    """Religion audit: exposure vs population share, predicted vs true, and learned weights."""
    groups = _read(out / "religion_audit_groups.csv")
    weights = _read(out / "religion_audit_weights.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    names = [g["religion"] for g in groups]
    y = np.arange(len(names))[::-1]
    for ax, (a, b, la, lb, title) in zip(
        axes[:2],
        [
            ("population_share", "exposure_share", "Share of profiles", "Share of impressions", "Exposure by candidate religion"),
            ("mean_true", "mean_predicted", "True mean score", "Predicted mean score", "Scores by candidate religion"),
        ],
    ):
        va, vb = [float(g[a]) for g in groups], [float(g[b]) for g in groups]
        ax.hlines(y, np.minimum(va, vb), np.maximum(va, vb), color=GRID, linewidth=2)
        ax.scatter(va, y, s=40, color=SERIES[0], label=la, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.scatter(vb, y, s=40, color=SERIES[1], label=lb, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(y, names)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    feats = [w["feature"].removeprefix("cand_religion=") for w in weights]
    pop = np.array([float(w["population_weight"]) for w in weights])
    sd = np.array([float(w["sd_user_weight"]) for w in weights])
    yw = np.arange(len(feats))[::-1]
    ax = axes[2]
    ax.barh(yw, pop, height=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.errorbar(pop, yw, xerr=sd, fmt="none", ecolor=TEXT_2, elinewidth=1, capsize=2)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(yw, feats)
    ax.set_title("Learned weights\n(population; whiskers = SD across users)", pad=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "religion_audit.png", dpi=150)
    plt.close(fig)


def make_figures(out: str | Path) -> None:
    out = Path(out)
    _style()
    for path in sorted(out.glob("summary_*.csv")):
        ablation_figure(path, out)
    for ablation in ("baselines", "policy", "exposure_cap", "calibration", "onboarding_strategy"):
        regret_figure(out / "regret_curves.json", out, ablation)
    if (out / "religion_audit_groups.csv").exists():
        audit_figure(out)


if __name__ == "__main__":
    make_figures(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "exports" / "other_proposal")
