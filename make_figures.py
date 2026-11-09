"""Build paper figures from the locked summary files (no private data is read).

Run from the repository root:  python 08_final_submission/iccies2027/scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
L2 = json.loads((ROOT / "06_experiment_results/sslmd_l2arctic_crc_v1/summary.json").read_text(encoding="utf-8"))
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

# Default validated categorical slots 1-3; markers give a second, non-colour encoding.
STYLE = {"sslmd": ("SSL-MD", "#2a78d6", "o"), "gop": ("GOP", "#eb6834", "s"), "prior": ("Phone prior", "#1baf7a", "^")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def fig_per_speaker():
    """Per-speaker FPR at alpha=0.10: CRC vs uncorrected empirical threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.2), sharey=True)
    for ax, score in zip(axes, ["sslmd", "gop"]):
        name, color, marker = STYLE[score]
        for x, method in enumerate(["crc", "empirical"]):
            rows = L2["P1_loso"][f"{score}__{method}__0.10"]["per_speaker"]
            fpr = sorted(r["fpr"] for r in rows)
            jitter = [x + (i - len(fpr) / 2) * 0.022 for i in range(len(fpr))]
            over = [f > 0.10 for f in fpr]
            ax.scatter([j for j, o in zip(jitter, over) if not o], [f for f, o in zip(fpr, over) if not o],
                       s=12, facecolor=color, edgecolor="none", marker=marker, zorder=3)
            ax.scatter([j for j, o in zip(jitter, over) if o], [f for f, o in zip(fpr, over) if o],
                       s=14, facecolor="white", edgecolor=color, linewidth=1.0, marker=marker, zorder=3)
            ax.text(x, 0.185, f"{sum(over)}/24 > α", ha="center", fontsize=7, color=MUTED)
        ax.axhline(0.10, color=INK, linewidth=0.8, linestyle="--", zorder=2)
        ax.set_xticks([0, 1], ["CRC", "Empirical"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(name, fontsize=8, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
    axes[0].set_ylabel("False-warning rate (per speaker)")
    axes[0].set_ylim(0, 0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_speaker_fpr.pdf")
    plt.close(fig)


def fig_tradeoff():
    """Mean FPR vs mean recall under CRC for alpha in {0.05, 0.10, 0.15}."""
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for score, (name, color, marker) in STYLE.items():
        pts = [L2["P1_loso"][f"{score}__crc__{a}"] for a in ("0.05", "0.10", "0.15")]
        ax.plot([p["mean_fpr"] for p in pts], [p["mean_recall"] for p in pts], color=color, linewidth=2,
                marker=marker, markersize=5, markeredgecolor="white", label=name, zorder=3)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_xlabel("Mean false-warning rate (held-out speaker)")
    ax.set_ylabel("Mean recall")
    ax.set_xlim(0, 0.16)
    ax.set_ylim(0, 0.40)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_crc_tradeoff.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_per_speaker()
    fig_tradeoff()
    print("written to", OUT)
