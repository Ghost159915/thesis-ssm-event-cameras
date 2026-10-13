#!/usr/bin/env python
"""Figures for progress_report_2026-10.tex (CPU only, no model runs).

fig_ladder_100k.pdf/.png  validation AP of the four 100k-step runs (Stage 20). Values are the `val/AP` lines of the
                          console logs results/stage20/{puressm_100k,analog_s234_100k,graded_s234_100k,
                          spike_s234_100k}/console_*.log, as recorded in docs/notes/Stage20_fullrun_notes.md §3.
fig_lif_panel.png         panel c of thesis/latex/images/fig_spiking_architecture.png (cropped, not redrawn).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

STEPS = list(range(10, 101, 10))                       # thousands of steps
RUNS = {                                               # val/AP x 100, Gen1 validation split, one run each
    "PureSSM": [23.96, 31.18, 33.18, 36.69, 38.91, 39.04, 40.15, 43.05, 43.80, 44.22],
    "analog":  [22.95, 29.40, 32.63, 37.03, 38.30, 39.59, 41.65, 41.75, 43.64, 43.84],
    "graded":  [19.17, 27.85, 30.87, 35.20, 36.85, 38.75, 39.36, 41.61, 41.48, 42.45],
    "spike":   [20.50, 30.34, 31.85, 35.10, 37.04, 37.92, 39.31, 40.37, 40.82, 41.80],
}
# Reference categorical palette, slots 1-4 in fixed order (validated: CVD/normal-vision PASS; slots 3-4 are below
# 3:1 contrast on white, so every series also has its own marker and the values are tabulated in the report).
COLOR = dict(zip(RUNS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MARKER = dict(zip(RUNS, ["o", "s", "^", "D"]))
INK, INK_2, RULE = "#0b0b0b", "#52514e", "#d8d7d2"


def style(ax):
    ax.grid(color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK_2, labelsize=9)


def ladder_figure():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.55, 1]})
    for name, ys in RUNS.items():
        ax.plot(STEPS, ys, color=COLOR[name], linewidth=2, marker=MARKER[name], markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, label=name)
    ax.set_xlabel("Training step (thousands)", fontsize=10, color=INK)
    ax.set_ylabel("Validation AP", fontsize=10, color=INK)
    ax.set_title("Training curves, 100k-step schedule", fontsize=10.5, color=INK, loc="left", fontweight="bold")
    ax.set_xticks(STEPS)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)

    names = list(RUNS)
    finals = [RUNS[n][-1] for n in names]
    for i, (n, y) in enumerate(zip(names, finals)):
        bx.plot(i, y, MARKER[n], color=COLOR[n], markersize=9, markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        bx.annotate(f"{y:.1f}", (i, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9.5,
                    color=INK, fontweight="bold")
    for i in range(len(names) - 1):                    # the cost of each step, in a row between the two arms
        d = finals[i + 1] - finals[i]
        bx.text(i + 0.5, 40.85, f"{d:+.1f}".replace("-", "−"), ha="center", va="center", fontsize=9, color=INK_2)
    bx.set_xticks(range(len(names)), names, fontsize=9.5, color=INK)
    bx.set_xlim(-0.5, len(names) - 0.5)
    bx.set_ylim(40.5, 45.5)
    bx.set_ylabel("Validation AP at 100k", fontsize=10, color=INK)
    bx.set_title("The ladder at 100k", fontsize=10.5, color=INK, loc="left", fontweight="bold")
    style(bx)
    fig.text(0.01, 0.005, "Gen1 validation split, one training run per arm. Bottom row of the right panel: change from "
             "the arm to the left. Test-set values with 95 % bootstrap intervals are pending (Stage 21).",
             fontsize=8, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in (".pdf", ".png"):
        fig.savefig(HERE / f"fig_ladder_100k{ext}", dpi=300, bbox_inches="tight", facecolor="white")


def lif_panel():
    im = Image.open(REPO / "thesis/latex/images/fig_spiking_architecture.png")
    w, h = im.size                                     # 3900 x 3900; panel c starts below the panel-b footnotes
    im.crop((0, int(0.693 * h) + 120, w, int(0.985 * h))).save(HERE / "fig_lif_panel.png")   # +120: drop the panel title


if __name__ == "__main__":
    ladder_figure()
    lif_panel()
