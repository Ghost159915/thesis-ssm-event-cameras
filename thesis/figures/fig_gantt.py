#!/usr/bin/env python
"""Report figure (Appendix, Project Timeline): the Gantt chart registered in Thesis A, redrawn in the report style.

Replaces thesis/latex/images/"Ghantt Chart.png" (no source file existed), whose right-aligned task names ran under
the first bar. The schedule is reproduced exactly as registered: bar extents and milestone weeks were measured from
the pixels of the original PNG (week grid: week 1 at x = 168 px, week 40 at x = 1524 px). It is the Thesis-A plan,
not what happened (the caption says so; simulation integration was later descoped).

Bars are split at the term boundaries and coloured by term with an ordinal ramp of the palette's sequential blue
(Thesis A light -> C dark; validated with `validate_palette.js --ordinal`), so the term each week of work falls in is
visible without reading the axis. Task names sit in the left margin, outside the plot, so nothing overlaps.

  python thesis/figures/fig_gantt.py   -> thesis/latex/images/fig_gantt.{pdf,png} + results/report_figs/fig_gantt.csv
"""
from __future__ import annotations

from matplotlib.patches import Patch

import report_style as rs
from report_style import plt

TERMS = [("Thesis A", 1, 12, "#86b6ef"), ("Thesis B", 13, 27, "#2a78d6"), ("Thesis C", 28, 40, "#104281")]
TASKS = [  # (name, first week, last week, milestone week or None, milestone label)
    ("Literature review & foundations", 1, 12, None, ""),
    ("MVP reproduction (S5-RVT)", 8, 16, 16, "M1"),
    ("Architecture design", 13, 18, 19, "M2"),
    ("Dataset preparation", 14, 18, None, ""),
    ("Model training & benchmarking", 18, 24, 24, "M3"),
    ("Temporal generalisation eval.", 23, 26, None, ""),
    ("Simulation integration", 25, 31, 31, "M4"),
    ("Thesis writing", 28, 40, 40, "M5"),
]
BAR_H = 0.56


def main() -> None:
    rs.apply()
    fig, ax = plt.subplots(figsize=(rs.TEXT_WIDTH, 3.0))
    rows = []
    for y, (name, a, b, mw, ml) in enumerate(TASKS):
        for term, t0, t1, c in TERMS:                     # one segment per term the task overlaps
            lo, hi = max(a, t0), min(b, t1)
            if lo <= hi:
                ax.barh(y, hi - lo + 1, left=lo - 0.5, height=BAR_H, color=c, edgecolor="white", linewidth=0.8,
                        zorder=2)
        if mw:
            ax.plot(mw, y, "D", color=rs.INK, markersize=5.5, markeredgecolor="white", markeredgewidth=0.8, zorder=4)
            ax.annotate(ml, (mw, y), xytext=(0, 6.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=rs.INK, fontweight="bold")
        rows.append({"task": name, "first_week": a, "last_week": b, "milestone": ml, "milestone_week": mw or ""})

    for _, t0, _, _ in TERMS[1:]:                         # term boundaries
        ax.axvline(t0 - 0.5, color=rs.AXIS, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    for term, t0, t1, _ in TERMS:
        ax.text((t0 + t1) / 2, -0.95, term, ha="center", va="bottom", fontsize=8, color=rs.INK_2)

    ax.set_yticks(range(len(TASKS)), [t[0] for t in TASKS])
    ax.set_ylim(len(TASKS) - 0.4, -1.05)                  # first task on top, room for the term labels
    ax.set_xlim(0.4, 40.6)
    ax.set_xticks(range(1, 41))
    ax.set_xticklabels([str(w) if w in (1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40) else "" for w in range(1, 41)])
    ax.set_xlabel("Week")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8, colors=rs.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=rs.GRID, linewidth=0.4)
    ax.spines["left"].set_visible(False)
    handles = [Patch(color=c, label=t) for t, _, _, c in TERMS]
    handles.append(plt.Line2D([], [], marker="D", color=rs.INK, linestyle="none", markersize=5, label="milestone"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=7.5,
              handlelength=1.2, columnspacing=1.6)
    rs.save(fig, "fig_gantt", rows)


if __name__ == "__main__":
    main()
