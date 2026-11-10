#!/usr/bin/env python
"""Report figure: post-hoc threshold sensitivity of the graded and binary readouts (100k-step schedule, Gen1 test).

Message: the threshold the readouts were trained with (theta = 1) is the most accurate point on both curves. Lowering
theta after training raises the firing rate without gaining accuracy; raising it lowers the rate at a steep
accuracy cost. Every point other than theta = 1 re-evaluates the SAME weights, so the curves measure sensitivity to
the threshold, not what a model trained at that sparsity would reach. The dense analog readout (rate 100 %) is the
reference line.

Input: results/stage22/threshold_sweep/threshold_pareto.json (event_ssm.eval.threshold_sweep; spike-weighted
nonzero rate = share of the spike-fed neck operations driven, over the whole test set after a 32-frame warm-up).

  python thesis/figures/fig_threshold.py
  -> thesis/latex/images/fig_threshold.{pdf,png} + results/report_figs/fig_threshold.csv
"""
from __future__ import annotations

import json

import report_style as rs
from report_style import plt

SWEEP = rs.REPO / "results/stage22/threshold_sweep/threshold_pareto.json"
BOOT = rs.REPO / "results/stage21_bootstrap/ladder_100k/bootstrap.json"
ARMS = [("graded", "graded", "Graded", (22, 36.5)), ("binary", "spike", "Spike", (50, 33.5))]
# sweep key, style key, direct label, label position (rate %, mAP)
OFFSETS = {  # (dx, dy) in points for the theta labels, per arm, so they clear the line and each other
    "graded": {0.5: (6, 8), 0.75: (12, 8), 1.0: (-12, 8), 1.5: (-17, 2), 2.0: (-15, 0), 3.0: (14, -2)},
    "binary": {0.5: (18, 0), 0.75: (0, -12), 1.0: (0, 11), 1.5: (18, -2), 2.0: (15, -2), 3.0: (15, 0)},
}


def main() -> None:
    rs.apply()
    sweep = json.loads(SWEEP.read_text())
    analog = 100 * json.loads(BOOT.read_text())["arms"]["analog"]["AP"]["point"]
    fig, ax = plt.subplots(figsize=(rs.TEXT_WIDTH * 0.72, 2.9))
    ax.axhline(analog, color=rs.MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(1, analog + 0.35, f"analog readout (dense, rate 100 %): {analog:.2f}", ha="left", va="bottom",
            fontsize=7.5, color=rs.INK_2)
    rows = []
    for key, style, name, (lx, ly) in ARMS:
        pts = sorted(sweep[key], key=lambda p: p["weighted_rate"])
        x = [100 * p["weighted_rate"] for p in pts]
        y = [100 * p["ap"] for p in pts]
        ax.plot(x, y, color=rs.colour(style), marker=rs.marker(style), ms=4.5, markeredgecolor="white",
                markeredgewidth=0.5, zorder=3)
        for p, xi, yi in zip(pts, x, y):
            th = p["threshold"]
            if th == 1.0:
                ax.plot(xi, yi, "o", ms=10, markerfacecolor="none", markeredgecolor=rs.INK, markeredgewidth=1.2,
                        zorder=4)
            ax.annotate(f"θ={th:g}", (xi, yi), xytext=OFFSETS[key][th], textcoords="offset points",
                        ha="center", va="center", fontsize=7, color=rs.INK if th == 1.0 else rs.INK_2,
                        fontweight="bold" if th == 1.0 else "normal")
            rows.append({"arm": key, "threshold": th, "weighted_rate_pct": round(xi, 2), "test_map": round(yi, 2),
                         "posthoc": th != 1.0})
        ax.text(lx, ly, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color=rs.colour(style))
    ax.set_xlim(0, 56)
    ax.set_ylim(21, 44.5)
    ax.set_xlabel("Firing rate (% of spike-fed operations driven)")
    ax.set_ylabel("Test mAP (100k steps)")
    ax.plot([], [], "o", ms=8, markerfacecolor="none", markeredgecolor=rs.INK, markeredgewidth=1.2,
            label="training threshold θ = 1")
    ax.legend(loc="lower right")
    rs.save(fig, "fig_threshold", rows)


if __name__ == "__main__":
    main()
