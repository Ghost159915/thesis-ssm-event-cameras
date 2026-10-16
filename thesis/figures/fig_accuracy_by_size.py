#!/usr/bin/env python
"""Report figure: test AP of each of our models minus S5-RVT, overall and per COCO object-size class.

Message: the gap to the baseline sits almost entirely in large objects (EventSSM AP_L -6.0); BiMamba spatial mixing
(PureSSM) closes about half of it (Stage 15). Graded 400k is added automatically once its Stage-21 eval exists.
Bars are differences in AP points (x 100) on the Gen1 test set; S5-RVT is the zero line. Single seed per model, so a
difference of about 1 point is within run-to-run noise; the bootstrap intervals (Stage 21) are added as error bars
when results/stage21_bootstrap/vs_s5rvt/bootstrap.json exists.

  python thesis/figures/test_metrics.py && python thesis/figures/fig_accuracy_by_size.py
  -> thesis/latex/images/fig_accuracy_by_size.{pdf,png} + results/report_figs/fig_accuracy_by_size.csv
"""
from __future__ import annotations

import json

import numpy as np

import report_style as rs
from report_style import plt

METRICS = [("AP", "All objects"), ("AP_S", "Small"), ("AP_M", "Medium"), ("AP_L", "Large")]
SERIES = [("eventssm", "eventssm"), ("puressm", "puressm"), ("graded_400k", "graded")]   # result key, style key


def main() -> None:
    rs.apply()
    test = json.loads((rs.DATA / "test_metrics.json").read_text())
    base = test["s5rvt"]
    series = [(k, s) for k, s in SERIES if k in test]

    fig, ax = plt.subplots(figsize=(rs.TEXT_WIDTH, 2.7))
    x = np.arange(len(METRICS))
    width = 0.8 / len(series)
    rows = []
    for i, (key, style) in enumerate(series):
        deltas = [test[key][m] - base[m] for m, _ in METRICS]
        pos = x - 0.4 + width * (i + 0.5)
        name = "SpikingSSM (graded)" if key == "graded_400k" else rs.label(style)
        ax.bar(pos, deltas, width * 0.92, color=rs.colour(style), label=name, zorder=2)
        for p, d, (m, _) in zip(pos, deltas, METRICS):
            rows.append({"model": name, "metric": m, "model_AP": round(test[key][m], 2),
                         "s5rvt_AP": round(base[m], 2), "delta": round(d, 2)})
            if m in ("AP", "AP_L"):                        # selective labels: the overall gap and the finding
                ax.annotate(rs.signed(d), (p, d), xytext=(0, -3 if d < 0 else 3), textcoords="offset points",
                            ha="center", va="top" if d < 0 else "bottom", fontsize=7.5, color=rs.INK)
    ax.axhline(0, color=rs.INK, linewidth=0.9, zorder=3)
    ax.text(len(METRICS) - 0.5, 0.15, "S5-RVT", ha="right", va="bottom", fontsize=7.5, color=rs.INK_2)
    ax.set_xticks(x, [f"{name}\n(S5-RVT {base[m]:.1f})" for m, name in METRICS])
    ax.set_ylabel("Test AP difference to S5-RVT\n(points, COCO 0.50:0.95)")
    ax.set_ylim(min(r["delta"] for r in rows) - 1.2, 1.5)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", ncol=len(series))
    rs.save(fig, "fig_accuracy_by_size", rows)


if __name__ == "__main__":
    main()
