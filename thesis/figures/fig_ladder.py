#!/usr/bin/env python
"""Report figure: the cost of spiking, step by step (the readout ladder, 100k-step schedule, Gen1 test set).

Message: moving from PureSSM to the binary readout costs 2.3 mAP, and the paired bootstrap splits it into the LIF
dynamics (not distinguishable from zero), sparsity (the largest step) and binarisation (smaller, but excludes zero).

(a) test mAP per arm with its 95 % interval: the four arms overlap, because each interval carries the difficulty of
    the test recordings, which every arm shares;
(b) the paired change at each step with its 95 % interval about zero: pairing cancels the shared difficulty, which
    is why these intervals are much narrower than the per-arm ones in (a). Each step is drawn in the colour of the
    arm it arrives at.

Input: results/stage21_bootstrap/ladder_100k/bootstrap.json (event_ssm.eval.bootstrap; B = 1000, seed 0, 470 test
recordings, percentile intervals; test-set sampling only, one training seed per arm).

  python thesis/figures/fig_ladder.py
  -> thesis/latex/images/fig_ladder.{pdf,png} + results/report_figs/fig_ladder.csv
"""
from __future__ import annotations

import json

import report_style as rs
from report_style import plt

BOOT = rs.REPO / "results/stage21_bootstrap/ladder_100k/bootstrap.json"
ARMS = [("puressm", "puressm", "PureSSM"), ("analog", "analog", "Analog"),
        ("graded", "graded", "Graded"), ("binary", "spike", "Spike")]      # bootstrap name, style key, tick label
STEPS = [("puressm:analog", "analog", "LIF dynamics"), ("analog:graded", "graded", "Sparsity"),
         ("graded:binary", "spike", "Binarisation")]       # PureSSM → analog → graded → spike (caption)


def main() -> None:
    rs.apply()
    r = json.loads(BOOT.read_text())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(rs.TEXT_WIDTH, 2.6), gridspec_kw={"width_ratios": [1, 1.15],
                                                                                   "wspace": 0.38})
    rows = []
    # (a) per arm
    for x, (name, style, tick) in enumerate(ARMS):
        s = r["arms"][name]["AP"]
        p, lo, hi = (100 * s[k] for k in ("point", "ci_lo", "ci_hi"))
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt=rs.marker(style), color=rs.colour(style), ms=6,
                    capsize=3, elinewidth=1.2, markeredgecolor="white", markeredgewidth=0.6)
        ax.text(x + 0.12, p, f"{p:.2f}", ha="left", va="center", fontsize=7.5, color=rs.INK)
        rows.append({"panel": "a", "item": name, "ap": round(p, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)})
    ax.set_xticks(range(len(ARMS)), [t for _, _, t in ARMS])
    ax.set_xlim(-0.4, len(ARMS) - 0.3)
    ax.set_ylabel("Test mAP")
    ax.grid(axis="x", visible=False)
    rs.panel_tag(ax, "a", "each model")

    # (b) paired step costs
    bx.axhline(0, color=rs.INK_2, linewidth=0.9, zorder=1)
    for x, (pair, style, tick) in enumerate(STEPS):
        s = r["gaps"][pair]["AP"]
        p, lo, hi = (100 * s[k] for k in ("point", "ci_lo", "ci_hi"))
        bx.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt=rs.marker(style), color=rs.colour(style), ms=6,
                    capsize=3, elinewidth=1.2, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        verdict = "not clear" if lo <= 0 <= hi else "clear cost"
        bx.text(x + 0.12, p, f"{rs.signed(p, 2)}\n{verdict}",
                ha="left", va="center", fontsize=7, color=rs.INK, linespacing=1.15)    # right of the point
        rows.append({"panel": "b", "item": pair, "ap": round(p, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                     "frac_reps_ge_0": s["frac_reps_ge_0"]})
    bx.set_xticks(range(len(STEPS)), [t for _, _, t in STEPS])
    bx.set_xlim(-0.35, len(STEPS) - 0.05)
    bx.set_ylim(-2.3, 0.6)
    bx.set_ylabel("Change in test mAP")
    bx.grid(axis="x", visible=False)
    rs.panel_tag(bx, "b", "each step")
    rs.save(fig, "fig_ladder", rows)


if __name__ == "__main__":
    main()
