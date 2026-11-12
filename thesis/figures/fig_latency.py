#!/usr/bin/env python
"""Report figure (Discussion, Section 'Latency and the Choice of Window'): where the reaction time goes.

Each bar is the time from an event reaching the detector to the detection that uses it, split into
  - waiting for the accumulation window to close: on average half the window (25 ms for 50 ms, 2.5 ms for 5 ms);
    the worst case, a full window, is written next to the bar;
  - processing one frame: the measured median time per frame of the citable benchmark (results/stage10/
    bench_results.json): normal (eager) mode for the 50-ms rows, CUDA-graph replay of the network plus eager
    post-processing for the 5-ms rows (as fig_efficiency and tab:efficiency), because
    in normal mode PureSSM (26.9 ms) cannot keep up with a new frame every 5 ms.
Message: with 50-ms windows the reaction time is 38-52 ms, set by the window and the processing, not by the
camera (microseconds); with 5-ms windows and replay mode it falls to about 7-8 ms.

  python thesis/figures/fig_latency.py   -> thesis/latex/images/fig_latency.{pdf,png} + results/report_figs/fig_latency.csv
"""
from __future__ import annotations

import json

import report_style as rs
from report_style import plt

BENCH = rs.REPO / "results/stage10/bench_results.json"
MODELS = [("eventssm", "eventssm"), ("puressm", "puressm")]          # JSON key, style key


def main() -> None:
    rs.apply()
    bench = json.loads(BENCH.read_text())["models"]
    rows, bars = [], []
    for window, mode in ((50.0, "normal"), (5.0, "replay")):
        for key, style in MODELS:
            lat = bench[key]["latency"]
            post = lat["bf16"]["components"]["postprocess"]["p50_ms"]          # eager post-processing
            # replay mode = CUDA-graph network replay + eager post-processing, as fig_efficiency and the
            # CUDA-graph throughput row of tab:efficiency (210 / 181 Hz)
            proc = lat["bf16"]["full"]["p50_ms"] if mode == "normal" else lat["graph"]["p50_ms"] + post
            label = f"{rs.label(style)}, {window:.0f}-ms window" + ("" if mode == "normal" else ",\nreplay mode")
            bars.append((label, window / 2, proc, style, window))
            rows.append({"model": style, "window_ms": window, "mode": mode, "wait_avg_ms": window / 2,
                         "wait_max_ms": window, "processing_ms": round(proc, 2), "total_avg_ms": round(window / 2 + proc, 2)})

    fig, ax = plt.subplots(figsize=(0.85 * rs.TEXT_WIDTH, 2.3))
    ys = list(range(len(bars)))[::-1]                                      # first bar on top
    for y, (label, wait, proc, style, window) in zip(ys, bars):
        ax.barh(y, wait, color=rs.AXIS, height=0.55, edgecolor="white", linewidth=0.6)
        ax.barh(y, proc, left=wait, color=rs.colour(style), height=0.55, edgecolor="white", linewidth=0.6)
        ax.text(wait + proc + 1.0, y, f"{wait + proc:.0f} ms  (up to {window + proc:.0f})", va="center",
                fontsize=7.5, color=rs.INK)
    ax.set_yticks(ys, [b[0] for b in bars], fontsize=7.5)
    ax.set_xlim(0, 80)
    ax.set_xlabel("Time from event to detection (ms)")
    ax.grid(axis="y", visible=False)
    ax.minorticks_off()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=rs.AXIS, label="waiting for the window to close (average)"),
                       Patch(color=rs.INK_2, label="processing one frame (colour: model)")],
              loc="lower right", fontsize=7, frameon=False)
    rs.save(fig, "fig_latency", rows)


if __name__ == "__main__":
    main()
