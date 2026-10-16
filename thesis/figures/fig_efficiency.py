#!/usr/bin/env python
"""Report figure: the efficiency trade-space of the three detectors (RTX 5070 Ti, bf16, batch 1, streaming).

(a) Where a frame's time goes (eager p50 per component: backbone / neck+head / post-processing), with the
    graph-deploy latency (CUDA-graph network replay + eager post-processing) as a marker. Message: the neck and head
    are the shared frozen pipeline and cost the same in every model, so every latency difference is the backbone;
    PureSSM does the least arithmetic (GFLOPs in the row label) yet is the slowest eager model because its many small
    scan kernels are launch-bound, which graph replay removes.
(b) Energy per frame (GPU board power above idle x time, eager). (c) Recurrent state carried per stream.
    Message: EventSSM is the fastest and most frugal per frame, and pays for it with a 31x larger streaming state.
The numbers are those of Table tab:efficiency (the three-model Stage-16 run in results/stage10/bench_results.json).

  python thesis/figures/fig_efficiency.py [--bench results/stage10/bench_results.json]
  -> thesis/latex/images/fig_efficiency.{pdf,png} + results/report_figs/fig_efficiency.csv
"""
from __future__ import annotations

import argparse
import json

import report_style as rs
from report_style import plt

ORDER = [("baseline", "s5rvt"), ("eventssm", "eventssm"), ("puressm", "puressm")]   # bench key, style key
NECK, POST = "#b4b2a9", "#dcdad3"                     # shared frozen pipeline: neutral, identical in every row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(rs.REPO / "results/stage10/bench_results.json"))
    bench = json.loads(open(ap.parse_args().bench).read())["models"]
    rs.apply()
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(rs.TEXT_WIDTH, 2.15), sharey=True,
                                     gridspec_kw={"width_ratios": [2.2, 1, 1], "wspace": 0.18})
    rows, ticks = [], []
    for y, (key, style) in enumerate(ORDER):
        m = bench[key]
        lat = m["latency"]["bf16"]
        comp = {k: v["p50_ms"] for k, v in lat["components"].items()}
        graph = m["latency"]["graph"]
        deploy = graph["p50_ms"] + comp["postprocess"] if "p50_ms" in graph else None
        full = lat["full"]["p50_ms"]
        state_mb = m["vram"]["state_kb_per_stream"] / 1024
        energy = m["energy"]["j_per_frame"]
        gflops = m["flops"]["total_gflops"]
        name = rs.label(style).replace(" (baseline)", "")
        ticks.append(f"{name}\n{m['params_m']['total']:.1f} M · {gflops:.1f} GFLOPs")

        left = 0
        for part, w, c in (("backbone", comp["backbone"], rs.colour(style)), ("neck+head", comp["neck_head"], NECK),
                           ("post-processing", comp["postprocess"], POST)):
            ax.barh(y, w, left=left, height=0.62, color=c, edgecolor="white", linewidth=1)
            if y == 0 and part != "post-processing":       # direct labels on the first row instead of a legend
                ax.text(left + w / 2, y, part.replace("+", "+\n"), ha="center", va="center", fontsize=6.5,
                        linespacing=1.0,
                        color="white" if part == "backbone" else rs.INK)
            left += w
        ax.annotate(f"{full:.1f} ms", (left, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=rs.INK)
        if deploy:
            ax.plot(deploy, y + 0.42, "v", color=rs.INK, markersize=4.5)
            ax.annotate(f"CUDA graph {deploy:.1f} ms", (deploy, y + 0.42), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7, color=rs.INK_2)
        bx.barh(y, energy, height=0.62, color=rs.colour(style))
        bx.annotate(f"{energy:.2f}", (energy, y), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=rs.INK)
        cx.barh(y, state_mb, height=0.62, color=rs.colour(style))
        cx.annotate(f"{state_mb:.1f}", (state_mb, y), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=rs.INK)
        rows.append({"model": name, "params_M": round(m["params_m"]["total"], 2), "gflops": round(gflops, 2),
                     **{f"{k}_p50_ms": round(v, 2) for k, v in comp.items()}, "full_p50_ms": round(full, 2),
                     "graph_deploy_p50_ms": round(deploy, 2) if deploy else "", "energy_J_per_frame": round(energy, 3),
                     "state_MB_per_stream": round(state_mb, 1)})

    ax.set_yticks(range(len(ORDER)), ticks)
    ax.invert_yaxis()
    ax.set_xlabel("Latency per frame, p50 (ms)")
    ax.set_xlim(0, 33)
    bx.set_xlabel("Energy (J/frame)")
    bx.set_xlim(0, 0.95)
    cx.set_xlabel("State (MB/stream)")
    cx.set_xlim(0, 200)
    for a, t in ((ax, "a"), (bx, "b"), (cx, "c")):
        a.grid(axis="y", visible=False)
        rs.panel_tag(a, t)
    rs.save(fig, "fig_efficiency", rows)


if __name__ == "__main__":
    main()
