#!/usr/bin/env python
"""Report figure: event-rate robustness, as retention = AP at a rate / the model's own AP at the training rate (1x).

(a) Regime 1, fixed 50-ms step cadence, accumulation window varied (0.25x = 200 ms ... 4x = 12 ms). Message: every
    model peaks at its training window and the three degrade alike; PureSSM holds best at the dense end.
(b) Regime 2, true rate change (step = window, gap-free; 2x = 25 ms, 10x = 5 ms), without (solid) and with (dashed)
    inference-time Delta-t compensation, the RVT ConvLSTM measured on the same test set (published checkpoint, no
    compensation exists for it), and the two published reference points of Zubic et al. Message: without
    compensation the SSMs keep 62-70 % at 10x and the ConvLSTM 80 %; the compensation costs 9-13 points at 10x for all
    three SSMs; neither published 10x value (S5 + comp. 83.5 %, ConvLSTM 17.7 %) is reproduced.
Second figure, fig_robustness_spiking (only when the true-rate logs of the spiking arms exist): the same
    uncompensated protocol for the readout ladder, one x position per arm in ladder order (PureSSM, analog, graded,
    spike) as in fig_ladder. Each arm is a drop from its 2x retention (small marker) to its 10x retention (large
    marker): 100k-step schedule filled + solid (the fair ladder comparison), 400k-step PureSSM and graded hollow +
    dashed to the right of their 100k run (values right of each marker). Each arm's 1x reference is its canonical Stage-21 test AP. One x position per arm keeps
    analog yellow away from PureSSM orange (report_style: never two adjacent lines in that pair).
Values are read from the eval logs of results/stage9/{sweep,sweep_tr}/ and results/stage9_rvt_lstm/ (Tables
tab:regime1, tab:regime2, tab:dt_comp).

  python thesis/figures/fig_robustness.py   -> thesis/latex/images/fig_robustness{,_spiking}.{pdf,png}
                                               + results/report_figs/*.csv
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import report_style as rs
from report_style import plt

S9 = rs.REPO / "results/stage9"
LSTM = rs.REPO / "results/stage9_rvt_lstm"
MODELS = [("baseline", "s5rvt"), ("eventssm", "eventssm"), ("puressm", "puressm")]      # log prefix, style key
WINDOWS = [(200, 0.25), (100, 0.5), (50, 1.0), (25, 2.0), (12, 4.0)]                     # regime 1: dt ms, mult
TRUE_RATE = [(25, 2.0, "0.5000"), (5, 10.0, "0.1000")]                                   # regime 2: dt, mult, step scale
# Zubic et al. 2024, Table 2 (Gen1 test): S5 47.71 -> 39.84 and RVT 47.16 -> 8.35 from 20 to 200 Hz.
PUBLISHED = {"S5 + comp.": 39.84 / 47.71 * 100, "ConvLSTM": 8.35 / 47.16 * 100}
# (sweep_tr log key, test_metrics key, style key, arm position on the x axis, schedule)
SPIKING = [("puressm_100k", "puressm_100k", "puressm", 0, "100k"),
           ("analog_s234_100k", "analog_100k", "analog", 1, "100k"),
           ("graded_s234_100k", "graded_100k", "graded", 2, "100k"),
           ("spike_s234_100k", "spike_100k", "spike", 3, "100k"),
           ("puressm", "puressm", "puressm", 0, "400k"),
           ("graded_s234_400k", "graded_400k", "graded", 2, "400k")]
ARM_TICKS = ["PureSSM", "Analog", "Graded", "Spike"]                                    # tick labels as fig_ladder
PAIR = 0.25                                                                              # 100k/400k pair: ±PAIR about the tick
ROW = re.compile(r"│\s*test/AP\s*│\s*([0-9.]+)\s*│")
LINE_PT = 9.0                                                                            # one 7.5-pt label line + leading


def ap(log: Path | None) -> float | None:
    if log is None or not log.exists():
        return None
    hit = ROW.search(re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", log.read_text(errors="replace")))
    return float(hit.group(1)) * 100 if hit else None


def newest(pattern: str) -> Path | None:
    hits = sorted(LSTM.glob(pattern))
    return hits[-1] if hits else None


def spread_labels(axis, items: list[tuple[float, float, str]]) -> None:
    """Right-of-point labels at (x, y), moved apart vertically so no two overlap (min gap = half the summed line
    heights), keeping the group centred on the data. Call after the axis limits are final."""
    fig = axis.figure
    to_pt = 72.0 / fig.dpi
    rows = sorted(((axis.transData.transform((x, y))[1] * to_pt, x, y, t) for x, y, t in items))
    placed: list[float] = []
    for i, (p, *_rest) in enumerate(rows):
        if placed:
            need = LINE_PT * (rows[i - 1][3].count("\n") + 1 + _rest[2].count("\n") + 1) / 2
            p = max(p, placed[-1] + need)
        placed.append(p)
    shift = sum(r[0] for r in rows) / len(rows) - sum(placed) / len(placed)
    for (p0, x, y, text), p in zip(rows, placed):
        axis.annotate(text, (x, y), xytext=(5, p + shift - p0), textcoords="offset points", va="center",
                      fontsize=7.5, color=rs.INK)


def main() -> None:
    rs.apply()
    test = json.loads((rs.DATA / "test_metrics.json").read_text())
    spiking = [s for s in SPIKING if ap(S9 / f"sweep_tr/{s[0]}_dt5_nc.log") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(rs.TEXT_WIDTH, 2.8), gridspec_kw={"wspace": 0.25,
                                                                               "width_ratios": [1, 1.25]})
    ax, bx = axes
    rows = []

    for prefix, style in MODELS:                                                     # (a) regime 1
        aps = [ap(S9 / f"sweep/{prefix}_dt{dt}.log") for dt, _ in WINDOWS]
        ref = aps[2]
        ret = [a / ref * 100 for a in aps]
        ax.plot([m for _, m in WINDOWS], ret, color=rs.colour(style), marker=rs.marker(style),
                markeredgecolor="white", markeredgewidth=0.6, label=rs.label(style).replace(" (baseline)", ""))
        rows += [{"regime": 1, "model": style, "mult": m, "dt_ms": dt, "comp": "none", "AP": round(a, 2),
                  "retention_pct": round(r, 1)} for (dt, m), a, r in zip(WINDOWS, aps, ret)]
    ax.set_xscale("log", base=2)
    ax.set_xticks([m for _, m in WINDOWS], ["0.25×\n200 ms", "0.5×", "1×\n50 ms", "2×", "4×\n12 ms"])
    ax.set_xlabel("Accumulation window (rate multiplier)")
    ax.set_ylabel("Retention (% of own 1× AP)")
    ax.set_ylim(70, 102)
    ax.legend(loc="lower center", fontsize=7.5)
    rs.panel_tag(ax, "a", "fixed cadence")

    labels_b = []
    xs = [1.0] + [m for _, m, _ in TRUE_RATE]
    for prefix, style in MODELS:                                                     # (b) regime 2
        ref = ap(S9 / f"sweep/{prefix}_dt50.log")
        for comp, ls, mfc in (("none", "-", None), ("dt", "--", "white")):
            aps = [ref] + [ap(S9 / f"sweep_tr/{prefix}_dt{dt}_{'nc' if comp == 'none' else 'ss' + ss}.log")
                           for dt, _, ss in TRUE_RATE]
            ret = [a / ref * 100 for a in aps]
            bx.plot(xs, ret, ls, color=rs.colour(style), marker=rs.marker(style), markerfacecolor=mfc,
                    markeredgecolor=rs.colour(style) if mfc else "white", markeredgewidth=0.8)
            if comp == "none":
                labels_b.append((xs[-1], ret[-1], f"{ret[-1]:.1f} %"))
            rows += [{"regime": 2, "model": style, "mult": x, "dt_ms": round(50 / x, 1), "comp": comp,
                      "AP": round(a, 2), "retention_pct": round(r, 1)} for x, a, r in zip(xs, aps, ret)]
    lstm = [ap(newest("rvt_lstm_gen1_*.log")), ap(newest("rvt_lstm_preproc_tr_dt=25_*.log")),
            ap(newest("rvt_lstm_preproc_tr_dt=5_*.log"))]
    if None not in lstm:                                                             # measured ConvLSTM, same test set
        ret = [a / lstm[0] * 100 for a in lstm]
        bx.plot(xs, ret, "-", color=rs.colour("convlstm"), marker=rs.marker("convlstm"), markeredgecolor="white",
                markeredgewidth=0.6, label="RVT ConvLSTM, measured")
        labels_b.append((xs[-1], ret[-1], f"ConvLSTM {ret[-1]:.1f} %"))
        rows += [{"regime": 2, "model": "convlstm", "mult": x, "dt_ms": round(50 / x, 1), "comp": "none",
                  "AP": round(a, 2), "retention_pct": round(r, 1)} for x, a, r in zip(xs, lstm, ret)]
    comp10 = [r["retention_pct"] for r in rows if r["regime"] == 2 and r["comp"] == "dt" and r["mult"] == 10]
    labels_b.append((10, sum(comp10) / len(comp10), f"{min(comp10):.1f}–{max(comp10):.1f} %\nwith comp."))
    for name, r in PUBLISHED.items():
        bx.plot(10, r, "*", color=rs.INK_2, markersize=8, markerfacecolor="white", markeredgewidth=0.9)
        labels_b.append((10, r, f"published {name}\n{r:.1f} %"))
        rows.append({"regime": 2, "model": f"{name} (Zubic et al.)", "mult": 10, "dt_ms": 5, "comp": "published",
                     "AP": "", "retention_pct": round(r, 1)})
    bx.plot([], [], "-", color=rs.INK_2, label="no compensation")
    bx.plot([], [], "--", color=rs.INK_2, label="Δt compensation")
    bx.set_xscale("log")
    bx.set_xticks([1, 2, 10], ["1×\n50 ms", "2×\n25 ms", "10×\n5 ms"])
    bx.set_xlim(0.85, 70)
    bx.set_ylim(10, 102)
    bx.set_xlabel("True event rate (rate multiplier)")
    bx.legend(loc="lower left", fontsize=7.5)
    rs.panel_tag(bx, "b", "true rate change")

    for a in axes:
        a.minorticks_off()
    fig.canvas.draw()                                                                # final transforms for the labels
    spread_labels(bx, labels_b)
    rs.save(fig, "fig_robustness", rows)

    if spiking:                                                                      # second figure: spiking ladder
        from matplotlib.lines import Line2D
        fig, cx = plt.subplots(figsize=(0.75 * rs.TEXT_WIDTH, 2.7))
        cx.axhline(100, color=rs.AXIS, linewidth=0.9, zorder=1)
        srows, by_rate = [], {2.0: [], 10.0: []}
        for key, tkey, style, arm, sched in spiking:
            ref = test[tkey]["AP"]
            aps = [ref] + [ap(S9 / f"sweep_tr/{key}_dt{dt}_nc.log") for dt, _, _ in TRUE_RATE]
            ret = [a / ref * 100 for a in aps]
            long, c = sched == "400k", rs.colour(style)
            paired = sum(1 for s in spiking if s[3] == arm) == 2                    # arms with a 400k run
            x = arm + (PAIR if long else -PAIR) * paired                              # 100k left, 400k right
            cx.plot([x, x], ret[1:], "--" if long else "-", color=c, linewidth=1.1 if long else 1.6, zorder=2)
            for (_, mult, _), r, ms in zip(TRUE_RATE, ret[1:], (4.0, 6.5)):           # small = 2x, large = 10x
                cx.plot(x, r, rs.marker(style), ms=ms, color=c, markerfacecolor="white" if long else c,
                        markeredgecolor=c if long else "white", markeredgewidth=0.9 if long else 0.6, zorder=3)
                cx.annotate(f"{r:.1f}", (x, r), xytext=(5, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=7, color=rs.INK)
                by_rate[mult].append(r)
            srows += [{"model": tkey, "schedule": sched, "mult": x_, "dt_ms": round(50 / x_, 1), "comp": "none",
                       "AP": round(a, 2), "retention_pct": round(r, 1)} for x_, a, r in zip(xs, aps, ret)]
        right = len(ARM_TICKS) - 1 + 0.55
        cx.text(right, 100.4, "1×  50 ms", ha="left", va="bottom", fontsize=7, color=rs.MUTED)
        for (_, mult, _), text in zip(TRUE_RATE, ("2×  25 ms", "10×  5 ms")):
            cx.text(right, sum(by_rate[mult]) / len(by_rate[mult]), text, ha="left", va="center", fontsize=7,
                    color=rs.MUTED)
        cx.legend(handles=[Line2D([], [], color=rs.INK_2, marker="o", ms=5, markeredgecolor="white",
                                  linewidth=1.6, label="100k-step schedule"),
                           Line2D([], [], color=rs.INK_2, marker="o", ms=5, markerfacecolor="white",
                                  linestyle="--", linewidth=1.1, label="400k-step schedule")],
                  loc="lower center", ncol=2, fontsize=7.5)
        cx.set_xticks(range(len(ARM_TICKS)), ARM_TICKS)
        cx.set_xlim(-0.55, len(ARM_TICKS) - 1 + 1.25)
        cx.set_ylim(50, 103)
        cx.set_ylabel("Retention (% of own 1× AP)")
        cx.grid(axis="x", visible=False)
        cx.minorticks_off()
        rs.save(fig, "fig_robustness_spiking", srows)

if __name__ == "__main__":
    main()
