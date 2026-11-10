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
    uncompensated protocol for the 100k ladder (PureSSM 100k anchor, graded, spike; same schedule = the fair
    comparison; solid) and the 400k graded run against PureSSM 400k (dashed). Each arm's 1x reference is its
    canonical Stage-21 test AP. Separate figure because three panels do not fit the text width legibly.
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
# (sweep_tr log key, test_metrics key, style, legend name, dashed)
SPIKING = [("puressm_100k", "puressm_100k", "puressm", "PureSSM 100k", False),
           ("graded_s234_100k", "graded_100k", "graded", "Graded 100k", False),
           ("spike_s234_100k", "spike_100k", "spike", "Spike 100k", False),
           ("puressm", "puressm", "puressm", "PureSSM 400k", True),
           ("graded_s234_400k", "graded_400k", "graded", "Graded 400k", True)]
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

    if spiking:                                                                      # second figure: spiking arms
        fig, cx = plt.subplots(figsize=(0.62 * rs.TEXT_WIDTH, 2.6))
        labels_c, srows = [], []
        for key, tkey, style, name, dashed in spiking:
            ref = test[tkey]["AP"]
            aps = [ref] + [ap(S9 / f"sweep_tr/{key}_dt{dt}_nc.log") for dt, _, _ in TRUE_RATE]
            ret = [a / ref * 100 for a in aps]
            cx.plot(xs, ret, "--" if dashed else "-", color=rs.colour(style), marker=rs.marker(style),
                    markerfacecolor="white" if dashed else None,
                    markeredgecolor=rs.colour(style) if dashed else "white", markeredgewidth=0.8,
                    linewidth=1.2 if dashed else 1.8)
            labels_c.append((xs[-1], ret[-1], f"{name}  {ret[-1]:.1f} %"))
            srows += [{"model": tkey, "mult": x, "dt_ms": round(50 / x, 1), "comp": "none", "AP": round(a, 2),
                       "retention_pct": round(r, 1)} for x, a, r in zip(xs, aps, ret)]
        cx.plot([], [], "-", color=rs.INK_2, label="100k-step schedule")
        cx.plot([], [], "--", color=rs.INK_2, linewidth=1.2, label="400k-step schedule")
        cx.set_xscale("log")
        cx.set_xticks([1, 2, 10], ["1×\n50 ms", "2×\n25 ms", "10×\n5 ms"])
        cx.set_xlim(0.85, 60)
        cx.set_ylim(55, 102)
        cx.set_xlabel("True event rate (rate multiplier)")
        cx.set_ylabel("Retention (% of own 1× AP)")
        cx.legend(loc="lower left", fontsize=7.5)
        cx.minorticks_off()
        fig.canvas.draw()
        spread_labels(cx, labels_c)
        rs.save(fig, "fig_robustness_spiking", srows)


if __name__ == "__main__":
    main()
