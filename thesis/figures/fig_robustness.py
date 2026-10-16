#!/usr/bin/env python
"""Report figure: event-rate robustness, as retention = AP at a rate / the model's own AP at the training rate (1x).

(a) Regime 1, fixed 50-ms step cadence, accumulation window varied (0.25x = 200 ms ... 4x = 12.5 ms). Message: every
    model peaks at its training window and the three degrade alike; PureSSM holds best at the dense end.
(b) Regime 2, true rate change (step = window, gap-free; 2x = 25 ms, 10x = 5 ms), without (solid) and with (dashed)
    inference-time Delta-t compensation, plus the two published reference points of Zubic et al. Message: without
    compensation the SSM recurrence keeps 62-70 % at 10x; the compensation costs 9-13 points at 10x for all three
    models; the published 39.84 is not reached under the published artefacts.
(c) Only when the Stage-22 true-rate logs of the spiking arms exist: the same uncompensated protocol for the spiking
    arms, with PureSSM as the reference line. Their 1x reference is the canonical Stage-21 test AP.
Values are read from the eval logs of results/stage9/{sweep,sweep_tr}/ (Tables tab:regime1, tab:regime2, tab:dt_comp).

  python thesis/figures/fig_robustness.py   -> thesis/latex/images/fig_robustness.{pdf,png} + results/report_figs/*.csv
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import report_style as rs
from report_style import plt

S9 = rs.REPO / "results/stage9"
MODELS = [("baseline", "s5rvt"), ("eventssm", "eventssm"), ("puressm", "puressm")]      # log prefix, style key
LABEL_DY = {"s5rvt": -4.5, "eventssm": 4.5, "puressm": 0.0}       # pt: S5-RVT and EventSSM end within 1 % at 10x
WINDOWS = [(200, 0.25), (100, 0.5), (50, 1.0), (25, 2.0), (12, 4.0)]                     # regime 1: dt ms, mult
TRUE_RATE = [(25, 2.0, "0.5000"), (5, 10.0, "0.1000")]                                   # regime 2: dt, mult, step scale
PUBLISHED = {"S5 + comp. (Zubic et al.)": 39.84 / 47.7 * 100, "ConvLSTM (Zubic et al.)": 8.35 / 47.2 * 100}
SPIKING = [("graded_s234_100k", "graded", "Graded 100k"), ("spike_s234_100k", "spike", "Spike 100k"),
           ("graded_s234_400k", "graded", "Graded 400k")]
ROW = re.compile(r"│\s*test/AP\s*│\s*([0-9.]+)\s*│")


def ap(log: Path) -> float | None:
    if not log.exists():
        return None
    hit = ROW.search(re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", log.read_text(errors="replace")))
    return float(hit.group(1)) * 100 if hit else None


def end_label(axis, x, y, text, dy=0.0):
    axis.annotate(text, (x, y), xytext=(5, dy), textcoords="offset points", va="center", fontsize=7.5, color=rs.INK)


def main() -> None:
    rs.apply()
    spiking = [(k, s, n) for k, s, n in SPIKING if ap(S9 / f"sweep_tr/{k}_dt5_nc.log") is not None]
    ncols = 3 if spiking else 2
    fig, axes = plt.subplots(1, ncols, figsize=(rs.TEXT_WIDTH, 2.6), gridspec_kw={"wspace": 0.3})
    ax, bx = axes[0], axes[1]
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
    ax.set_xticks([m for _, m in WINDOWS], ["0.25×\n200 ms", "0.5×", "1×\n50 ms", "2×", "4×\n12.5 ms"])
    ax.set_xlabel("Accumulation window (rate multiplier)")
    ax.set_ylabel("Retention (% of own 1× AP)")
    ax.set_ylim(70, 102)
    ax.legend(loc="lower center", fontsize=7.5)
    rs.panel_tag(ax, "a", "fixed cadence")

    for prefix, style in MODELS:                                                     # (b) regime 2
        ref = ap(S9 / f"sweep/{prefix}_dt50.log")
        for comp, ls, mfc in (("none", "-", None), ("dt", "--", "white")):
            aps = [ref] + [ap(S9 / f"sweep_tr/{prefix}_dt{dt}_{'nc' if comp == 'none' else 'ss' + ss}.log")
                           for dt, _, ss in TRUE_RATE]
            ret = [a / ref * 100 for a in aps]
            xs = [1.0] + [m for _, m, _ in TRUE_RATE]
            bx.plot(xs, ret, ls, color=rs.colour(style), marker=rs.marker(style), markerfacecolor=mfc,
                    markeredgecolor=rs.colour(style) if mfc else "white", markeredgewidth=0.8)
            if comp == "none":
                end_label(bx, xs[-1], ret[-1], f"{ret[-1]:.1f} %", LABEL_DY[style])
            rows += [{"regime": 2, "model": style, "mult": x, "dt_ms": round(50 / x, 1), "comp": comp,
                      "AP": round(a, 2), "retention_pct": round(r, 1)} for x, a, r in zip(xs, aps, ret)]
    comp10 = [r["retention_pct"] for r in rows if r["regime"] == 2 and r["comp"] == "dt" and r["mult"] == 10]
    end_label(bx, 10, sum(comp10) / len(comp10), f"{min(comp10):.1f}–{max(comp10):.1f} %\nwith comp.")
    for name, r in PUBLISHED.items():
        bx.plot(10, r, "*", color=rs.INK_2, markersize=8, markerfacecolor="white", markeredgewidth=0.9)
        end_label(bx, 10, r, f"{name.split(' (')[0]}, published\n{r:.1f} %")
        rows.append({"regime": 2, "model": name, "mult": 10, "dt_ms": 5, "comp": "published", "AP": "",
                     "retention_pct": round(r, 1)})
    bx.plot([], [], "-", color=rs.INK_2, label="no compensation")
    bx.plot([], [], "--", color=rs.INK_2, label="Δt compensation")
    bx.set_xscale("log")
    bx.set_xticks([1, 2, 10], ["1×\n50 ms", "2×\n25 ms", "10×\n5 ms"])
    bx.set_xlim(0.85, 30)
    bx.set_ylim(10, 102)
    bx.set_xlabel("True event rate (rate multiplier)")
    bx.legend(loc="lower left", fontsize=7.5)
    rs.panel_tag(bx, "b", "true rate change")

    if spiking:                                                                      # (c) spiking arms
        cx = axes[2]
        test = json.loads((rs.DATA / "test_metrics.json").read_text())
        pure = [r["retention_pct"] for r in rows if r["model"] == "puressm" and r["regime"] == 2
                and r["comp"] == "none"]
        cx.plot([1, 2, 10], pure, "-", color=rs.colour("puressm"), linewidth=1, alpha=0.6, label="PureSSM 400k")
        for key, style, name in spiking:
            ref = test[key.replace("_s234", "")]["AP"]
            aps = [ref] + [ap(S9 / f"sweep_tr/{key}_dt{dt}_nc.log") for dt, _, _ in TRUE_RATE]
            ret = [a / ref * 100 for a in aps]
            cx.plot([1, 2, 10], ret, "--" if "400k" in name else "-", color=rs.colour(style),
                    marker=rs.marker(style), markeredgecolor="white", markeredgewidth=0.6, label=name)
            end_label(cx, 10, ret[-1], f"{ret[-1]:.1f} %")
            rows += [{"regime": 2, "model": key, "mult": x, "dt_ms": round(50 / x, 1), "comp": "none",
                      "AP": round(a, 2), "retention_pct": round(r, 1)} for x, a, r in zip([1, 2, 10], aps, ret)]
        cx.set_xscale("log")
        cx.set_xticks([1, 2, 10], ["1×", "2×", "10×"])
        cx.set_xlim(0.85, 30)
        cx.set_ylim(40, 102)
        cx.set_xlabel("True event rate")
        cx.legend(loc="lower left", fontsize=7)
        rs.panel_tag(cx, "c", "spiking arms")
    for a in axes:
        a.minorticks_off()
    rs.save(fig, "fig_robustness", rows)


if __name__ == "__main__":
    main()
