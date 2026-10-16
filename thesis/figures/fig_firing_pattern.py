#!/usr/bin/env python
"""Report figure: firing rate of the 100k-step graded and spike readouts over one test recording, against the
input activity (Stage 22 item 5; the report-style redraw of `event_ssm.eval.firing_pattern`'s figure).

Message: the readout's sparsity is structural, not event-driven. Only graded stage 2 follows the events per frame,
and even there its rate moves by a few percentage points; every other stage is flat, and the spike readout fires at
two to four times the graded rate at every stage.

Layout (D1-D3 of docs/notes/Report_figures_notes.md): colour = arm, so the stages are separate panels rather than
colours; (a) events per frame in a neutral fill; (b)-(d) stages 2/3/4 on one shared rate axis, so the flatness and
the graded-vs-spike gap are read on the same scale. The first 32 frames (1.6 s, the recurrent state warming up from
zero, as in the SOP benchmark's warm-up) are shaded and excluded from the correlations and ranges.

Input: results/stage22/firing/{graded,spike}_s234_<rec>.json (written on the GPU by
code/event_ssm/scripts/stage22_firing_pattern.py; per frame t_s, events, rates[stage]).

  python thesis/figures/fig_firing_pattern.py
  -> thesis/latex/images/fig_firing_pattern.{pdf,png} + results/report_figs/fig_firing_pattern.csv
"""
from __future__ import annotations

import json

import numpy as np

import report_style as rs
from report_style import plt

REC = "18-03-29_13-15-02_122500000_182500000"
ARMS = [("graded", "Graded"), ("spike", "Spike")]        # style key, legend label (100k checkpoints, stages 2-4)
STAGES = ("2", "3", "4")
WARMUP = 32                                              # frames; the SOP benchmark's warm-up (benchmark/sop.py)
EVENTS_FILL, EVENTS_EDGE = "#dcdad3", rs.MUTED           # neutral: the input is not a model


def load(arm: str) -> dict:
    d = json.loads((rs.REPO / f"results/stage22/firing/{arm}_s234_{REC}.json").read_text())
    rec = d["records"]
    return {"t": np.array([r["t_s"] for r in rec]), "events": np.array([r["events"] for r in rec]),
            "rates": {s: np.array([r["rates"][s] for r in rec]) for s in STAGES}}


def main() -> None:
    rs.apply()
    data = {arm: load(arm) for arm, _ in ARMS}
    t, events = data["graded"]["t"], data["graded"]["events"]
    assert all(np.array_equal(data[a]["events"], events) for a, _ in ARMS), "arms must share the recording"
    warm_end = t[WARMUP]

    fig, axes = plt.subplots(4, 1, figsize=(rs.TEXT_WIDTH, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1, 1, 1], "hspace": 0.42})
    ev_ax, stage_axes = axes[0], axes[1:]
    ev_ax.fill_between(t, events / 1e3, color=EVENTS_FILL, linewidth=0)
    ev_ax.plot(t, events / 1e3, color=EVENTS_EDGE, linewidth=0.8)
    ev_ax.set_ylabel("Events per\nframe (k)")
    ev_ax.set_ylim(0, None)
    rs.panel_tag(ev_ax, "a", "input activity")

    rows, stats = [], {}
    for ax, stage, tag in zip(stage_axes, STAGES, "bcd"):
        for arm, name in ARMS:
            rate = data[arm]["rates"][stage]
            ax.plot(t, rate, color=rs.colour(arm), linewidth=1.0, label=name)
            r = float(np.corrcoef(rate[WARMUP:], events[WARMUP:])[0, 1])
            lo, hi = float(rate[WARMUP:].min()), float(rate[WARMUP:].max())
            stats[(arm, stage)] = (r, lo, hi, float(rate[WARMUP:].mean()))
            # direct label at the line's end, in ink: the arm is carried by the colour of the line beside it
            ax.annotate(f"{name.lower()}  r = {rs.signed(r, 2)}", (t[-1], rate[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=7.5, color=rs.INK)
            rows += [{"arm": arm, "stage": stage, "t_s": round(float(ti), 3), "events": int(e),
                      "rate": round(float(v), 5), "warmup": i < WARMUP}
                     for i, (ti, e, v) in enumerate(zip(t, events, rate))]
        ax.set_ylim(0, 0.55)
        ax.set_yticks([0, 0.25, 0.5])
        ax.set_ylabel("Nonzero\nrate")
        rs.panel_tag(ax, tag, f"stage {stage}")
    for ax in axes:
        ax.axvspan(t[0], warm_end, color=rs.GRID, alpha=0.6, linewidth=0, zorder=0)
        ax.grid(axis="x", visible=False)
    ev_ax.text(warm_end, ev_ax.get_ylim()[1], " warm-up", ha="left", va="top", fontsize=7, color=rs.INK_2)
    stage_axes[-1].set_xlabel("Time in test recording (s)")
    stage_axes[-1].set_xlim(t[0], t[-1])
    fig.subplots_adjust(top=0.92)
    fig.legend(*stage_axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))

    for (arm, stage), (r, lo, hi, mean) in stats.items():
        print(f"{arm:6s} stage {stage}: mean {mean:.3f}  range {lo:.3f}-{hi:.3f}  r {r:+.3f}  (after {WARMUP} frames)")
    rs.save(fig, "fig_firing_pattern", rows)


if __name__ == "__main__":
    main()
