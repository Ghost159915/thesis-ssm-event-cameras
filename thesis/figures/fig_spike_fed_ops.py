#!/usr/bin/env python
"""Report figure: how much of the detector's arithmetic the spikes can reach (Stage 22 SOP accounting).

Message: the spiking readout feeds only the first 1x1 convolutions of the neck, 52.4 M of the 5.00 G MACs per
frame (1.05 %), so even a silent readout on zero-skipping hardware could save at most about 1 %; at the measured
nonzero rates the readouts drive only part of that 52.4 M, and the estimated saving is smaller still.

(a) one bar of the dense MACs per frame, split into the part no spike reaches and the spike-fed part;
(b) a zoom on the spike-fed MACs: per arm, the share driven at its measured nonzero rate (arm colour) and the share
    left idle (neutral), with the estimated energy saving. Rows appear for every per-arm benchmark that exists.

Input: the `sop` block of a SpikingSSM entry in a Stage-22 bench_results.json (code/event_ssm/benchmark/sop.py;
rates over 16 test clips after a 32-frame warm-up; Horowitz 45-nm constants; operation counts only, memory traffic
not modelled).

  python thesis/figures/fig_spike_fed_ops.py
  -> thesis/latex/images/fig_spike_fed_ops.{pdf,png} + results/report_figs/fig_spike_fed_ops.csv
"""
from __future__ import annotations

import json

import report_style as rs
from report_style import plt

S22 = rs.REPO / "results/stage22"
ARMS = [  # row label, style key, benchmark dir (rows whose JSON does not exist yet are skipped)
    ("Graded 400k", "graded", S22 / "graded_400k_with_ann"),
    ("Analog 100k", "analog", S22 / "analog_100k"),
    ("Graded 100k", "graded", S22 / "graded_100k"),
    ("Spike 100k", "spike", S22 / "spike_100k"),
]
DENSE, IDLE, FED = "#b4b2a9", "#dcdad3", rs.INK          # neutral fills: op classes, not models


def sop_of(folder):
    path = folder / "bench_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["models"]["spikingssm"]["sop"]


def main() -> None:
    rs.apply()
    arms = [(name, style, sop) for name, style, d in ARMS if (sop := sop_of(d)) is not None]
    assert arms, "no Stage-22 SOP result found"
    ref = arms[0][2]
    total, fed = ref["dense"]["total_macs"], ref["spike_fed_total"]
    for _, _, sop in arms:                                 # every arm shares the architecture, hence the counts
        assert sop["dense"]["total_macs"] == total and sop["spike_fed_total"] == fed

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(rs.TEXT_WIDTH, 1.7 + 0.3 * len(arms)),
                                 gridspec_kw={"height_ratios": [0.9, 0.35 + 0.4 * len(arms)],
                                              "hspace": 2.4 / (1.25 + 0.4 * len(arms))})   # ~constant gap
    # (a) the whole frame, in G MACs
    ax.barh(0, (total - fed) / 1e9, height=0.6, color=DENSE, edgecolor="white", linewidth=1)
    ax.barh(0, fed / 1e9, left=(total - fed) / 1e9, height=0.6, color=FED, edgecolor="white", linewidth=0.3)
    ax.text((total - fed) / 2e9, 0, f"never spike-fed: {(total - fed) / 1e9:.2f} G ({(1 - fed / total) * 100:.2f} %)",
            ha="center", va="center", fontsize=8, color=rs.INK)
    ax.annotate(f"spike-fed: {fed / 1e6:.1f} M ({fed / total * 100:.2f} %)", xy=((total - fed / 2) / 1e9, 0.32),
                xytext=((total - fed / 2) / 1e9, 1.0), ha="right", va="bottom", fontsize=8, color=rs.INK,
                arrowprops={"arrowstyle": "-", "color": rs.INK, "linewidth": 0.8, "relpos": (1, 0),
                            "shrinkA": 0, "shrinkB": 0})
    ax.set_xlim(0, total / 1e9 * 1.0)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Dense MACs per frame (G)")
    rs.panel_tag(ax, "a", "the whole detector")

    # (b) zoom on the spike-fed MACs, in M MACs
    rows = []
    for y, (name, style, sop) in enumerate(arms):
        driven, saving, ceiling = sop["driven_ops"], sop["saving"], sop["ceiling"]
        op = "AC" if sop["op_class"] == "AC" else "MAC"
        bx.barh(y, driven / 1e6, height=0.6, color=rs.colour(style), edgecolor="white", linewidth=1)
        bx.barh(y, (fed - driven) / 1e6, left=driven / 1e6, height=0.6, color=IDLE, edgecolor="white", linewidth=1)
        inside = driven / fed > 0.6                        # a dense readout: label inside its own bar instead
        bx.text(0.8 if inside else driven / 1e6 + 0.8, y,
                f"{driven / 1e6:.1f} M driven ({driven / fed * 100:.0f} %, priced as {op})",
                ha="left", va="center", fontsize=7.5, color=rs.INK)
        bx.text(fed / 1e6 + 0.8, y, f"saving {saving * 100:.2f} %", ha="left", va="center", fontsize=7.5,
                color=rs.INK)
        rates = sop["rates"]["stages"]
        rows.append({"arm": name, "op_class": op, "total_macs": total, "spike_fed_macs": fed,
                     "driven_ops": round(driven), "energy_mj": round(sop["energy_j"] * 1e3, 3),
                     "energy_dense_mj": round(sop["energy_dense_j"] * 1e3, 3), "saving_pct": round(saving * 100, 3),
                     "ceiling_pct": round(ceiling * 100, 3),
                     **{f"rate_s{s}": round(v["rate"], 4) for s, v in sorted(rates.items())}})
    bx.set_yticks(range(len(arms)), [n for n, _, _ in arms])
    bx.set_ylim(len(arms) - 0.5, -0.5)
    bx.set_xlim(0, fed / 1e6)
    bx.grid(axis="y", visible=False)
    bx.set_xlabel("Spike-fed MACs per frame (M)")
    rs.panel_tag(bx, "b", f"zoom on the spike-fed {fed / 1e6:.1f} M; ceiling {ref['ceiling'] * 100:.2f} %")
    for name, *_ in arms:
        print(f"{name}: driven {[r for r in rows if r['arm'] == name][0]['driven_ops'] / 1e6:.2f} M")
    rs.save(fig, "fig_spike_fed_ops", rows)


if __name__ == "__main__":
    main()
