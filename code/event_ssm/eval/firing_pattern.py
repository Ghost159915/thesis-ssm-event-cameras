"""Firing activity against input activity over one test recording (item 5 of
docs/plans/2026-10-11-post-training-spiking-analysis.md). Torch-free: summary and figure only.

The GPU side (scripts/stage22_firing_pattern.py) streams a recording frame by frame through a trained SpikingSSM
checkpoint and writes one record per 50-ms frame:
    {"t_s": time from the recording start, "events": event count in the frame (sum of the stacked histogram),
     "active_px": pixels with at least one event, "rates": {"2": r2, "3": r3, "4": r4}}
where r_s is the share of nonzero outputs of stage s's readout in that frame. If firing tracks scene activity,
the rates rise and fall with the event count, which is visual evidence of event-driven behaviour.

Usage:
  python -m event_ssm.eval.firing_pattern --records results/stage22/firing/<tag>_<rec>.json --out <stem>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]       # reference palette slots 1-3 (validated all-pairs)
INK, INK_2, RULE, AREA = "#0b0b0b", "#52514e", "#d8d7d2", "#8a8984"


def stages_of(records: List[dict]) -> List[str]:
    return sorted(records[0]["rates"], key=int)


def summarise(records: List[dict]) -> dict:
    """Per stage: mean and spread of the per-frame rate and its Pearson correlation with the event count."""
    if len(records) < 3:
        raise ValueError("need at least 3 frames")
    ev = np.array([r["events"] for r in records], dtype=float)
    out = {"frames": len(records), "events_mean": float(ev.mean()), "stages": {}}
    for s in stages_of(records):
        rate = np.array([r["rates"][s] for r in records], dtype=float)
        corr = float(np.corrcoef(ev, rate)[0, 1]) if ev.std() > 0 and rate.std() > 0 else float("nan")
        out["stages"][s] = {"rate_mean": float(rate.mean()), "rate_min": float(rate.min()),
                            "rate_max": float(rate.max()), "corr_with_events": corr}
    return out


def plot(records: List[dict], title: Optional[str] = None):
    """Two stacked panels on one time axis (each with its own y-axis): input events per frame, and the firing
    rate per stage. Never a dual-axis chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r["t_s"] for r in records])
    ev = np.array([r["events"] for r in records], dtype=float)
    stages = stages_of(records)
    if len(stages) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} stages")
    fig, (ax_e, ax_r) = plt.subplots(2, 1, figsize=(9.5, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    ax_e.fill_between(t, ev / 1e3, color=AREA, alpha=0.35, linewidth=0)
    ax_e.plot(t, ev / 1e3, color=AREA, linewidth=1.2)
    ax_e.set_ylabel("Events per frame\n(thousands)", fontsize=10, color=INK)
    summ = summarise(records)
    for s, color in zip(stages, SERIES):
        rate = np.array([r["rates"][s] for r in records], dtype=float)
        r_corr = summ["stages"][s]["corr_with_events"]
        ax_r.plot(t, 100 * rate, color=color, linewidth=1.6, label=f"stage {s} (r = {r_corr:.2f})")
        ax_r.annotate(f"stage {s}", (t[-1], 100 * rate[-1]), xytext=(6, 0), textcoords="offset points",
                      va="center", fontsize=9.5, color=INK)
    ax_r.set_ylabel("Readout firing rate (%)", fontsize=10, color=INK)
    ax_r.set_xlabel("Time in recording (s)", fontsize=10, color=INK)
    ax_r.legend(frameon=False, fontsize=9, loc="upper left", title="r = correlation with events",
                title_fontsize=8.5)
    for ax in (ax_e, ax_r):
        ax.grid(color=RULE, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)
        ax.tick_params(colors=INK_2, labelsize=9)
    ax_r.margins(x=0.08)
    if title:
        ax_e.set_title(title, fontsize=12, color=INK, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Summarise and plot firing activity against input activity.")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output stem: .png, .svg, _summary.json")
    p.add_argument("--title", default=None)
    a = p.parse_args(argv)
    d = json.loads(a.records.read_text())
    records = d["records"] if isinstance(d, dict) else d
    summ = summarise(records)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    fig = plot(records, a.title)
    for ext in (".png", ".svg"):
        fig.savefig(a.out.with_suffix(ext), dpi=300, bbox_inches="tight", facecolor="white")
    return 0


if __name__ == "__main__":
    sys.exit(main())
