#!/usr/bin/env python
"""Report figure (Appendix, Project Timeline): what was actually done, week by week — the companion of fig_gantt
(the Thesis-A plan).

Evidence, not recollection:
* Work-stream rows: a week is filled when at least one commit of that stream was authored in it. Every commit
  subject in `git log` is mapped to a stream by the ordered rules in STREAM_RULES (first match wins; housekeeping
  commits match nothing and are dropped). The mapping of every commit is written to
  results/report_figs/fig_gantt_actual_commits.csv so it can be audited.
* GPU-training row: the exact span of every training run, from the W&B offline run folders (folder timestamp =
  start, newest file = end; external/ssms_event_cameras/RVT/wandb, 2026-10-16) and, for the two cloud PureSSM runs,
  the day-level dates in docs/notes/Stage13_cloud_notes.md and docs/roadmap/Stage_14_PureSSM_Full_Training.md.
* Milestones: dated results recorded in CLAUDE.md and the stage notes.
Not captured: work outside the repository (reading, meetings, Thesis-A work before the first commit on
2026-04-15). Only the Thesis-C term (T3, 14 Sep - 20 Nov 2026) is shaded; the other term dates are not recorded.

  python thesis/figures/fig_gantt_actual.py
  -> thesis/latex/images/fig_gantt_actual.{pdf,png} + results/report_figs/fig_gantt_actual{,_commits}.csv
"""
from __future__ import annotations

import csv
import re
import subprocess
from datetime import date, datetime, timedelta

import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

import report_style as rs
from report_style import plt

NEUTRAL = "#4a4945"                      # cross-cutting work (not one model): dark neutral, distinct from S5-RVT grey
STREAMS = [  # key, label, colour
    ("mvp", "S5-RVT reproduction (MVP)", rs.colour("s5rvt")),
    ("eventssm", "EventSSM: build, train, test", rs.colour("eventssm")),
    ("temporal", "Temporal-generalisation study", NEUTRAL),
    ("efficiency", "Efficiency benchmark", NEUTRAL),
    ("puressm", "PureSSM: build, cloud training, test", rs.colour("puressm")),
    ("spike_build", "SpikingSSM: build, smoke, pilots", rs.colour("graded")),
    ("spike_train", "SpikingSSM: training ladder, analysis", rs.colour("graded")),
    ("research", "Research and planning", NEUTRAL),
    ("inrc", "Neuromorphic track (INRC proposal)", NEUTRAL),
    ("writing", "Report writing, supervisor reports", NEUTRAL),
]
STREAM_RULES = [  # (stream, regex on the commit subject), first match wins; "stage[- ]?N" covers stage9/Stage-9/Stage 9
    (None, r"^chore|^refactor\(repo\)"),                       # housekeeping: dropped
    ("inrc", r"\binrc\b|loihi"),
    ("writing", r"^thesis:|\(thesis\)|report-figs|\(report\)|fact-check|presentation|supervisor|\(admin\)|"
                r"standard report|final report|main\.tex|50 pages|label provenance|per-scale taps"),
    ("spike_train", r"stage[- ]?2[0-2]|graded|ladder|spikingssm_np|post-training"),
    ("spike_build", r"stage[- ]?1[7-9]|spikingssm|spiking firing-rate|\(monitors\)|\(integration\)|kill-switch"),
    ("puressm", r"stage[- ]?1[1-6]|puressm|stage9-plots"),
    ("efficiency", r"stage[- ]?10"),
    ("temporal", r"stage[- ]?9|\(d1\)"),
    ("eventssm", r"stage[- ]?[0-8][a-z]?\b|stage7-8|stage0-3"),
    ("mvp", r"\bMVP\b|baseline reproduced|Initial commit"),
    ("research", r"\(research\)|\(plan\)|\(spec\)|roadmap|Thesis C timeline|SNN target"),
]
GPU_RUNS = [  # (start, end, colour key, what) — W&B offline folders unless marked (cloud, day-level)
    ("2026-06-16 15:36", "2026-06-16 22:09", "eventssm", "Stage 6 short runs"),
    ("2026-06-17 11:34", "2026-06-18 04:45", "eventssm", "Stage 7 mid-run"),
    ("2026-06-18 07:46", "2026-06-19 12:26", "eventssm", "EventSSM 400k"),
    ("2026-07-13 00:00", "2026-07-15 23:59", "puressm", "PureSSM 25k + 400k (cloud RTX 5090, day-level)"),
    ("2026-10-06 21:53", "2026-10-07 21:48", "graded", "Stage 19 25k pilots"),
    ("2026-10-08 22:41", "2026-10-09 19:02", "graded", "spike 100k, graded 100k (stopped)"),
    ("2026-10-10 14:30", "2026-10-10 20:36", "graded", "graded 100k (resumed, stopped)"),
    ("2026-10-11 11:14", "2026-10-12 07:05", "graded", "graded 100k, analog 100k"),
    ("2026-10-12 07:05", "2026-10-12 20:31", "puressm", "PureSSM 100k anchor"),
    ("2026-10-12 20:31", "2026-10-13 23:52", "graded", "graded 400k (stopped)"),
    ("2026-10-14 17:45", "2026-10-15 16:48", "graded", "graded 400k (resumed, stopped)"),
    ("2026-10-15 19:02", "2026-10-15 20:15", "graded", "graded 400k (resumed, stopped)"),
    ("2026-10-16 01:52", "2026-10-16 10:06", "graded", "graded 400k (finished)"),
]
MILESTONES = [  # (date, stream, label, side l/r[, anchor date]): label just outside the week block, in free space
    ("2026-06-05", "mvp", "S5-RVT reproduced (47.7)", "r"),
    ("2026-06-19", "eventssm", "EventSSM test 46.2", "r"),
    ("2026-07-11", "temporal", "Δt rescaling not reproducible", "r"),
    ("2026-07-11", "efficiency", "first efficiency results", "r"),
    ("2026-07-15", "puressm", "PureSSM test 46.4, AP$_L$ +3.0", "r"),
    ("2026-09-07", "research", "Thesis-C plan", "l"),
    ("2026-10-07", "spike_build", "kill-switch passed", "r"),
    ("2026-10-12", "spike_train", "", "r"),
    ("2026-10-16", "spike_train", "100k ladder (12 Oct), 400k (16 Oct)", "l", "2026-10-05"),
    ("2026-07-16", "inrc", "proposal drafted", "r"),
    ("2026-10-11", "inrc", "finalised", "r"),
    ("2026-10-13", "writing", "restructured", "r"),
]
START, END = date(2026, 4, 13), date(2026, 11, 29)       # Monday of the first commit's week .. after T3 teaching
T3 = (date(2026, 9, 14), date(2026, 11, 20))
BAND = "#f1f0eb"                                          # Thesis-C term shading
TODAY = date(2026, 10, 16)


def commits() -> list[tuple[date, str]]:
    out = subprocess.run(["git", "log", "--format=%ad%x09%s", "--date=short"], cwd=rs.REPO, check=True,
                         capture_output=True, text=True).stdout
    return [(date.fromisoformat(d), s) for d, s in (line.split("\t", 1) for line in out.splitlines() if line)]


def classify(subject: str) -> str | None:
    for stream, rule in STREAM_RULES:
        if re.search(rule, subject, flags=re.IGNORECASE):
            return stream                                  # None for housekeeping
    return None


def main() -> None:
    rs.apply()
    row = {k: i for i, (k, _, _) in enumerate(STREAMS)}
    gpu_y = len(STREAMS) + 0.15
    active: dict[str, set[date]] = {k: set() for k, _, _ in STREAMS}
    audit = []
    for d, subject in commits():
        stream = classify(subject)
        audit.append({"date": d.isoformat(), "stream": stream or "", "subject": subject})
        if stream:
            active[stream].add(d - timedelta(days=d.weekday()))          # Monday of the ISO week

    fig, ax = plt.subplots(figsize=(rs.TEXT_WIDTH, 3.9))
    ax.axvspan(T3[0], T3[1], color=BAND, zorder=0)
    ax.text(T3[0] + (T3[1] - T3[0]) / 2, -0.95, "Thesis C (T3)", ha="center", va="bottom", fontsize=8,
            color=rs.INK_2)
    rows = []
    for key, _, colour in STREAMS:
        for monday in sorted(active[key]):
            ax.add_patch(Rectangle((mdates.date2num(monday) + 0.4, row[key] - 0.3), 6.2, 0.6, color=colour,
                                   linewidth=0, zorder=2))
            rows.append({"row": key, "week_monday": monday.isoformat(), "kind": "commit week"})
    for s, e, ck, what in GPU_RUNS:
        s, e = datetime.fromisoformat(s), datetime.fromisoformat(e)
        ax.add_patch(Rectangle((mdates.date2num(s), gpu_y - 0.22), mdates.date2num(e) - mdates.date2num(s), 0.44,
                               color=rs.colour(ck), linewidth=0, zorder=2))
        rows.append({"row": "gpu", "week_monday": s.isoformat(timespec="minutes"), "kind": f"run to "
                     f"{e.isoformat(timespec='minutes')}: {what}"})
    for d, key, text, side, *anchor in MILESTONES:
        day = date.fromisoformat(d)
        x = mdates.date2num(day)
        ax.plot(x, row[key], "D", color=rs.INK, markersize=4.8, markeredgecolor="white", markeredgewidth=0.7,
                zorder=4)
        monday = day - timedelta(days=day.weekday())
        if anchor:
            x = mdates.date2num(date.fromisoformat(anchor[0]))
        elif monday in active[key]:                        # anchor the label at the edge of the filled week
            x = mdates.date2num(monday + timedelta(days=7 if side == "r" else 0))
        if text:                                           # a label running across the today line hides it
            crosses = side == "r" and T3[0] <= mdates.num2date(x).date() < TODAY
            ax.annotate(text, (x, row[key]), xytext=(4 if side == "r" else -4, 0), textcoords="offset points",
                        va="center", ha="left" if side == "r" else "right", fontsize=6.8, color=rs.INK, zorder=5,
                        bbox={"boxstyle": "square,pad=0.08", "facecolor": BAND, "edgecolor": "none"}
                        if crosses else None)
        rows.append({"row": key, "week_monday": d, "kind": f"milestone: {text or 'see next'}"})

    ax.axvline(TODAY, color=rs.INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=3)
    ax.text(TODAY, len(STREAMS) + 0.75, " today", ha="left", va="center", fontsize=7, color=rs.INK)
    ax.set_yticks(list(range(len(STREAMS))) + [gpu_y], [lab for _, lab, _ in STREAMS] + ["GPU training runs"])
    ax.set_ylim(len(STREAMS) + 0.95, -1.05)
    ax.set_xlim(START, END)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.tick_params(axis="x", which="major", labelsize=8)
    ax.tick_params(axis="y", labelsize=7.5, colors=rs.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", which="major", color=rs.GRID, linewidth=0.6)
    ax.grid(axis="x", which="minor", color=rs.GRID, linewidth=0.25)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("2026 (columns = calendar weeks)")
    rs.save(fig, "fig_gantt_actual", rows)
    with open(rs.DATA / "fig_gantt_actual_commits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "stream", "subject"])
        w.writeheader()
        w.writerows(audit)
    n = sum(1 for a in audit if a["stream"])
    print(f"{n}/{len(audit)} commits mapped to a stream; per stream:",
          {k: len(v) for k, v in active.items()}, "weeks")


if __name__ == "__main__":
    main()
