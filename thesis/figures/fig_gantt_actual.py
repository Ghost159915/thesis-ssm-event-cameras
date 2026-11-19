#!/usr/bin/env python
"""Report figure (Appendix, Project Timeline): what was actually done, February to November 2026 — the companion of
fig_gantt (the Thesis-A plan).

Evidence, not recollection:
* Work-stream bars: every commit subject in `git log` is mapped to a stream by the ordered rules in STREAM_RULES
  (first match wins; housekeeping is dropped). A stream's commit days are grouped into work periods: a period
  continues through quiet spells of up to BRIDGE_DAYS (4 weeks) and runs from its first to its last commit
  day; a short burst is drawn MIN_DAYS (10 days) long. The mapping
  of every commit is written to results/report_figs/fig_gantt_actual_commits.csv so it can be audited.
* One inferred period: the literature review and Thesis-A report from the start of Term 1 to the first commit
  (15 Apr 2026). The repository has no record of it, so it is drawn as a light tint, not a solid bar.
* GPU-training row: the exact span of every training run, from the W&B offline run folders (folder timestamp =
  start, newest file = end; external/ssms_event_cameras/RVT/wandb, 2026-10-16) and, for the two cloud PureSSM runs,
  the day-level dates in docs/notes/Stage13_cloud_notes.md and docs/roadmap/Stage_14_PureSSM_Full_Training.md.
* Milestones: dated results recorded in CLAUDE.md and the stage notes.
* Terms (UNSW 2026, "About UNSW 3+" trimester page and the 2026 timetable): T1 16 Feb - 14 May (Thesis A),
  T2 1 Jun - 27 Aug (Thesis B), T3 14 Sep - 10 Dec (Thesis C; teaching ends 20 Nov).
Drawn in the style of fig_gantt so the two charts read as a pair: square bars split at the term boundaries and
coloured with the same term ramp (Thesis A light -> C dark blue, imported from fig_gantt), dashed term boundaries,
a light weekly grid, and diamond milestones. As in the plan, the terms are contiguous: the break before a term
counts with that term (it was spent preparing it, e.g. the Thesis-C plan of 7 Sep). The single inferred period (the
literature review before the first commit) is drawn as a normal Thesis A bar.
Not captured: work outside the repository (reading, meetings).

  python thesis/figures/fig_gantt_actual.py
  -> thesis/latex/images/fig_gantt_actual.{pdf,png} + results/report_figs/fig_gantt_actual{,_commits}.csv
"""
from __future__ import annotations

import csv
import re
import subprocess
from datetime import date, datetime, timedelta

import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import report_style as rs
from fig_gantt import BAR_H, TERMS as PLAN_TERMS   # same bar height and term colours as the plan chart
from report_style import plt

STREAMS = [  # key, label
    ("litreview", "Literature review, Thesis-A report"),
    ("mvp", "S5-RVT reproduction"),
    ("eventssm", "EventSSM: build, train, test"),
    ("temporal", "Event-rate study"),
    ("efficiency", "Efficiency benchmark"),
    ("puressm", "PureSSM: build, cloud training, test"),
    ("spike_build", "SpikingSSM: build, first tests"),
    ("spike_train", "SpikingSSM: training, analysis"),
    ("research", "Research and planning"),
    ("inrc", "Neuromorphic-chip proposal (Intel)"),
    ("writing", "Report writing, supervisor reports"),
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
INFERRED = [("litreview", date(2026, 2, 16), date(2026, 4, 15))]   # Term-1 start .. first commit (no git record)
GPU_RUNS = [  # (start, end, what) — W&B offline folders unless marked (cloud, day-level)
    ("2026-06-16 15:36", "2026-06-16 22:09", "Stage 6 short runs"),
    ("2026-06-17 11:34", "2026-06-18 04:45", "Stage 7 mid-run"),
    ("2026-06-18 07:46", "2026-06-19 12:26", "EventSSM 400k"),
    ("2026-07-13 00:00", "2026-07-15 23:59", "PureSSM 25k + 400k (cloud RTX 5090, day-level)"),
    ("2026-10-06 21:53", "2026-10-07 21:48", "Stage 19 25k pilots"),
    ("2026-10-08 22:41", "2026-10-09 19:02", "spike 100k, graded 100k (stopped)"),
    ("2026-10-10 14:30", "2026-10-10 20:36", "graded 100k (resumed, stopped)"),
    ("2026-10-11 11:14", "2026-10-12 07:05", "graded 100k, analog 100k"),
    ("2026-10-12 07:05", "2026-10-12 20:31", "PureSSM 100k anchor"),
    ("2026-10-12 20:31", "2026-10-13 23:52", "graded 400k (stopped)"),
    ("2026-10-14 17:45", "2026-10-15 16:48", "graded 400k (resumed, stopped)"),
    ("2026-10-15 19:02", "2026-10-15 20:15", "graded 400k (resumed, stopped)"),
    ("2026-10-16 01:52", "2026-10-16 10:06", "graded 400k (finished)"),
]
MILESTONES = [  # (date, stream, label, side l/r): the label sits beyond the end of the bar that holds the milestone
    ("2026-06-05", "mvp", "reproduced, 47.7", "r"),
    ("2026-06-19", "eventssm", "test 46.2", "r"),
    ("2026-07-11", "temporal", "correction not reproduced", "r"),
    ("2026-07-11", "efficiency", "first efficiency results", "r"),
    ("2026-07-15", "puressm", "46.4, large objects +2.95", "r"),
    ("2026-09-07", "research", "Thesis-C plan", "l"),
    ("2026-10-07", "spike_build", "go/no-go check passed", "l"),
    ("2026-10-12", "spike_train", "", "l"),
    ("2026-10-16", "spike_train", "100k runs (12 Oct), 400k run (16 Oct)", "l"),
    ("2026-07-16", "inrc", "proposal drafted", "r"),
    ("2026-10-11", "inrc", "finalised", "r"),
    ("2026-10-13", "writing", "restructured", "r"),
]
START, END = date(2026, 2, 9), date(2026, 11, 30)
TODAY = date(2026, 10, 16)
TERM_STARTS = [START, date(2026, 5, 15), date(2026, 8, 28)]  # the day after T1 and T2 end: a break counts with the next term
TERMS = [(name, a, b, c) for (name, _, _, c), a, b in            # (name, first day, day after the last, colour)
         zip(PLAN_TERMS, TERM_STARTS, TERM_STARTS[1:] + [END])]
BRIDGE_DAYS = 28                                  # a work period continues through up to four quiet weeks
MIN_DAYS = 10                                     # a short burst of work is drawn at least ten days long
HATCH = "//////"                                  # the inferred (estimated) period


def commits() -> list[tuple[date, str]]:
    out = subprocess.run(["git", "log", "--format=%ad%x09%s", "--date=short"], cwd=rs.REPO, check=True,
                         capture_output=True, text=True).stdout
    return [(date.fromisoformat(d), s) for d, s in (line.split("\t", 1) for line in out.splitlines() if line)]


def classify(subject: str) -> str | None:
    for stream, rule in STREAM_RULES:
        if re.search(rule, subject, flags=re.IGNORECASE):
            return stream                                  # None for housekeeping
    return None


def periods(days: set[date]) -> list[tuple[date, date]]:
    """Group commit days into work periods: a gap of up to BRIDGE_DAYS keeps the period open."""
    out: list[list[date]] = []
    for d in sorted(days):
        if out and (d - out[-1][1]).days <= BRIDGE_DAYS:
            out[-1][1] = d
        else:
            out.append([d, d])
    end = TODAY + timedelta(days=1)                   # nothing is drawn in the future
    return [(a, min(max(b + timedelta(days=1), a + timedelta(days=MIN_DAYS)), end)) for a, b in out]


def main() -> None:
    rs.apply()
    plt.rcParams["hatch.linewidth"] = 0.6
    row = {k: i for i, (k, _) in enumerate(STREAMS)}
    gpu_y = len(STREAMS) + 0.15
    days: dict[str, set[date]] = {k: set() for k, _ in STREAMS}
    audit = []
    for d, subject in commits():
        if d > TODAY:                                  # the figure shows the project as of TODAY
            continue
        stream = classify(subject)
        audit.append({"date": d.isoformat(), "stream": stream or "", "subject": subject})
        if stream:
            days[stream].add(d)

    fig, ax = plt.subplots(figsize=(rs.TEXT_WIDTH, 3.9))
    ax.xaxis_date()
    num = mdates.date2num

    def bar(y, a, b, estimated=False, edge=0.8):
        """A square bar over [a, b), one segment per term it overlaps, coloured by term (as in fig_gantt)."""
        for _, t0, t1, c in TERMS:
            lo, hi = max(num(a), num(t0)), min(num(b), num(t1))   # numbers: a, b may be dates or datetimes
            if lo < hi:
                style = ({"facecolor": "white", "edgecolor": c, "hatch": HATCH, "linewidth": edge} if estimated
                         else {"color": c, "edgecolor": "white", "linewidth": edge})
                ax.barh(y, hi - lo, left=lo, height=BAR_H, zorder=2, **style)

    for _, t0, _, _ in TERMS[1:]:                         # term boundaries
        ax.axvline(num(t0), color=rs.AXIS, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    for name, t0, t1, _ in TERMS:
        ax.text(num(t0) + (num(t1) - num(t0)) / 2, -0.95, name, ha="center", va="bottom", fontsize=8,
                color=rs.INK_2)

    rows, spans = [], {k: [] for k, _ in STREAMS}
    for key, a, b in INFERRED:
        bar(row[key], a, b)                            # drawn as a normal Thesis A bar (user, 2026-11-19)
        rows.append({"row": key, "start": a.isoformat(), "end": b.isoformat(), "kind": "inferred (no git record)"})
    for key, _ in STREAMS:
        for a, b in periods(days[key]):
            bar(row[key], a, b)
            spans[key].append((a, b))
            n = sum(1 for d in days[key] if a <= d < b)
            rows.append({"row": key, "start": a.isoformat(), "end": b.isoformat(), "kind": f"{n} commit days"})
    for s, e, what in GPU_RUNS:
        s, e = datetime.fromisoformat(s), datetime.fromisoformat(e)
        bar(gpu_y, s, e, edge=0)                       # runs last hours: a white edge would hide them
        rows.append({"row": "gpu", "start": s.isoformat(timespec="minutes"), "end": e.isoformat(timespec="minutes"),
                     "kind": what})

    for d, key, text, side in MILESTONES:
        day = date.fromisoformat(d)
        ax.plot(num(day), row[key], "D", color=rs.INK, markersize=5.5, markeredgecolor="white",
                markeredgewidth=0.8, zorder=4)
        rows.append({"row": key, "start": d, "end": d, "kind": f"milestone: {text or 'see next'}"})
        if not text:
            continue
        a, b = next(((a, b) for a, b in spans[key] if a <= day < b), (day, day))
        x = num(b if side == "r" else a)
        crosses = side == "r" and TERMS[2][1] <= b < TODAY      # a label running across the today line hides it
        ax.annotate(text, (x, row[key]), xytext=(5 if side == "r" else -5, 0), textcoords="offset points",
                    va="center", ha="left" if side == "r" else "right", fontsize=7, color=rs.INK, zorder=5,
                    bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none"} if crosses
                    else None)

    ax.axvline(num(TODAY), color=rs.INK, linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=3)
    ax.text(num(TODAY), len(STREAMS) + 0.75, " today", ha="left", va="center", fontsize=7, color=rs.INK)
    ax.set_yticks(list(range(len(STREAMS))) + [gpu_y], [lab for _, lab in STREAMS] + ["GPU training runs"])
    ax.set_ylim(len(STREAMS) + 0.95, -1.05)
    ax.set_xlim(num(START), num(END))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))   # weekly grid, as in fig_gantt
    ax.tick_params(axis="x", which="major", labelsize=7)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="y", labelsize=8, colors=rs.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", which="major", visible=False)
    ax.grid(axis="x", which="minor", color=rs.GRID, linewidth=0.4)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("2026")
    key = [Patch(color=c, label=t) for t, _, _, c in TERMS]
    key.append(Line2D([], [], marker="D", color=rs.INK, linestyle="none", markersize=5, label="result"))
    ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, fontsize=7.5,
              handlelength=1.2, columnspacing=1.6)
    rs.save(fig, "fig_gantt_actual", rows)
    with open(rs.DATA / "fig_gantt_actual_commits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "stream", "subject"])
        w.writeheader()
        w.writerows(audit)
    print(f"{sum(1 for a in audit if a['stream'])}/{len(audit)} commits mapped; periods:",
          {k: len(v) for k, v in spans.items()})


if __name__ == "__main__":
    main()
