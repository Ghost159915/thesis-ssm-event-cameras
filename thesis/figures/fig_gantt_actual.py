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

import report_style as rs
from report_style import plt

NEUTRAL = "#4a4945"                      # cross-cutting work (not one model): dark neutral, distinct from S5-RVT grey
STREAMS = [  # key, label, colour
    ("litreview", "Literature review, Thesis-A report", NEUTRAL),
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
INFERRED = [("litreview", date(2026, 2, 16), date(2026, 4, 15))]   # Term-1 start .. first commit (no git record)
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
MILESTONES = [  # (date, stream, label, side l/r): the label sits beyond the end of the bar that holds the milestone
    ("2026-06-05", "mvp", "reproduced, 47.7", "r"),
    ("2026-06-19", "eventssm", "test 46.2", "r"),
    ("2026-07-11", "temporal", "Δt rescaling not reproducible", "r"),
    ("2026-07-11", "efficiency", "first efficiency results", "r"),
    ("2026-07-15", "puressm", "test 46.4, AP$_L$ +3.0", "r"),
    ("2026-09-07", "research", "Thesis-C plan", "l"),
    ("2026-10-07", "spike_build", "kill-switch passed", "r"),
    ("2026-10-12", "spike_train", "", "l"),
    ("2026-10-16", "spike_train", "100k ladder (12 Oct), 400k (16 Oct)", "l"),
    ("2026-07-16", "inrc", "proposal drafted", "r"),
    ("2026-10-11", "inrc", "finalised", "r"),
    ("2026-10-13", "writing", "restructured", "r"),
]
TERMS = [("Thesis A (T1)", date(2026, 2, 16), date(2026, 5, 14)),
         ("Thesis B (T2)", date(2026, 6, 1), date(2026, 8, 27)),
         ("Thesis C (T3)", date(2026, 9, 14), date(2026, 12, 10))]
START, END = date(2026, 2, 1), date(2026, 11, 30)
TODAY = date(2026, 10, 16)
BRIDGE_DAYS = 28                                  # a work period continues through up to four quiet weeks
MIN_DAYS = 10                                     # a short burst of work is drawn at least ten days long
BAND = "#f1f0eb"                                  # term shading
BAR_PT = 9.0                                      # bar thickness (points); the round caps add BAR_PT / 2 at each end


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
    row = {k: i for i, (k, _, _) in enumerate(STREAMS)}
    gpu_y = len(STREAMS) + 0.15
    days: dict[str, set[date]] = {k: set() for k, _, _ in STREAMS}
    audit = []
    for d, subject in commits():
        stream = classify(subject)
        audit.append({"date": d.isoformat(), "stream": stream or "", "subject": subject})
        if stream:
            days[stream].add(d)

    fig, ax = plt.subplots(figsize=(rs.TEXT_WIDTH, 3.9))
    fig.canvas.draw()                                  # axes size is needed to turn the round caps into days
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    cap_days = (BAR_PT / 2 / 72) * (END - START).days / bbox.width

    def bar(y, a, b, colour, alpha=1.0, lw=BAR_PT, cap="round"):
        """A rounded bar whose visible extent is exactly [a, b] (the round caps are drawn inside the span)."""
        x0, x1 = mdates.date2num(a), mdates.date2num(b)
        inset = min(cap_days, (x1 - x0 - 1.5) / 2) if cap == "round" else 0  # keep >= 1.5 d of straight line:
        # a near-zero segment is dropped by the renderer, which made short bars vanish
        x0, x1 = x0 + inset, x1 - inset
        ax.plot([x0, x1], [y, y], color=colour, alpha=alpha, linewidth=lw, solid_capstyle=cap, zorder=2)

    for name, a, b in TERMS:
        ax.axvspan(a, min(b, END), color=BAND, zorder=0)
        ax.text(a + (min(b, END) - a) / 2, -0.95, name, ha="center", va="bottom", fontsize=8, color=rs.INK_2)

    rows, spans = [], {k: [] for k, _, _ in STREAMS}
    for key, a, b in INFERRED:
        bar(row[key], a, b, dict((k, c) for k, _, c in STREAMS)[key], alpha=0.3)
        rows.append({"row": key, "start": a.isoformat(), "end": b.isoformat(), "kind": "inferred (no git record)"})
    for key, _, colour in STREAMS:
        for a, b in periods(days[key]):
            bar(row[key], a, b, colour)
            spans[key].append((a, b))
            n = sum(1 for d in days[key] if a <= d < b)
            rows.append({"row": key, "start": a.isoformat(), "end": b.isoformat(), "kind": f"{n} commit days"})
    for s, e, ck, what in GPU_RUNS:
        s, e = datetime.fromisoformat(s), datetime.fromisoformat(e)
        ax.plot([mdates.date2num(s), mdates.date2num(e)], [gpu_y, gpu_y], color=rs.colour(ck), linewidth=6,
                solid_capstyle="butt", zorder=2)
        rows.append({"row": "gpu", "start": s.isoformat(timespec="minutes"), "end": e.isoformat(timespec="minutes"),
                     "kind": what})

    for d, key, text, side in MILESTONES:
        day = date.fromisoformat(d)
        ax.plot(mdates.date2num(day), row[key], "D", color=rs.INK, markersize=4.6, markeredgecolor="white",
                markeredgewidth=0.7, zorder=4)
        rows.append({"row": key, "start": d, "end": d, "kind": f"milestone: {text or 'see next'}"})
        if not text:
            continue
        a, b = next(((a, b) for a, b in spans[key] if a <= day < b), (day, day))
        x = mdates.date2num(b if side == "r" else a)
        crosses = side == "r" and TERMS[2][1] <= b < TODAY      # a label running across the today line hides it
        ax.annotate(text, (x, row[key]), xytext=(4 if side == "r" else -4, 0), textcoords="offset points",
                    va="center", ha="left" if side == "r" else "right", fontsize=6.8, color=rs.INK, zorder=5,
                    bbox={"boxstyle": "square,pad=0.08", "facecolor": BAND, "edgecolor": "none"} if crosses
                    else None)

    ax.axvline(TODAY, color=rs.INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=3)
    ax.text(TODAY, len(STREAMS) + 0.75, " today", ha="left", va="center", fontsize=7, color=rs.INK)
    ax.set_yticks(list(range(len(STREAMS))) + [gpu_y], [lab for _, lab, _ in STREAMS] + ["GPU training runs"])
    ax.set_ylim(len(STREAMS) + 0.95, -1.05)
    ax.set_xlim(START, END)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", which="major", labelsize=8)
    ax.tick_params(axis="y", labelsize=7.5, colors=rs.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", which="major", color=rs.GRID, linewidth=0.6)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("2026")
    key = [Line2D([], [], color=NEUTRAL, linewidth=BAR_PT, solid_capstyle="round", label="work period (commits)"),
           Line2D([], [], color=NEUTRAL, alpha=0.3, linewidth=BAR_PT, solid_capstyle="round",
                  label="inferred, before the repository"),
           Line2D([], [], marker="D", color=rs.INK, linestyle="none", markersize=4.6, label="result / milestone")]
    ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7.5,
              handlelength=1.6, columnspacing=1.6)
    rs.save(fig, "fig_gantt_actual", rows)
    with open(rs.DATA / "fig_gantt_actual_commits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "stream", "subject"])
        w.writeheader()
        w.writerows(audit)
    print(f"{sum(1 for a in audit if a['stream'])}/{len(audit)} commits mapped; periods:",
          {k: len(v) for k, v in spans.items()})


if __name__ == "__main__":
    main()
