"""Shared style for every data figure of the final report (thesis/latex/main.tex). Import, never copy.

One model = one colour and one marker in every figure, so a reader who has learnt the key once can read every chart.
Colours are slots of the validated reference categorical palette (dataviz skill, references/palette.md), checked
with its validator on the white page (`validate_palette.js --pairs all --surface #ffffff`):

* S5-RVT is the reference baseline and is drawn in neutral grey, not a hue, so it recedes behind our models
  (grey vs blue/orange/aqua: normal-vision dE >= 17.4, CVD pass).
* EventSSM blue / PureSSM orange / SpikingSSM-graded aqua: the first three palette slots, which pass all-pairs.
* The two other spiking arms, analog yellow and spike violet, pass all-pairs together with graded aqua (worst CVD
  dE 9.1, normal-vision 22.9). Never put analog (yellow) next to PureSSM (orange) as two *lines*: that pair is
  below the normal-vision floor (13.7). In the ladder figures PureSSM is therefore a reference (zero line or its own
  x position), not a fourth line.
* Aqua and yellow are below 3:1 contrast on white, so those series are always direct-labelled or tabulated.

Every figure is saved as vector PDF (for LaTeX) + 300-dpi PNG (preview), plus a CSV of exactly the plotted values
(the table view; results/report_figs/). Sizes are in inches at the report's text width (A4, 1 in margins).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
IMAGES = REPO / "thesis/latex/images"
DATA = REPO / "results/report_figs"

TEXT_WIDTH = 6.27                      # in: 210 mm - 2 x 25.4 mm
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# key -> (label, colour, marker). Keys match the result files (val_histories.json, captures, eval tags).
MODELS = {
    "s5rvt":    ("S5-RVT (baseline)", "#6f6e69", "s"),
    "eventssm": ("EventSSM",          "#2a78d6", "o"),
    "puressm":  ("PureSSM",           "#eb6834", "^"),
    "analog":   ("Analog",            "#eda100", "h"),
    "graded":   ("Graded",            "#1baf7a", "D"),
    "spike":    ("Spike",             "#6250d6", "v"),
}


def label(key: str) -> str:
    return MODELS[key][0]


def colour(key: str) -> str:
    return MODELS[key][1]


def marker(key: str) -> str:
    return MODELS[key][2]


def apply() -> None:
    """rcParams for every report figure: sans text at 8.5-9 pt, recessive axes, editable fonts in the PDF."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        "pdf.fonttype": 42,              # TrueType: text stays text in the PDF
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def signed(v: float, digits: int = 1) -> str:
    """+1.2 / −1.2 with a true minus sign, matching matplotlib's axis ticks."""
    return f"{v:+.{digits}f}".replace("-", "−")


def panel_tag(ax, tag: str, note: str = "") -> None:
    """(a)/(b) tag at the top-left corner of a panel, with an optional short note (e.g. the schedule); the caption
    explains each panel, the figure itself has no title."""
    t = ax.text(0, 1.03, f"({tag})", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold",
                color=INK)
    if note:
        ax.annotate(f" {note}", xy=(1, 0), xycoords=t, va="bottom", fontsize=8, color=INK_2)


def save(fig, name: str, rows: list[dict] | None = None) -> None:
    """thesis/latex/images/<name>.pdf + .png, and results/report_figs/<name>.csv with the plotted values."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / f"{name}.pdf")
    fig.savefig(IMAGES / f"{name}.png", dpi=300)
    if rows:
        DATA.mkdir(parents=True, exist_ok=True)
        with open(DATA / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    plt.close(fig)
    print(f"wrote thesis/latex/images/{name}.pdf/.png" + (f" + results/report_figs/{name}.csv" if rows else ""))
