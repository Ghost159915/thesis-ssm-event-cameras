"""Generates the Thesis-C architecture figure: the two orthogonal mixing axes of the
backbone, where each of the four models intervenes, and the LIF readout mechanism.

Truthfulness: the membrane trace in panel (c) is NOT drawn by hand -- it is produced by
running the real `LIFReadout` from code/event_ssm/models/spikingssm/lif.py on a fixed
illustrative stimulus, so the figure cannot drift from the implementation. Re-run this
script after any change to the LIF and the figure updates itself.

Usage:  python thesis/diagrams/fig_spiking_architecture.py [--compact]
Writes: fig_spiking_architecture.svg (source of truth) next to this file.
        --compact writes fig_spiking_architecture_compact.svg instead: panel (b), the model grid, is left out
        (the final report has the same information in Table tab:four_models), the LIF panel becomes (b) and the
        vertical gaps are tightened, so the figure is ~30 % shorter at the same text size.
        Convert to PDF/PNG for LaTeX with cairosvg (see the runbook line at the bottom).
"""
import pathlib
import sys

# --- real dynamics, pulled from the implementation -------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "code"))
import torch                                                              # noqa: E402
from event_ssm.models.spikingssm.lif import LIFReadout                    # noqa: E402

BETA, THR = 0.85, 1.0
STIM = [.45, .55, .30, .62, .70, .25, .15, .58, .66, .40, .20, .52, .61, .35]

def _trace():
    x = torch.tensor(STIM).view(1, len(STIM), 1)
    kw = dict(d_model=1, beta=BETA, threshold=THR, learn_beta=False)
    pre, _ = LIFReadout(output_mode="analog", **kw).eval()(x)
    ref = LIFReadout(output_mode="spike", **kw).eval()
    spk, _ = ref(x)
    pre = pre.flatten().tolist()
    spk = [int(v) for v in spk.flatten().tolist()]
    post = [p - s * THR for p, s in zip(pre, spk)]
    return pre, post, spk, ref.last_firing_rate

# --- palette / type --------------------------------------------------------------------
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
INK, MUTE = "#0f172a", "#475569"
SPA, SPA_BG = "#1d4ed8", "#dbeafe"          # spatial axis
TEM, TEM_BG = "#b45309", "#fef3c7"          # temporal axis
SPK, SPK_BG = "#6d28d9", "#ede9fe"          # the new spiking layer
FRZ, FRZ_BG = "#475569", "#f1f5f9"          # frozen / shared
LINE = "#94a3b8"

W, H = 1300, 1300
COMPACT = "--compact" in sys.argv[1:]     # final report: no model grid, tighter gaps
out = []
def add(s): out.append(s)

def rect(x, y, w, h, fill="#fff", stroke=LINE, rx=7, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

def txt(x, y, s, size=15, anchor="middle", weight="normal", fill=INK, style="normal"):
    s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" fill="{fill}" text-anchor="{anchor}">{s}</text>')

def arrow(x1, y1, x2, y2, stroke=MUTE, sw=1.8):
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" '
        f'marker-end="url(#ah)"/>')

def line(x1, y1, x2, y2, stroke=LINE, sw=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

def panel(x, y, letter, title):
    txt(x, y, letter, 20, "start", "bold", INK)
    txt(x + 26, y, title, 17, "start", "bold", INK)

ZY = 270 if COMPACT else 286                 # top of the "inside one stage" zoom box
LY = ZY - 4 + 250 + 46 if COMPACT else 928   # top of the LIF panel
TOP, BOTTOM = 78, LY + 340                   # visible band: the title area above panel (a) is cropped
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{BOTTOM - TOP}" '
    f'viewBox="0 {TOP} {W} {BOTTOM - TOP}">')
add('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTE}"/>'
    '</marker></defs>')
add(f'<rect y="{TOP}" width="{W}" height="{BOTTOM - TOP}" fill="#ffffff"/>')


# ============================== (a) pipeline ===========================================
panel(40, 105, "a", "The detector, shared by all models")
Y = 140
def chip(x, w, label, sub, fill, stroke, h=56):
    rect(x, Y, w, h, fill, stroke)
    txt(x + w / 2, Y + 24, label, 14, "middle", "bold", INK)
    txt(x + w / 2, Y + 42, sub, 11.5, "middle", "normal", MUTE)

chip(40, 96, "events", "from camera", "#fff", LINE)
arrow(140, Y + 28, 168, Y + 28)
chip(172, 150, "event histogram", "20 x 240 x 304", "#fff", LINE)
arrow(324, Y + 28, 352, Y + 28)
rect(356, Y - 16, 462, 88, "#fbfdff", INK, 9, 2.0)
txt(587, Y + 10, "Backbone  (4 stages)", 14, "middle", "bold", INK)
txt(587, Y + 32, "the only part that differs between models", 12, "middle", "normal", MUTE)
txt(587, Y + 52, "stages 2, 3 and 4 feed the neck", 12, "middle", "normal", MUTE)
arrow(822, Y + 28, 850, Y + 28)
chip(854, 118, "Neck", "combines stages", FRZ_BG, FRZ)
arrow(976, Y + 28, 1004, Y + 28)
chip(1008, 118, "Head", "class + box", FRZ_BG, FRZ)
arrow(1130, Y + 28, 1158, Y + 28)
chip(1162, 98, "detections", "cars, people", "#fff", LINE)

line(854, Y + 70, 1126, Y + 70, FRZ, 1.4)
line(854, Y + 70, 854, Y + 62, FRZ, 1.4); line(1126, Y + 70, 1126, Y + 62, FRZ, 1.4)
txt(990, Y + 88, "the same in all four models", 12, "middle", "bold", FRZ)

# ---- zoom into one stage
line(430, Y + 72, 320, ZY - 8, LINE, 1.2, "4 3")
line(744, Y + 72, 1000 if COMPACT else 1100, ZY - 8, LINE, 1.2, "4 3")   # compact: clear of the label
rect(290, ZY - 4, 840, 250, "#fcfcfd", LINE, 10, 1.6, "6 4")
txt(710, ZY + 20, "Inside one backbone stage: two mixers", 14, "middle", "bold", INK)

rect(320, ZY + 38, 350, 86, SPA_BG, SPA, 8, 2.0)
txt(495, ZY + 66, "Spatial mixer", 15, "middle", "bold", SPA)
txt(495, ZY + 90, "looks across the image, one frame at a time", 12.5, "middle", "normal", INK)
txt(495, ZY + 110, "no memory between frames", 12, "middle", "normal", MUTE)

arrow(674, ZY + 80, 706, ZY + 80)

rect(710, ZY + 38, 350, 86, TEM_BG, TEM, 8, 2.0)
txt(885, ZY + 66, "Temporal mixer", 15, "middle", "bold", TEM)
txt(885, ZY + 90, "looks across time, at each position", 12.5, "middle", "normal", INK)
txt(885, ZY + 110, "keeps a memory from frame to frame", 12, "middle", "normal", MUTE)

arrow(885, ZY + 126, 885, ZY + 144, SPK)
rect(710, ZY + 146, 350, 76, SPK_BG, SPK, 8, 2.2)
txt(885, ZY + 170, "+  spiking readout (LIF)", 15, "middle", "bold", SPK)
txt(885, ZY + 191, "new: added after the temporal mixer", 12, "middle", "normal", SPK)
txt(885, ZY + 211, "numbers in, spikes out", 12, "middle", "bold", SPK)

rect(320, ZY + 146, 350, 76, "#fff", LINE, 8, 1.3)
txt(495, ZY + 170, "Stage 1 has no temporal mixer", 12.5, "middle", "bold", MUTE)
txt(495, ZY + 191, "its output does not reach the neck,", 12, "middle", "normal", MUTE)
txt(495, ZY + 209, "so a mixer there would have no effect", 12, "middle", "normal", MUTE)

# ============================== (b) model grid =========================================
if not COMPACT:
    GY = 588
    panel(40, GY, "b", "What each model changes")
    rows = [
        ("S5-RVT  (baseline)",           "MaxViT attention", False, "S5",              False, "47.7", ""),
        ("EventSSM  (this work)",         "ResNet-18 (CNN)",  False, "Mamba-2",         False, "46.2", ""),
        ("PureSSM  (this work)",          "BiMamba",          True,  "Mamba-2",         False, "46.4", ""),
        ("SpikingSSM  (this work)",      "BiMamba",          False, "Mamba-2  +  LIF", True,  "46.2", "graded"),
    ]
    cx = [56, 430, 760, 1032]
    hy = GY + 34
    txt(cx[0], hy, "model", 13, "start", "bold", MUTE)
    txt(cx[1] + 140, hy, "Spatial mixer", 13, "middle", "bold", SPA)
    txt(cx[2] + 130, hy, "Temporal mixer", 13, "middle", "bold", TEM)
    txt(cx[3] + 52, hy, "Test mAP", 13, "middle", "bold", MUTE)
    line(48, hy + 10, 1256, hy + 10, LINE, 1.2)

    for i, (name, sp, sp_new, tp, tp_new, ap, note) in enumerate(rows):
        ry = hy + 40 + i * 48
        last = i == len(rows) - 1
        if last:
            rect(48, ry - 21, 1208, 42, "#fbfaff", SPK, 6, 1.6)
        txt(cx[0], ry + 5, name, 13.5, "start", "bold" if last else "normal", SPK if last else INK)
        if sp_new or tp_new:                                   # caret marks the CHANGED cell
            mx, mc = (cx[1] - 15, SPA) if sp_new else (cx[2] - 15, SPK)
            add(f'<path d="M {mx} {ry - 2} L {mx + 9} {ry + 4} L {mx} {ry + 10} z" fill="{mc}"/>')
        rect(cx[1], ry - 15, 280, 30, SPA_BG if sp_new else "#f8fafc", SPA if sp_new else LINE, 6,
             2.0 if sp_new else 1.2)
        txt(cx[1] + 140, ry + 5, sp, 13, "middle", "bold" if sp_new else "normal", SPA if sp_new else INK)
        rect(cx[2], ry - 15, 260, 30, SPK_BG if tp_new else "#f8fafc", SPK if tp_new else LINE, 6,
             2.2 if tp_new else 1.2)
        txt(cx[2] + 130, ry + 5, tp, 13, "middle", "bold" if tp_new else "normal", SPK if tp_new else INK)
        txt(cx[3] + 52, ry + 5, ap, 14, "middle", "bold", INK)
        if note:
            txt(cx[3] + 140, ry + 5, note, 12, "middle", "normal", SPK)

    fy = hy + 40 + 4 * 48 + 14
    txt(56, fy + 6, "The small arrow marks what changes from the row above. PureSSM changes the spatial mixer; "
        "SpikingSSM changes the temporal one.", 12.5, "start", "normal", MUTE)
    txt(56, fy + 28, "Everything else (data, neck, head, loss and training settings) is the same for all four models.",
        12.5, "start", "normal", MUTE)

# ============================== (c) LIF mechanism ======================================
panel(40, LY, "b" if COMPACT else "c", "How the spiking readout works (trace from the real code)")
pre, post, spk, rate = _trace()

rect(48, LY + 26, 470, 300, "#fcfcfd", LINE, 10, 1.6)
txt(283, LY + 52, "at every time step, for each value y from the mixer", 13, "middle", "normal", MUTE)
eqs = [("m  \u2190  \u03b2 m  +  y", "add the input; the old total slowly leaks away", TEM),
       ("s  =  1  if  m \u2265 \u03d1,  else 0", "fire a spike when m reaches the threshold \u03d1", SPK),
       ("m  \u2190  m  \u2212  s \u03d1", "after a spike, take the threshold off m", SPK)]
for i, (e, c, col) in enumerate(eqs):
    ey = LY + 90 + i * 58
    rect(70, ey - 21, 426, 46, "#ffffff", col, 6, 1.5)
    txt(283, ey - 2, e, 14, "middle", "bold", INK)
    txt(283, ey + 16, c, 12, "middle", "normal", MUTE)

txt(283, LY + 282, "Training uses a smooth stand-in for the step,", 12.5, "middle", "bold", INK)
txt(283, LY + 302, "because the step itself gives no slope to learn from.", 12, "middle", "normal", MUTE)

PX, PY, PW, PH = 566, LY + 66, 686, 172
rect(548, LY + 26, 708, 300, "#fcfcfd", LINE, 10, 1.6)
txt(902, LY + 52, "The running total m over time", 13, "middle", "bold", INK)
vmax = 1.45
def sx(i): return PX + 26 + i * (PW - 90) / (len(STIM) - 1)
def sy(v): return PY + PH - 22 - (v / vmax) * (PH - 46)

line(PX + 8, sy(0), PX + PW - 40, sy(0), MUTE, 1.4)
line(PX + 8, sy(0), PX + 8, PY + 4, MUTE, 1.4)
line(PX + 8, sy(THR), PX + PW - 40, sy(THR), SPK, 1.6, "6 4")
txt(PX + 16, sy(THR) - 9, "threshold \u03d1", 12, "start", "bold", SPK)
txt(PX, sy(0) + 5, "0", 11, "end", "normal", MUTE)

pts = []
for i in range(len(STIM)):
    pts.append((sx(i), sy(pre[i])))
    if spk[i]:
        pts.append((sx(i), sy(post[i])))
add('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
    f'" fill="none" stroke="{TEM}" stroke-width="2.4" stroke-linejoin="round"/>')
for i in range(len(STIM)):
    if spk[i]:
        add(f'<circle cx="{sx(i):.1f}" cy="{sy(pre[i]):.1f}" r="4.2" fill="{SPK}"/>')
        line(sx(i), PY + PH - 8, sx(i), PY + PH + 16, SPK, 3.2)
txt(PX + 8, PY + PH + 36, "output spikes", 12, "start", "bold", SPK)
txt(PX + PW - 40, PY + PH + 36, f"fires on {rate * 100:.0f} % of steps", 12, "end", "normal", MUTE)
txt(902, PY + PH + 66, "Each spike lowers m by one threshold; anything above it is kept.", 12, "middle", "normal", MUTE)

add('</svg>')

dest = pathlib.Path(__file__).with_name("fig_spiking_architecture" + ("_compact" if COMPACT else "") + ".svg")
dest.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {dest}  (firing rate {rate:.3f}, {sum(spk)}/{len(spk)} steps fired)")
