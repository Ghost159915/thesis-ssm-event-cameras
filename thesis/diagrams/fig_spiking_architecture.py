"""Generates the Thesis-C architecture figure: the two orthogonal mixing axes of the
backbone, where each of the four models intervenes, and the LIF readout mechanism.

Truthfulness: the membrane trace in panel (c) is NOT drawn by hand -- it is produced by
running the real `LIFReadout` from code/event_ssm/models/spikingssm/lif.py on a fixed
illustrative stimulus, so the figure cannot drift from the implementation. Re-run this
script after any change to the LIF and the figure updates itself.

Usage:  python thesis/diagrams/fig_spiking_architecture.py
Writes: fig_spiking_architecture.svg (source of truth) next to this file.
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

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTE}"/>'
    '</marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

txt(W / 2, 40, "Spiking SSM detector: where it sits, and how it differs from PureSSM", 21, "middle", "bold")
txt(W / 2, 63, "All four models share one frozen pipeline; only the backbone changes.", 14, "middle", "normal", MUTE)

# ============================== (a) pipeline ===========================================
panel(40, 105, "a", "Shared detector pipeline")
Y = 140
def chip(x, w, label, sub, fill, stroke, h=56):
    rect(x, Y, w, h, fill, stroke)
    txt(x + w / 2, Y + 24, label, 14, "middle", "bold", INK)
    txt(x + w / 2, Y + 42, sub, 11.5, "middle", "normal", MUTE)

chip(40, 96, "events", "async DVS", "#fff", LINE)
arrow(140, Y + 28, 168, Y + 28)
chip(172, 150, "stacked histogram", "20 x 240 x 304", "#fff", LINE)
arrow(324, Y + 28, 352, Y + 28)
rect(356, Y - 16, 462, 88, "#fbfdff", INK, 9, 2.0)
txt(587, Y + 4, "BACKBONE  (4 stages)", 14, "middle", "bold", INK)
txt(587, Y + 23, "the only part that differs between models", 11.5, "middle", "italic", MUTE)
txt(587, Y + 41, "dims 64 / 128 / 256 / 512      strides 4 / 8 / 16 / 32", 11, "middle", "normal", MUTE)
txt(587, Y + 59, "FPN consumes stages 2, 3, 4  (strides 8 / 16 / 32)", 11, "middle", "normal", MUTE)
arrow(822, Y + 28, 850, Y + 28)
chip(854, 118, "YOLO-PAFPN", "neck", FRZ_BG, FRZ)
arrow(976, Y + 28, 1004, Y + 28)
chip(1008, 118, "YOLOX head", "cls + box", FRZ_BG, FRZ)
arrow(1130, Y + 28, 1158, Y + 28)
chip(1162, 98, "detections", "car / ped", "#fff", LINE)

line(854, Y + 70, 1126, Y + 70, FRZ, 1.4)
line(854, Y + 70, 854, Y + 62, FRZ, 1.4); line(1126, Y + 70, 1126, Y + 62, FRZ, 1.4)
txt(990, Y + 88, "FROZEN - identical in all four models", 12, "middle", "bold", FRZ)
txt(990, Y + 104, "(also: loss, Gen1 data pipeline, evaluator, training recipe)", 11, "middle", "italic", FRZ)

# ---- zoom into one stage
ZY = 286
line(430, Y + 72, 320, ZY - 8, LINE, 1.2, "4 3")
line(744, Y + 72, 1100, ZY - 8, LINE, 1.2, "4 3")
rect(290, ZY - 4, 840, 250, "#fcfcfd", LINE, 10, 1.6, "6 4")
txt(710, ZY + 20, "inside one backbone stage - two orthogonal mixing axes", 14, "middle", "bold", INK)

rect(320, ZY + 38, 350, 86, SPA_BG, SPA, 8, 2.0)
txt(495, ZY + 62, "SPATIAL mixer", 15, "middle", "bold", SPA)
txt(495, ZY + 82, "mixes across H, W  -  within ONE frame", 12, "middle", "normal", INK)
txt(495, ZY + 100, "(L\u00b7B, C, H, W)   time folded into batch", 11.5, "middle", "normal", MUTE)
txt(495, ZY + 116, "stateless  \u00b7  may be bidirectional (space is not causal)", 10.5, "middle", "italic", MUTE)

arrow(674, ZY + 80, 706, ZY + 80)

rect(710, ZY + 38, 350, 86, TEM_BG, TEM, 8, 2.0)
txt(885, ZY + 62, "TEMPORAL mixer", 15, "middle", "bold", TEM)
txt(885, ZY + 82, "mixes across time  -  at ONE pixel", 12, "middle", "normal", INK)
txt(885, ZY + 100, "(B\u00b7H\u00b7W, L, C)   space folded into batch", 11.5, "middle", "normal", MUTE)
txt(885, ZY + 116, "causal  \u00b7  state carried across clips", 10.5, "middle", "italic", MUTE)

arrow(885, ZY + 126, 885, ZY + 144, SPK)
rect(710, ZY + 146, 350, 76, SPK_BG, SPK, 8, 2.2)
txt(885, ZY + 170, "+  LIF spiking readout", 15, "middle", "bold", SPK)
txt(885, ZY + 190, "NEW - appended, replaces nothing", 12, "middle", "italic", SPK)
txt(885, ZY + 210, "continuous in  ->  binary spikes out", 11.5, "middle", "bold", SPK)

rect(320, ZY + 146, 350, 76, "#fff", LINE, 8, 1.3)
txt(495, ZY + 170, "stage 1 has NO temporal block", 12.5, "middle", "bold", MUTE)
txt(495, ZY + 189, "temporal_stages = (2, 3, 4). Stage 1 is not FPN-fed,", 11, "middle", "normal", MUTE)
txt(495, ZY + 206, "so a block there would be dead parameters.", 11, "middle", "normal", MUTE)

# ============================== (b) model grid =========================================
GY = 588
panel(40, GY, "b", "What each model changes - the two axes are independent")
rows = [
    ("S5-RVT  (reproduced baseline)", "MaxViT attention", False, "S5 diagonal SSM", False, "47.7", ""),
    ("EventSSM  (Thesis B)",          "ResNet-18 conv",   False, "Mamba-2",         False, "46.2", ""),
    ("PureSSM  (Thesis B)",           "BiMamba scan",     True,  "Mamba-2",         False, "46.4", ""),
    ("Spiking-SSM  (Thesis C)",       "BiMamba scan",     False, "Mamba-2  +  LIF", True,  "?",    "3 readout arms"),
]
cx = [56, 430, 760, 1032]
hy = GY + 34
txt(cx[0], hy, "model", 13, "start", "bold", MUTE)
txt(cx[1] + 140, hy, "SPATIAL mixer", 13, "middle", "bold", SPA)
txt(cx[2] + 130, hy, "TEMPORAL mixer", 13, "middle", "bold", TEM)
txt(cx[3] + 52, hy, "Gen1 test mAP", 13, "middle", "bold", MUTE)
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
        txt(cx[3] + 168, ry + 5, note, 11.5, "middle", "italic", SPK)

fy = hy + 40 + 4 * 48 + 14
txt(56, fy, "PureSSM moved the SPATIAL axis; Spiking-SSM moves the TEMPORAL one - so the spiking block is "
    "spatial-agnostic and pairs with either backbone.", 12, "start", "italic", MUTE)
txt(56, fy + 20, "The caret marks the changed box. S5-RVT is the reproduced published baseline; EventSSM / PureSSM / Spiking-SSM are drop-in "
    "backbone swaps into its pipeline. mAP is COCO 0.50:0.95, Gen1 test split.", 11.5, "start", "italic", MUTE)

# ============================== (c) LIF mechanism ======================================
LY = 928
panel(40, LY, "c", "The LIF readout - measured from the implementation, not drawn by hand")
pre, post, spk, rate = _trace()

rect(48, LY + 26, 470, 300, "#fcfcfd", LINE, 10, 1.6)
txt(283, LY + 52, "per timestep t, at every pixel and channel", 13, "middle", "italic", MUTE)
eqs = [("mem  =  beta * mem  +  y[t]", "leaky integration of the SSM output", TEM),
       ("s[t]  =  1  if  mem >= threshold  else 0", "fire", SPK),
       ("mem  =  mem  -  s[t] * threshold", "reset by subtraction (keeps the residue)", SPK)]
for i, (e, c, col) in enumerate(eqs):
    ey = LY + 90 + i * 58
    rect(70, ey - 21, 426, 46, "#ffffff", col, 6, 1.5)
    txt(283, ey - 2, e, 14, "middle", "bold", INK)
    txt(283, ey + 16, c, 11, "middle", "italic", MUTE)

txt(283, LY + 282, "forward: a hard step.   backward: arctan surrogate gradient.", 12.5, "middle", "bold", INK)
txt(283, LY + 302, "Without the surrogate the derivative is zero almost everywhere", 11, "middle", "italic", MUTE)
txt(283, LY + 317, "and nothing downstream of the spike can train.", 11, "middle", "italic", MUTE)

PX, PY, PW, PH = 566, LY + 66, 686, 172
rect(548, LY + 26, 708, 300, "#fcfcfd", LINE, 10, 1.6)
txt(902, LY + 52, "membrane potential integrating the Mamba-2 output", 13, "middle", "bold", INK)
vmax = 1.45
def sx(i): return PX + 26 + i * (PW - 90) / (len(STIM) - 1)
def sy(v): return PY + PH - 22 - (v / vmax) * (PH - 46)

line(PX + 8, sy(0), PX + PW - 40, sy(0), MUTE, 1.4)
line(PX + 8, sy(0), PX + 8, PY + 4, MUTE, 1.4)
line(PX + 8, sy(THR), PX + PW - 40, sy(THR), SPK, 1.6, "6 4")
txt(PX + 16, sy(THR) - 9, "threshold", 11, "start", "bold", SPK)
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
txt(PX + 8, PY + PH + 36, "output spikes", 11.5, "start", "bold", SPK)
txt(PX + PW - 40, PY + PH + 36, f"firing rate {rate:.2f}", 11.5, "end", "italic", MUTE)
txt(902, PY + PH + 62, "Each spike drops the membrane by exactly one threshold: the residue", 11, "middle", "italic", MUTE)
txt(902, PY + PH + 77, "above threshold is carried forward, not discarded.", 11, "middle", "italic", MUTE)

add('</svg>')

dest = pathlib.Path(__file__).with_suffix(".svg")
dest.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {dest}  (firing rate {rate:.3f}, {sum(spk)}/{len(spk)} steps fired)")
