"""Stage 4 proof: the unmodified RVT YoloXDetector, built with our drop-in ResNetMamba backbone,
assembles and runs one TRAIN step + one EVAL step (with RVT-style state detach/reset) on a
synthetic (L,B,20,256,320) clip. Writes a shape/loss/param table to proofs/out/u4_integration.md.

Run (events_signals env, CUDA): python proofs/proof_integration.py
"""
import sys, pathlib, torch
REPO = pathlib.Path(__file__).resolve().parents[3]   # proofs/ -> event_ssm/ -> code/ -> repo root
for p in (REPO / "code", REPO / "external/ssms_event_cameras/RVT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from omegaconf import OmegaConf
from event_ssm.integration.register import register_resnet_mamba
register_resnet_mamba()
from models.detection.yolox_extension.models.detector import YoloXDetector
from modules.utils.detection import RNNStates

OUT = pathlib.Path(__file__).parent / "out"; OUT.mkdir(parents=True, exist_ok=True)

# num_classes=2 mirrors config/modifier.py for gen1 (YOLOXHead default is 80).
model_cfg = OmegaConf.create({
    "backbone": {"name": "ResNetMamba", "input_channels": 20, "pretrained": False,
                 "d_state": 16, "num_layers_per_stage": 1,
                 "compile": {"enable": False}},
    "fpn": {"name": "PAFPN", "depth": 0.67, "in_stages": [2, 3, 4],
            "depthwise": False, "act": "silu", "compile": {"enable": False}},
    "head": {"name": "YoloX", "num_classes": 2, "depthwise": False, "act": "silu",
             "compile": {"enable": False}},
})
model = YoloXDetector(model_cfg).cuda()
n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
n_total, n_bb, n_fpn, n_head = n(model), n(model.backbone), n(model.fpn), n(model.yolox_head)
assert 15 < n_total < 30, f"param count {n_total:.1f}M out of range"

L, B = 5, 2
x = torch.randn(L, B, 20, 256, 320, device="cuda")

# ---- TRAIN step ----
model.train()
feats, _ = model.forward_backbone(x, previous_states=None, train_step=True)
assert feats[2].shape == (L, B, 128, 32, 40), feats[2].shape
sel = {k: v[-1] for k, v in feats.items() if k in (2, 3, 4)}      # last window: (B,c,h,w)
targets = torch.zeros(B, 3, 5, device="cuda")                     # (B, max_objs, 5)=[cls,cx,cy,w,h]
targets[:, 0] = torch.tensor([0., 160., 128., 40., 30.], device="cuda")  # one synthetic box/image
_, losses = model.forward_detect(backbone_features=sel, targets=targets)
# YOLOX head returns dict with "loss" = total (iou + conf + cls [+ l1]); fall back defensively.
if isinstance(losses, dict):
    loss = losses.get("loss")
    if loss is None:
        loss = sum(v for v in losses.values() if torch.is_tensor(v) and v.requires_grad)
else:
    loss = losses
loss.backward()
assert torch.isfinite(loss).all()
# Grad-flow health: every backbone grad that exists must be finite (no NaN/Inf), and the bulk of
# the backbone must be in the detection path. NOTE: temporal[0] (stage-1 Mamba) legitimately gets
# NO grad -- the FPN's in_stages=[2,3,4] drops feats[1], so the stage-1 temporal side-branch is
# unused (a known divergence from RVT, where the stage-1 recurrent module feeds forward). See the
# Stage 4 report. We therefore check finiteness-where-present + a coverage fraction, not all-present.
bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
present = [p.grad is not None for p in bb_params]
finite = [bool(torch.isfinite(p.grad).all()) for p in bb_params if p.grad is not None]
cover = sum(present) / len(present)
# Tight bound: only the documented stage-1 temporal side-branch (~9% of params) is legitimately
# grad-less, so healthy coverage is ~0.91. A regression that disconnected any FPN-fed stage would
# drop well below 0.85 -> caught. (A loose 0.5 would miss an entire-stage break.)
assert all(finite) and cover > 0.85, f"grad-flow unhealthy: cover={cover:.0%}, all-finite={all(finite)}"

# ---- EVAL step (carry + reset state, RVT-style) ----
model.eval()
with torch.no_grad():
    _, st1 = model.forward_backbone(x, previous_states=None, train_step=False)
    st1 = RNNStates.recursive_detach(st1)
    feats_e, st2 = model.forward_backbone(x, previous_states=st1, train_step=False)
    RNNStates.recursive_reset(RNNStates.recursive_detach(st2), indices_or_bool_tensor=[0])
    sel_e = {k: v[-1] for k, v in feats_e.items() if k in (2, 3, 4)}
    out_e, _ = model.forward_detect(backbone_features=sel_e)
assert out_e.shape[-1] == 7 and out_e.shape[1] == 32*40 + 16*20 + 8*10, out_e.shape   # 1680

lines = [
    "# Stage 4 - full-model integration proof", "",
    "Unmodified RVT `YoloXDetector` (PAFPN + YOLOX head + SimOTA losses) assembled with the",
    "drop-in `ResNetMamba` backbone; one synthetic train step (+backward) and one eval step",
    "(with RVT `RNNStates` detach/reset) on a `(L=5, B=2, 20, 256, 320)` clip.", "",
    "| metric | value |", "|---|---|",
    f"| params total | {n_total:.2f} M |",
    f"| params backbone / fpn / head | {n_bb:.2f} / {n_fpn:.2f} / {n_head:.2f} M |",
    f"| train feats[2] (L,B,c,h,w) | {tuple(feats[2].shape)} |",
    f"| train loss (finite) | {loss.item():.4f} |",
    f"| backbone grad coverage | {cover:.0%} (all finite; stage-1 temporal unused by FPN) |",
    f"| eval output (B, anchors, 5+ncls) | {tuple(out_e.shape)} |",
    "| train step | PASS |",
    "| eval step + state detach/reset | PASS |",
]
(OUT / "u4_integration.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print("wrote", OUT / "u4_integration.md")
