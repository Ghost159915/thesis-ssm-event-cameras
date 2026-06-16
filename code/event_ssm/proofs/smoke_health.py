"""Stage-6 health probes on the assembled YoloXDetector (drop-in Mamba-2 ResNetMamba backbone):
parameter-count breakdown, gradient flow, VRAM at batch sizes {1,2,4}, and eval single-window step
latency. Uses synthetic real-shaped clips (no data module needed).
Run (events_signals, CUDA): python proofs/smoke_health.py
"""
import sys, pathlib, time, torch
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))          # parents[2] == repo/code (so `event_ssm` imports)
from event_ssm.integration.smoke_harness import setup_paths, register, REPO
setup_paths(); register()
from omegaconf import OmegaConf
from models.detection.yolox_extension.models.detector import YoloXDetector

OUT = REPO / "results/smoke_test"; OUT.mkdir(parents=True, exist_ok=True)
RES = REPO / "results/stage6"; RES.mkdir(parents=True, exist_ok=True)
# Mirrors the Stage-4 integration proof config (the as-built drop-in contract), now Mamba-2:
# d_state=64, temporal only on FPN stages 2/3/4 (Finding S8 dropped the dead stage-1 block).
mcfg = OmegaConf.create({
    "backbone": {"name": "ResNetMamba", "input_channels": 20, "pretrained": False,
                 "d_state": 64, "num_layers_per_stage": 1, "in_stages": [2, 3, 4],
                 "compile": {"enable": False}},
    "fpn": {"name": "PAFPN", "depth": 0.67, "in_stages": [2, 3, 4],
            "depthwise": False, "act": "silu", "compile": {"enable": False}},
    "head": {"name": "YoloX", "num_classes": 2, "depthwise": False, "act": "silu",
             "compile": {"enable": False}},
})
model = YoloXDetector(mcfg).cuda()

# ---------- parameter-count breakdown ----------
# Mamba-1 (d_state=16, with a dead stage-1 temporal block) -> Mamba-2 (d_state=64, temporal on FPN
# stages 2/3/4 only). Bucket params by submodule; write results/stage6/params.md (Stage-6 delta vs
# the Stage-5 ~19.26M).
from collections import OrderedDict
pbuckets = OrderedDict()
for _n, _p in model.named_parameters():
    _top = _n.split(".")[0]
    if _top == "backbone":
        _sub = ("backbone.spatial" if ".spatial." in _n else
                "backbone.temporal" if ".temporal." in _n else "backbone.other")
    else:
        _sub = _top
    pbuckets[_sub] = pbuckets.get(_sub, 0) + _p.numel()
ptotal = sum(pbuckets.values())
plines = ["# Stage 6 - parameter counts (Mamba-2 backbone, d_state=64)", "",
          "Temporal blocks on FPN stages 2/3/4 only (Finding S8). Compare to Stage-5 (Mamba-1,",
          "d_state=16, *with* the now-removed dead stage-1 temporal block): ~19.26M total.", "",
          "| component | params | % of total |", "|---|---|---|"]
for _k, _v in pbuckets.items():
    plines.append(f"| {_k} | {_v/1e6:.3f}M | {100*_v/ptotal:.1f}% |")
plines.append(f"| **total** | **{ptotal/1e6:.3f}M** | 100% |")
(RES / "params.md").write_text("\n".join(plines) + "\n")
print("\n".join(plines)); print("wrote", RES / "params.md", "\n")


def _loss(losses):
    """YOLOX head returns dict with 'loss' = total; fall back defensively (matches proof_integration)."""
    if isinstance(losses, dict):
        lo = losses.get("loss")
        if lo is None:
            lo = sum(v for v in losses.values() if torch.is_tensor(v) and v.requires_grad)
        return lo
    return losses


def make_targets(B):
    """Multi-scale synthetic boxes (small/medium/large) at spread locations so SimOTA can assign a
    positive anchor on each FPN level (strides 8/16/32). [cls,cx,cy,w,h] in the 320x256 padded frame;
    cls in {0,1} (num_classes=2). A single box would leave the un-matched levels' positive-only head
    branches grad-less -- an assignment artifact, not a wiring fault."""
    boxes = torch.tensor([
        [0.,  44.,  44.,  18.,  18.],     # small  -> fine level (stride 8)
        [1., 160., 128.,  64.,  56.],     # medium -> stride 16
        [0., 250., 196., 150., 120.],     # large  -> coarse level (stride 32)
    ], device="cuda")
    return boxes.unsqueeze(0).expand(B, 3, 5).contiguous()


torch.manual_seed(0)                      # reproducible SimOTA assignment / random init
# SimOTA optimises cls + box only on POSITIVE (matched) anchors, so these three head branches can be
# grad-less on any FPN level that happens to get no positive match -- expected & data-dependent, NOT a
# wiring fault. The deterministic path that MUST always receive grad: the WHOLE backbone (temporal now
# built only on FPN stages 2/3/4 -- Finding S8 removed the dead stage-1 block, so there is nothing to
# exclude), the FPN, and the obj/reg-stem (objectness BCE covers every anchor).
HEAD_POS_ONLY = ("yolox_head.cls_convs.", "yolox_head.cls_preds.", "yolox_head.reg_preds.")

# ---------- grad-flow ----------
model.train()
tgt = make_targets(2)
x = torch.randn(5, 2, 20, 256, 320, device="cuda")
feats, _ = model.forward_backbone(x, previous_states=None, train_step=True)
sel = {k: v[-1] for k, v in feats.items() if k in (2, 3, 4)}     # last window: (B,c,h,w)
_, losses = model.forward_detect(backbone_features=sel, targets=tgt)
_loss(losses).backward()
real_missing, head_pos_missing, nan = [], [], []
n_pos_only = 0
for n_, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if n_.startswith(HEAD_POS_ONLY):
        n_pos_only += 1
    if p.grad is not None and not torch.isfinite(p.grad).all():
        nan.append(n_)
    if p.grad is None or p.grad.abs().max() == 0:
        (head_pos_missing if n_.startswith(HEAD_POS_ONLY) else real_missing).append(n_)
grad_ok = not real_missing and not nan      # positive-only head branches may legitimately be empty
# Robustness (not RNG-fragile): tolerate *some* positive-only branches being grad-less on an
# unmatched FPN level, but require at least one to be alive -- if EVERY positive-only branch is
# grad-less, the cls/box positive path is genuinely broken (no level ever matched), which IS a fault.
pos_path_alive = len(head_pos_missing) < n_pos_only
model.zero_grad(set_to_none=True)

# ---------- VRAM sweep ----------
vram = {}
for B in (1, 2, 4):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    xb = torch.randn(5, B, 20, 256, 320, device="cuda")
    fb, _ = model.forward_backbone(xb, previous_states=None, train_step=True)
    s = {k: v[-1] for k, v in fb.items() if k in (2, 3, 4)}
    _, lo = model.forward_detect(backbone_features=s, targets=make_targets(B))
    _loss(lo).backward()
    vram[B] = torch.cuda.max_memory_allocated() / 1024 ** 3
    model.zero_grad(set_to_none=True)

# ---------- eval single-window step latency ----------
model.eval()
xw = torch.randn(1, 1, 20, 256, 320, device="cuda")
st = None
with torch.no_grad():
    for _ in range(20):                         # warmup
        f, st = model.forward_backbone(xw, previous_states=st, train_step=False)
    torch.cuda.synchronize()
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        f, st = model.forward_backbone(xw, previous_states=st, train_step=False)
        sd = {k: v[-1] for k, v in f.items() if k in (2, 3, 4)}
        _ = model.forward_detect(backbone_features=sd)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
mean = sum(ts) / len(ts)
std = (sum((t - mean) ** 2 for t in ts) / len(ts)) ** 0.5

lines = [
    "# Stage 6 - smoke health probes (Mamba-2 backbone, d_state=64)", "",
    "Synthetic real-shaped clips through the assembled `YoloXDetector` (drop-in `ResNetMamba` +",
    "PAFPN + YOLOX head). Temporal blocks are built ONLY on FPN stages 2/3/4 (Finding S8 removed the",
    "dead stage-1 block), so the deterministic grad-flow path spans the WHOLE backbone -- no exclusion.", "",
    "| test | result | notes |", "|---|---|---|",
    f"| gradient flow (deterministic path) | {'PASS' if grad_ok else 'FAIL'} | "
    f"real-missing={len(real_missing)} nan={len(nan)} (whole backbone, incl. temporal 2/3/4) |",
    f"| SimOTA pos-only head branches w/o grad | {len(head_pos_missing)}/{n_pos_only} | "
    f"expected (data-dependent per-level matching); >=1 alive ({'PASS' if pos_path_alive else 'FAIL'}) |",
    f"| VRAM @ bs1/bs2/bs4 | {vram[1]:.2f}/{vram[2]:.2f}/{vram[4]:.2f} GB | target <10GB @ bs4 (d_state=64 > S5) |",
    f"| eval step latency | {mean:.2f} +/- {std:.2f} ms | S5-RVT ~12 ms/window |",
    f"| max throughput | {1000/mean:.0f} Hz | window dt=50ms -> need <50ms |",
]
if real_missing: lines.append(f"\nREAL MISSING GRAD ({len(real_missing)}): " + ", ".join(real_missing[:10]))
if head_pos_missing: lines.append(f"\nSimOTA pos-only head w/o grad ({len(head_pos_missing)}): "
                                  + ", ".join(head_pos_missing[:10]))
if nan: lines.append(f"\nNAN GRAD ({len(nan)}): " + ", ".join(nan[:10]))
(OUT / "smoke_results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print("wrote", OUT / "smoke_results.md")
assert grad_ok, f"grad-flow FAIL (deterministic path): real_missing={real_missing[:5]} nan={nan[:5]}"
assert pos_path_alive, (f"SimOTA positive path broken: all {n_pos_only} positive-only head branches "
                        f"grad-less (no FPN level matched a positive)")
assert vram[4] < 10.0, f"VRAM @ bs4 = {vram[4]:.2f}GB exceeds 10GB"
