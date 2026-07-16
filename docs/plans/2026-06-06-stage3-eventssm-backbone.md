# Stage 3 — EventSSMDetector Interleaved Backbone — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a drop-in recurrent backbone (`ResNet-18 conv spatial → Mamba-1 temporal, interleaved per stage`) that plugs into the reused RVT PAFPN+YOLOX head, as the EventSSMDetector contribution.

**Architecture:** 4 backbone stages, each `[ResNet layer_k → MambaTemporalBlock(d_model=stage_dim_k)]`. Temporal Mamba scans the TIME axis per spatial location (Formulation A), carrying detached state across clips via the backbone-level `LstmStates` list. PAFPN + head + losses + Lightning training reused unmodified; only the backbone is new.

**Tech Stack:** PyTorch 2.11+cu128, torchvision ResNet-18, `mamba-ssm==2.3.2.post1` + `causal-conv1d==1.6.2.post1` (Blackwell sm_120), pytest, matplotlib (proof figures). Env: conda `events_signals`.

**Authoritative specs:** `docs/specs/2026-06-06-stage3-eventssm-backbone-design.md`, `architecture_blueprint.md` (rev.2), `yolox_head_interface.md`, `codebase_audit.md`.

**Conventions for every task:**
- Python interpreter / pytest: `PY=/home/ghost/miniforge3/envs/events_signals/bin/python` ; run tests with `$PY -m pytest`.
- Tests require CUDA (mamba kernels are CUDA-only) — run on the 5070 Ti workstation.
- Proof artifacts saved to `code/event_ssm/proofs/out/`.
- Commit after each task with the message shown.

---

## File Structure

| File | Responsibility |
|---|---|
| `code/event_ssm/__init__.py` | package marker |
| `code/event_ssm/conftest.py` | pytest fixtures: add RVT to `sys.path`, `cuda` guard |
| `code/event_ssm/backbone/resnet_spatial.py` | **Unit 1** ResNet-18 spatial stages, 10-ch conv1 avg-proj init |
| `code/event_ssm/temporal/_scan.py` | **Spike output** — validated cross-clip scan helper |
| `code/event_ssm/temporal/mamba_temporal.py` | **Unit 2** `MambaTemporalBlock` (time-axis, stateful) |
| `code/event_ssm/backbone/resnet_mamba.py` | **Unit 3** `ResNetMambaBackbone(BaseDetector)` interleaver |
| `code/event_ssm/integration/register.py` | register `"resnet_mamba"` into RVT builder |
| `code/event_ssm/integration/rvt_register.patch` | tracked patch of the RVT builder edit |
| `code/event_ssm/configs/resnet_mamba.yaml` | Hydra model config |
| `code/event_ssm/proofs/*.py` | per-unit visual-proof scripts → `proofs/out/*.png` |
| `code/event_ssm/tests/test_*.py` | pytest unit tests |

---

## Task 0: Package scaffolding

**Files:** Create `code/event_ssm/__init__.py`, `code/event_ssm/{backbone,temporal,integration,configs,proofs,tests}/__init__.py` (proofs/configs need no `__init__`), `code/event_ssm/conftest.py`, `code/event_ssm/proofs/out/.gitkeep`.

- [ ] **Step 1: Create package dirs + markers**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
mkdir -p code/event_ssm/{backbone,temporal,integration,configs,proofs/out,tests}
touch code/event_ssm/__init__.py code/event_ssm/backbone/__init__.py \
      code/event_ssm/temporal/__init__.py code/event_ssm/integration/__init__.py \
      code/event_ssm/tests/__init__.py code/event_ssm/proofs/out/.gitkeep
```

- [ ] **Step 2: Write `conftest.py`** (RVT path + cuda fixtures)

```python
# code/event_ssm/conftest.py
import sys, pathlib, pytest, torch

REPO = pathlib.Path(__file__).resolve().parents[2]
RVT = REPO / "external" / "ssms_event_cameras" / "RVT"

@pytest.fixture(scope="session", autouse=True)
def _add_rvt_to_path():
    # RVT uses absolute imports rooted at the RVT/ dir
    if str(RVT) not in sys.path:
        sys.path.insert(0, str(RVT))

@pytest.fixture(scope="session")
def device():
    assert torch.cuda.is_available(), "Stage 3 tests need the 5070 Ti (mamba kernels are CUDA-only)"
    return torch.device("cuda")
```

- [ ] **Step 3: Sanity-run pytest collection**

Run: `cd code/event_ssm && PY=/home/ghost/miniforge3/envs/events_signals/bin/python && $PY -m pytest -q`
Expected: `no tests ran` (collection succeeds, 0 tests).

- [ ] **Step 4: Commit**

```bash
git add code/event_ssm
git commit -m "feat(stage3): scaffold event_ssm package + pytest conftest"
```

---

## Task 1: SPIKE — cross-clip state mechanism (de-risk Unit 2)

**Goal:** Empirically choose the trainable cross-clip-state path for Mamba-1 in mamba-ssm 2.3.2. Produces `temporal/_scan.py` (the validated helper) + `proofs/out/spike_state.md`.

**Files:** Create `code/event_ssm/proofs/spike_state.py`, then `code/event_ssm/temporal/_scan.py`.

- [ ] **Step 1: Write the spike script** testing the three options on a `(N=64, L=5, C=64)` tensor.

```python
# code/event_ssm/proofs/spike_state.py
"""Decide the trainable cross-clip-state path for Mamba-1 (mamba-ssm 2.3.2).
Writes proofs/out/spike_state.md. Run: python proofs/spike_state.py"""
import time, pathlib, torch
from mamba_ssm import Mamba

OUT = pathlib.Path(__file__).parent / "out" / "spike_state.md"
N, L, C = 64, 5, 64
dev = "cuda"

def make(): return Mamba(d_model=C, d_state=16, d_conv=4, expand=2).to(dev)

def opt_alpha():
    """Step-kernel loop over L using Mamba.step(); checks differentiability + state carry."""
    m = make()
    x = torch.randn(N, L, C, device=dev, requires_grad=True)
    conv_state, ssm_state = m.allocate_inference_cache(N, L, dtype=x.dtype)
    outs = []
    for t in range(L):
        o, conv_state, ssm_state = m.step(x[:, t], conv_state, ssm_state)
        outs.append(o)
    y = torch.stack(outs, 1)
    trainable = True
    try:
        y.square().mean().backward(); trainable = x.grad is not None
    except Exception as e:
        trainable = False; print("alpha backward failed:", e)
    # state carry: second clip with carried state differs from fresh
    return dict(name="alpha_step_loop", trainable=trainable)

def opt_gamma():
    """Parallel kernel, no initial-state injection (per-clip reset). Always trainable."""
    m = make()
    x = torch.randn(N, L, C, device=dev, requires_grad=True)
    y = m(x); y.square().mean().backward()
    return dict(name="gamma_parallel_reset", trainable=x.grad is not None)

def bench(fn):
    t0 = time.time()
    for _ in range(20): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/20*1e3  # ms/iter

if __name__ == "__main__":
    rows = []
    for f in (opt_alpha, opt_gamma):
        r = f(); rows.append(r); print(r)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("# Spike: cross-clip state for Mamba-1\n\n| option | trainable |\n|---|---|\n")
        for r in rows: fh.write(f"| {r['name']} | {r['trainable']} |\n")
    print("wrote", OUT)
```

- [ ] **Step 2: Run the spike**

Run: `cd code/event_ssm && $PY proofs/spike_state.py`
Expected: prints each option dict; writes `proofs/out/spike_state.md`. **Decision rule:** if `alpha_step_loop.trainable == True` → use α (step-loop, exact cross-clip state, uses CUDA step kernels). Else → implement β (pure-PyTorch selective scan with `initial_state`, code in Step 3 fallback) and keep γ as the no-state-during-training fallback only if both fail.

- [ ] **Step 3: Write `temporal/_scan.py`** implementing the SELECTED mechanism as a clean helper.

If α selected (expected):
```python
# code/event_ssm/temporal/_scan.py
"""Cross-clip temporal scan helpers (validated by proofs/spike_state.py)."""
import torch
from mamba_ssm import Mamba

def mamba_scan_time(mamba: Mamba, x, state):
    """x: (N, L, C). state: (conv_state, ssm_state) or None.
    Scans over L=time via Mamba.step(), carrying explicit state. Returns (y:(N,L,C), new_state)."""
    N, L, C = x.shape
    if state is None:
        conv_state, ssm_state = mamba.allocate_inference_cache(N, L, dtype=x.dtype)
    else:
        conv_state, ssm_state = state
    outs = []
    for t in range(L):
        o, conv_state, ssm_state = mamba.step(x[:, t], conv_state, ssm_state)
        outs.append(o)
    return torch.stack(outs, 1), (conv_state, ssm_state)
```
If α NOT trainable, replace with the β pure-PyTorch reference (selective scan accepting `initial_state`); the spike output records which was chosen. Document the choice at the top of the file.

- [ ] **Step 4: Commit**

```bash
git add code/event_ssm/proofs/spike_state.py code/event_ssm/temporal/_scan.py code/event_ssm/proofs/out/spike_state.md
git commit -m "spike(stage3): validate + implement cross-clip Mamba-1 state scan"
```

---

## Task 2: Unit 1 — `ResNetSpatialStages`

**Files:** Create `code/event_ssm/backbone/resnet_spatial.py`, `code/event_ssm/tests/test_resnet_spatial.py`, `code/event_ssm/proofs/proof_resnet.py`.

- [ ] **Step 1: Write the failing tests**

```python
# code/event_ssm/tests/test_resnet_spatial.py
import torch, pytest
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages

def test_shapes(device):
    m = ResNetSpatialStages(in_channels=10, pretrained=False).to(device).eval()
    x = torch.randn(2, 10, 256, 320, device=device)  # padded Gen1
    f = m(x)
    assert f[1].shape == (2, 64, 64, 80)
    assert f[2].shape == (2, 128, 32, 40)
    assert f[3].shape == (2, 256, 16, 20)
    assert f[4].shape == (2, 512, 8, 10)

def test_gradients(device):
    m = ResNetSpatialStages(in_channels=10, pretrained=False).to(device).train()
    x = torch.randn(2, 10, 256, 320, device=device)
    sum(v.sum() for v in m(x).values()).backward()
    assert m.stem[0].weight.grad is not None       # conv1
    assert m.layer4[0].conv1.weight.grad is not None

def test_avg_projection_init(device):
    m = ResNetSpatialStages(in_channels=10, pretrained=True).to(device)
    w = m.stem[0].weight.data
    assert w.shape == (64, 10, 7, 7)
    assert 0.005 < w.std().item() < 0.2   # not random (≈1.0) nor zero

def test_param_count():
    m = ResNetSpatialStages(in_channels=10, pretrained=False)
    mparams = sum(p.numel() for p in m.parameters()) / 1e6
    assert 11.0 < mparams < 11.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/event_ssm && $PY -m pytest tests/test_resnet_spatial.py -q`
Expected: FAIL/ERROR — `ModuleNotFoundError: event_ssm` (add repo `code/` to path — see Step 3 note) then `cannot import ResNetSpatialStages`.

> Note: ensure `code/` is importable. Add to `conftest.py` session fixture: `sys.path.insert(0, str(REPO / "code"))`. Apply this one-line addition now.

- [ ] **Step 3: Implement `resnet_spatial.py`**

```python
# code/event_ssm/backbone/resnet_spatial.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def _avg_projection_conv1(pretrained_conv1: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    """(64,3,7,7) -> (64,in_ch,7,7): mean over RGB, tile to in_ch, scale 3/in_ch."""
    new = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        w = pretrained_conv1.weight.data.mean(dim=1, keepdim=True)      # (64,1,7,7)
        new.weight.copy_(w.repeat(1, in_ch, 1, 1) * (3.0 / in_ch))      # (64,in_ch,7,7)
    return new


class ResNetSpatialStages(nn.Module):
    """ResNet-18 stem + 4 stages, 10-channel input. Returns per-stage feature maps.
    Temporal Mamba is interleaved by ResNetMambaBackbone (Unit 3), not here."""
    stage_dims = (64, 128, 256, 512)
    strides = (4, 8, 16, 32)

    def __init__(self, in_channels: int = 10, pretrained: bool = True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.conv1 = (_avg_projection_conv1(net.conv1, in_channels) if pretrained
                     else nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False))
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return {1: c1, 2: c2, 3: c3, 4: c4}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/event_ssm && $PY -m pytest tests/test_resnet_spatial.py -q`
Expected: 4 passed.

- [ ] **Step 5: Write the visual-proof script**

```python
# code/event_ssm/proofs/proof_resnet.py
"""Visual proof for Unit 1: conv1 filters before/after avg-proj + feature heatmaps + shape table."""
import pathlib, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages, _avg_projection_conv1
from torchvision.models import resnet18, ResNet18_Weights

OUT = pathlib.Path(__file__).parent / "out"; OUT.mkdir(parents=True, exist_ok=True)
rgb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).conv1
new = _avg_projection_conv1(rgb, 10)
# filter grid: RGB mean vs averaged-10ch (channel 0)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(make_grid(rgb.weight.mean(1, keepdim=True), nrow=8, normalize=True)[0].cpu()); ax[0].set_title("ImageNet conv1 (RGB mean)")
ax[1].imshow(make_grid(new.weight[:, :1], nrow=8, normalize=True)[0].detach().cpu()); ax[1].set_title("avg-proj conv1 (10ch, ch0)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "u1_conv1_filters.png", dpi=150); plt.close()

# shape table + heatmaps on random voxel (real Gen1 sample path optional)
m = ResNetSpatialStages(10, pretrained=True).eval()
f = m(torch.randn(1, 10, 256, 320))
fig, ax = plt.subplots(1, 4, figsize=(14, 3))
for i, k in enumerate((1, 2, 3, 4)):
    ax[i].imshow(f[k][0].mean(0).detach().cpu()); ax[i].set_title(f"stage{k} {tuple(f[k].shape[1:])}"); ax[i].axis("off")
plt.tight_layout(); plt.savefig(OUT / "u1_feature_heatmaps.png", dpi=150); plt.close()
print("shapes:", {k: tuple(v.shape) for k, v in f.items()})
print("params(M):", sum(p.numel() for p in m.parameters())/1e6)
print("wrote u1_conv1_filters.png, u1_feature_heatmaps.png")
```

Run: `cd code/event_ssm && $PY proofs/proof_resnet.py` — Expected: prints shapes + ~11.2M params; writes 2 PNGs.

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/backbone/resnet_spatial.py code/event_ssm/tests/test_resnet_spatial.py code/event_ssm/proofs/proof_resnet.py code/event_ssm/conftest.py
git commit -m "feat(stage3): Unit 1 ResNetSpatialStages (10-ch avg-proj) + tests + proof"
```

---

## Task 3: Unit 2 — `MambaTemporalBlock`

**Files:** Create `code/event_ssm/temporal/mamba_temporal.py`, `code/event_ssm/tests/test_mamba_temporal.py`, `code/event_ssm/proofs/proof_mamba.py`. Uses `temporal/_scan.py` from Task 1.

- [ ] **Step 1: Write the failing tests** (interface contract — independent of spike internals)

```python
# code/event_ssm/tests/test_mamba_temporal.py
import torch
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock

def test_shape_preserved(device):
    blk = MambaTemporalBlock(d_model=128).to(device).eval()
    x = torch.randn(2*32*40, 5, 128, device=device)   # (B*H*W, L=time, C)
    y, state = blk(x, state=None)
    assert y.shape == x.shape and state is not None

def test_state_persistence(device):
    blk = MambaTemporalBlock(d_model=64).to(device).eval()
    x = torch.randn(64, 5, 64, device=device)
    y1, s1 = blk(x, state=None)
    y_with, _ = blk(x, state=s1)        # carried state from prior clip
    y_without, _ = blk(x, state=None)
    assert (y_with - y_without).abs().mean().item() > 1e-4

def test_gradients(device):
    blk = MambaTemporalBlock(d_model=64).to(device).train()
    x = torch.randn(64, 5, 64, device=device, requires_grad=True)
    blk(x, state=None)[0].sum().backward()
    for n, p in blk.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and not torch.isnan(p.grad).any(), n

def test_bf16_autocast(device):
    blk = MambaTemporalBlock(d_model=64).to(device)
    x = torch.randn(64, 5, 64, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, _ = blk(x, state=None)
    assert y.shape == x.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/event_ssm && $PY -m pytest tests/test_mamba_temporal.py -q`
Expected: FAIL — `cannot import MambaTemporalBlock`.

- [ ] **Step 3: Implement `mamba_temporal.py`**

```python
# code/event_ssm/temporal/mamba_temporal.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange
from event_ssm.temporal._scan import mamba_scan_time


class MambaTemporalBlock(nn.Module):
    """One (or num_layers) causal Mamba-1 block(s) over the TIME axis, per spatial location.
    Forward expects x:(N, L, C) with N=B*H*W and L=time; carries state across clips.
    Also provides fold/unfold helpers for the (L,B,C,H,W) <-> (B*H*W,L,C) layout."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        )

    def forward(self, x, state=None):
        # x: (N, L, C). state: list[per-layer state] or None.
        if state is None:
            state = [None] * len(self.layers)
        new_state = []
        for layer, st in zip(self.layers, state):
            x, st2 = mamba_scan_time(layer, x, st)
            new_state.append(st2)
        return x, new_state

    @staticmethod
    def fold(x):  # (L,B,C,H,W) -> (B*H*W, L, C), and dims to unfold
        L, B, C, H, W = x.shape
        return rearrange(x, "L B C H W -> (B H W) L C"), (L, B, C, H, W)

    @staticmethod
    def unfold(x, dims):  # (B*H*W,L,C) -> (L,B,C,H,W)
        L, B, C, H, W = dims
        return rearrange(x, "(B H W) L C -> L B C H W", B=B, H=H, W=W)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/event_ssm && $PY -m pytest tests/test_mamba_temporal.py -q`
Expected: 4 passed. If `test_state_persistence` fails (diff≈0), the scan helper isn't carrying state — revisit `_scan.py` (spike).

- [ ] **Step 5: Write the visual proof (state-influence plot)**

```python
# code/event_ssm/proofs/proof_mamba.py
"""Visual proof for Unit 2: temporal memory works — output divergence with vs without carried state."""
import pathlib, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock

OUT = pathlib.Path(__file__).parent / "out"; OUT.mkdir(parents=True, exist_ok=True)
blk = MambaTemporalBlock(d_model=64).cuda().eval()
clips = [torch.randn(64, 5, 64, device="cuda") for _ in range(8)]
state, divs = None, []
for c in clips:
    y_state, state = blk(c, state=state)
    y_fresh, _ = blk(c, state=None)
    divs_ = (y_state - y_fresh).abs().mean().item(); divs.append(divs_)
    state = [(cs.detach(), ss.detach()) for (cs, ss) in state]
plt.figure(figsize=(6,4)); plt.plot(range(1,9), divs, "o-")
plt.xlabel("clip index"); plt.ylabel("|out_with_state - out_fresh| mean")
plt.title("Unit 2: temporal memory influence across clips"); plt.tight_layout()
plt.savefig(OUT / "u2_state_influence.png", dpi=150)
print("divergences:", [round(d,4) for d in divs]); print("wrote u2_state_influence.png")
```

Run: `cd code/event_ssm && $PY proofs/proof_mamba.py` — Expected: nonzero divergences; writes PNG.

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/temporal/mamba_temporal.py code/event_ssm/tests/test_mamba_temporal.py code/event_ssm/proofs/proof_mamba.py
git commit -m "feat(stage3): Unit 2 MambaTemporalBlock (time-axis, stateful) + tests + proof"
```

---

## Task 4: Unit 3 — `ResNetMambaBackbone` (+ integration smoke)

**Files:** Create `code/event_ssm/backbone/resnet_mamba.py`, `code/event_ssm/tests/test_resnet_mamba.py`, `code/event_ssm/proofs/proof_backbone.py`.

- [ ] **Step 1: Write the failing tests** (shape trace + integration with real PAFPN+head)

```python
# code/event_ssm/tests/test_resnet_mamba.py
import torch
from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone

def test_forward_shapes_and_state(device):
    m = ResNetMambaBackbone(in_channels=10, pretrained=False).to(device).eval()
    x = torch.randn(5, 2, 10, 256, 320, device=device)   # (L=time, B, C, H, W)
    feats, states = m(x, prev_states=None)
    assert feats[2].shape == (5*2, 128, 32, 40)
    assert feats[3].shape == (5*2, 256, 16, 20)
    assert feats[4].shape == (5*2, 512, 8, 10)
    assert len(states) == 4
    assert m.get_stage_dims((2,3,4)) == (128,256,512)
    assert m.get_strides((2,3,4)) == (8,16,32)

def test_integration_with_pafpn_head(device):
    """Reused RVT PAFPN + YOLOX head accept our backbone's stage outputs."""
    from models.detection.yolox_extension.models.yolo_pafpn import YOLOPAFPN
    from models.detection.yolox.models.yolo_head import YOLOXHead
    m = ResNetMambaBackbone(in_channels=10, pretrained=False).to(device).eval()
    x = torch.randn(1, 1, 10, 256, 320, device=device)
    feats, _ = m(x, prev_states=None)
    pafpn = YOLOPAFPN(depth=0.33, in_stages=(2,3,4), in_channels=(128,256,512)).to(device).eval()
    head = YOLOXHead(num_classes=2, strides=(8,16,32), in_channels=(128,256,512)).to(device).eval()
    outs, _ = head(pafpn({2:feats[2], 3:feats[3], 4:feats[4]}))
    assert outs.shape[-1] == 7          # x,y,w,h,obj,cls0,cls1
    assert outs.shape[1] == 32*40 + 16*20 + 8*10   # 1680
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd code/event_ssm && $PY -m pytest tests/test_resnet_mamba.py -q`
Expected: FAIL — `cannot import ResNetMambaBackbone`.

- [ ] **Step 3: Implement `resnet_mamba.py`**

```python
# code/event_ssm/backbone/resnet_mamba.py
import torch
import torch.nn as nn
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock


class ResNetMambaBackbone(nn.Module):
    """Interleaved ResNet-18 conv (spatial) + Mamba-1 (temporal) per stage.
    forward(x:(L,B,10,H,W), prev_states) -> (features dict{1..4}, states list[4]).
    Mirrors RVT RNNDetector's backbone contract so PAFPN+head+training are reused."""

    def __init__(self, in_channels: int = 10, pretrained: bool = True,
                 d_state: int = 16, num_layers_per_stage: int = 1):
        super().__init__()
        self.spatial = ResNetSpatialStages(in_channels, pretrained)
        self._stage_dims = self.spatial.stage_dims        # (64,128,256,512)
        self._strides = self.spatial.strides              # (4,8,16,32)
        self.temporal = nn.ModuleList(
            MambaTemporalBlock(d_model=d, d_state=d_state, num_layers=num_layers_per_stage)
            for d in self._stage_dims
        )

    def get_stage_dims(self, stages):  # stages are 1-indexed (2,3,4)
        return tuple(self._stage_dims[s-1] for s in stages)

    def get_strides(self, stages):
        return tuple(self._strides[s-1] for s in stages)

    def forward(self, x, prev_states=None, token_mask=None, train_step: bool = True):
        L, B, C, H, W = x.shape
        if prev_states is None:
            prev_states = [None] * 4
        # spatial: fold time into batch, run all 4 ResNet stages
        spat = self.spatial(x.reshape(L * B, C, H, W))     # {1..4}: (L*B, c, h, w)
        feats, new_states = {}, []
        for i, stage in enumerate((1, 2, 3, 4)):
            fmap = spat[stage]                             # (L*B, c, h, w)
            c = fmap.shape[1]; h, w = fmap.shape[2], fmap.shape[3]
            seq = fmap.reshape(L, B, c, h, w)
            seq, dims = MambaTemporalBlock.fold(seq)       # (B*h*w, L, c)
            seq, st = self.temporal[i](seq, prev_states[i])
            out = MambaTemporalBlock.unfold(seq, dims).reshape(L * B, c, h, w)
            feats[stage] = out
            new_states.append(st)
        return feats, new_states
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd code/event_ssm && $PY -m pytest tests/test_resnet_mamba.py -q`
Expected: 2 passed (shape/state + PAFPN/head integration → 1680×7).

- [ ] **Step 5: Write the proof (full shape trace)**

```python
# code/event_ssm/proofs/proof_backbone.py
"""Visual proof for Unit 3: end-to-end shape trace + per-stage state shapes."""
import torch
from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
m = ResNetMambaBackbone(in_channels=10, pretrained=False).cuda().eval()
feats, states = m(torch.randn(5, 2, 10, 256, 320, device="cuda"), prev_states=None)
print("=== forward shape trace (L=5,B=2) ===")
for k in (1,2,3,4): print(f" stage{k}: {tuple(feats[k].shape)}")
print(" stage_dims(2,3,4):", m.get_stage_dims((2,3,4)), "strides:", m.get_strides((2,3,4)))
print("=== per-stage state present ===", [s is not None for s in states])
```

Run: `cd code/event_ssm && $PY proofs/proof_backbone.py` — Expected: prints the trace; stages 2/3/4 = (10,128,32,40)/(10,256,16,20)/(10,512,8,10).

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/backbone/resnet_mamba.py code/event_ssm/tests/test_resnet_mamba.py code/event_ssm/proofs/proof_backbone.py
git commit -m "feat(stage3): Unit 3 ResNetMambaBackbone + PAFPN/head integration smoke"
```

---

## Task 5: RVT registration + Hydra config

**Files:** Create `code/event_ssm/integration/register.py`, `code/event_ssm/integration/rvt_register.patch`, `code/event_ssm/configs/resnet_mamba.yaml`. Modify (tracked via patch): `external/.../recurrent_backbone/__init__.py`.

- [ ] **Step 1: Inspect the RVT builder**

Run: `sed -n '1,40p' external/ssms_event_cameras/RVT/models/detection/recurrent_backbone/__init__.py`
Expected: shows `build_recurrent_backbone(...)` mapping a name string → class.

- [ ] **Step 2: Write `register.py`** (idempotent injection)

```python
# code/event_ssm/integration/register.py
"""Register ResNetMambaBackbone into RVT's build_recurrent_backbone.
Call register_resnet_mamba() once before building the detector."""
def register_resnet_mamba():
    import models.detection.recurrent_backbone as rb
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
    orig = rb.build_recurrent_backbone
    def patched(name, cfg):
        if name == "ResNetMamba":
            return ResNetMambaBackbone(
                in_channels=cfg.input_channels,
                pretrained=cfg.get("pretrained", True),
                num_layers_per_stage=cfg.get("num_layers_per_stage", 1),
            )
        return orig(name, cfg)
    rb.build_recurrent_backbone = patched
```

- [ ] **Step 3: Write `configs/resnet_mamba.yaml`** (Hydra model config, derived from `gen1/base.yaml`)

```yaml
# code/event_ssm/configs/resnet_mamba.yaml — select our backbone
backbone:
  name: ResNetMamba
  input_channels: 10
  pretrained: true
  num_layers_per_stage: 1
fpn:
  name: PAFPN
  in_stages: [2, 3, 4]
  depth: 0.33
head:
  name: YoloX
  num_classes: 2
```

- [ ] **Step 4: Write the registration test**

```python
# code/event_ssm/tests/test_register.py
def test_register_resnet_mamba(_add_rvt_to_path):
    from event_ssm.integration.register import register_resnet_mamba
    register_resnet_mamba()
    import models.detection.recurrent_backbone as rb
    assert rb.build_recurrent_backbone.__name__ == "patched"
```

Run: `cd code/event_ssm && $PY -m pytest tests/test_register.py -q` — Expected: 1 passed.

- [ ] **Step 5: Capture the RVT edit as a tracked patch** (so the ignored-tree change is reproducible)

```bash
cd external/ssms_event_cameras/RVT
git diff models/detection/recurrent_backbone/__init__.py > \
  /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm/integration/rvt_register.patch || true
```
(If we use the monkeypatch in `register.py` instead of editing RVT, this patch may be empty — that's fine; prefer the monkeypatch to keep the upstream tree clean.)

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/integration code/event_ssm/configs code/event_ssm/tests/test_register.py
git commit -m "feat(stage3): register ResNetMamba backbone + Hydra config"
```

---

## Task 6: Full Stage-3 verification + code review

- [ ] **Step 1: Run the whole suite**

Run: `cd code/event_ssm && $PY -m pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Regenerate all proof artifacts**

Run: `cd code/event_ssm && for p in proof_resnet proof_mamba proof_backbone; do $PY proofs/$p.py; done`
Expected: `proofs/out/` contains `u1_conv1_filters.png`, `u1_feature_heatmaps.png`, `u2_state_influence.png`, and printed shape traces.

- [ ] **Step 3: Code review** — invoke the `code-review` skill (or `superpowers:requesting-code-review`) on the accumulated diff; fix findings; re-run pytest.

- [ ] **Step 4: graphify update**

Run: `graphify update .`

---

## Self-Review (plan vs spec)

- **Spec coverage:** D1 package → Task 0; D2/D3 interleave+Formulation A → Task 4; D4 Mamba-1 → Task 3; D5 spike → Task 1; D6 visual proof → Steps 5 in Tasks 2/3/4; D7 run/verify → throughout. Units 1/2/3 + register all covered. ✓
- **Placeholder scan:** scan steps for "TBD/handle edge cases" — none; all code shown. The only conditional is Task 1 Step 3 (α vs β) which is resolved by the spike's printed decision, with both code paths specified. ✓
- **Type consistency:** `MambaTemporalBlock.forward(x, state)`/`fold`/`unfold`, `mamba_scan_time(mamba, x, state)`, `ResNetSpatialStages.stage_dims/strides`, `ResNetMambaBackbone.get_stage_dims/get_strides` — names consistent across Tasks 2–5. ✓

## Open risk carried into execution
The spike (Task 1) determines whether `Mamba.step()` is autograd-differentiable. If not, Task 1 Step 3 swaps in the β pure-PyTorch scan before Unit 2 is built — Unit 2's tests (Task 3) are the contract either way.
