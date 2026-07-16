# Stage 4 — Drop-in Integration of ResNetMambaBackbone into RVT — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Environment note:** all `pytest`/proof runs require the `events_signals` conda env on the RTX 5070 Ti (mamba
> kernels are CUDA-only — `conftest.py::device` asserts CUDA). These run-steps are executed by the **user**, who
> pastes back output. Commit messages must be plain (no AI co-author trailer); never stage `CLAUDE.md` /
> `PLAN_FIXES_FOR_CLAUDE.md`.

**Goal:** Make `ResNetMambaBackbone` a true drop-in for RVT's recurrent backbone so the unmodified `YoloXDetector`
(PAFPN + YOLOX head + losses) assembles and runs one train + one eval step, selected by Hydra config.

**Architecture:** Approach A — the backbone natively satisfies RVT's contract: features `(L,B,c,h,w)` (RVT indexes
`v[tidx]`) and states that are `None`-free with batch as **dim 0** (RVT's `RNNStates.recursive_detach/reset`). β
(cross-clip training state) is deferred to Stage 6 via a zero placeholder. No baseline file is edited.

**Tech Stack:** PyTorch, `mamba-ssm` (CUDA), torchvision ResNet-18, OmegaConf/Hydra, pytest, the RVT codebase under
`external/ssms_event_cameras/RVT`.

**Spec:** `docs/superpowers/specs/2026-06-12-stage4-integration-design.md`.

---

## File Structure

- **Modify** `code/event_ssm/backbone/resnet_mamba.py` — feature return shape `(L,B,c,h,w)`; backbone-owned state
  conversion `(N,…)↔dim0=B`; train zero placeholder. (Add two module-level helpers `_state_to_bmajor`/
  `_state_from_bmajor`.)
- **Modify** `code/event_ssm/integration/register.py` — also patch the detector module's already-bound
  `build_recurrent_backbone`.
- **Create** `external/ssms_event_cameras/RVT/config/model/resnet_mamba_yolox/default.yaml` — Hydra model config
  selecting our backbone (fpn/head/postprocess copied from `maxvit_yolox/default.yaml`).
- **Create** `code/event_ssm/proofs/proof_integration.py` — synthetic full-model smoke + visual proof table.
- **Modify** `code/event_ssm/tests/test_resnet_mamba.py` — feature-shape asserts + integration test indexing.
- **Modify** `code/event_ssm/tests/test_register.py` — detector-binding dispatch test.
- `MambaTemporalBlock`/`_scan.py` and `test_mamba_temporal.py` are **unchanged** (they live in the `(N,…)` world).

---

## Task 1: Backbone returns `(L, B, c, h, w)` features

**Files:**
- Modify: `code/event_ssm/backbone/resnet_mamba.py:45` (the `out = … .reshape(L*B, …)` line)
- Test: `code/event_ssm/tests/test_resnet_mamba.py`

- [ ] **Step 1: Update the failing tests to the new contract**

In `code/event_ssm/tests/test_resnet_mamba.py`, change the shape asserts in `test_forward_shapes_and_state`:

```python
def test_forward_shapes_and_state(device):
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).eval()
    x = torch.randn(5, 2, 20, 256, 320, device=device)   # (L=time, B, C=20, H, W)
    feats, states = m(x, prev_states=None)
    assert feats[2].shape == (5, 2, 128, 32, 40)
    assert feats[3].shape == (5, 2, 256, 16, 20)
    assert feats[4].shape == (5, 2, 512, 8, 10)
    assert len(states) == 4
    assert m.get_stage_dims((2,3,4)) == (128,256,512)
    assert m.get_strides((2,3,4)) == (8,16,32)
```

And in `test_integration_with_pafpn_head`, index timestep 0 before the PAFPN (features are now `(L,B,c,h,w)`):

```python
    feats, _ = m(x, prev_states=None)                       # x=(1,1,20,256,320) -> feats[k]=(1,1,c,h,w)
    pafpn = YOLOPAFPN(depth=0.33, in_stages=(2,3,4), in_channels=(128,256,512)).to(device).eval()
    head = YOLOXHead(num_classes=2, strides=(8,16,32), in_channels=(128,256,512)).to(device).eval()
    outs, _ = head(pafpn({2:feats[2][0], 3:feats[3][0], 4:feats[4][0]}))   # [0] selects L=0 -> (B,c,h,w)
    assert outs.shape[-1] == 7
    assert outs.shape[1] == 32*40 + 16*20 + 8*10            # 1680
```

- [ ] **Step 2: Run to verify it fails**

Run (user, `events_signals` env):
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python -m pytest tests/test_resnet_mamba.py -v
```
Expected: FAIL — `test_forward_shapes_and_state` asserts `(5,2,128,32,40)` but backbone still returns `(10,128,32,40)`.

- [ ] **Step 3: Fix the return shape in the backbone**

In `code/event_ssm/backbone/resnet_mamba.py`, change the per-stage output line (currently
`out = MambaTemporalBlock.unfold(seq, dims).reshape(L * B, c, h, w)`) to keep `(L,B,c,h,w)`:

```python
            out = MambaTemporalBlock.unfold(seq, dims)      # (L, B, c, h, w) — RVT indexes v[tidx]
            feats[stage] = out
```

- [ ] **Step 4: Run to verify it passes**

Run (user):
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python -m pytest tests/test_resnet_mamba.py -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add code/event_ssm/backbone/resnet_mamba.py code/event_ssm/tests/test_resnet_mamba.py
git commit -m "feat(stage4): backbone returns (L,B,c,h,w) features for RVT v[tidx] contract"
```

---

## Task 2: State contract — `None`-free, batch as dim 0 (β placeholder in train)

**Files:**
- Modify: `code/event_ssm/backbone/resnet_mamba.py` (add helpers; branch state handling on `self.training`)
- Test: `code/event_ssm/tests/test_resnet_mamba.py`

- [ ] **Step 1: Write the failing contract tests**

Append to `code/event_ssm/tests/test_resnet_mamba.py`:

```python
def test_eval_state_rvt_compatible(device):
    """Eval states: None-free, batch is dim0, survive RVT recursive detach + per-seq reset,
    and can be carried back into the step path."""
    from modules.utils.detection import RNNStates
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).eval()
    B = 2
    x = torch.randn(3, B, 20, 256, 320, device=device)        # (L,B,C,H,W)
    feats, states = m(x, prev_states=None)
    assert feats[2].shape == (3, B, 128, 32, 40)
    conv_b, ssm_b = states[1][0]                              # stage1, layer0  (eval -> (conv,ssm))
    assert conv_b.shape[0] == B and ssm_b.shape[0] == B       # batch is dim0
    detached = RNNStates.recursive_detach(states)             # must not raise on None
    RNNStates.recursive_reset(detached, indices_or_bool_tensor=[0])   # per-seq reset zeros dim0
    feats2, _ = m(x, prev_states=detached)                   # carry back into step path
    assert feats2[2].shape == (3, B, 128, 32, 40)

def test_train_state_placeholder_rvt_compatible(device):
    """Train states: None-free placeholder with batch dim0 (beta deferred -> zero-init per clip)."""
    from modules.utils.detection import RNNStates
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).train()
    B = 2
    x = torch.randn(3, B, 20, 256, 320, device=device)
    feats, states = m(x, prev_states=None)
    assert feats[2].shape == (3, B, 128, 32, 40)
    assert states[1].shape[0] == B                           # (B,1) placeholder, dim0=B
    detached = RNNStates.recursive_detach(states)            # no None -> ok
    RNNStates.recursive_reset(detached, indices_or_bool_tensor=[0])
```

- [ ] **Step 2: Run to verify it fails**

Run (user):
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python -m pytest tests/test_resnet_mamba.py::test_eval_state_rvt_compatible tests/test_resnet_mamba.py::test_train_state_placeholder_rvt_compatible -v
```
Expected: FAIL — `recursive_detach` raises `NotImplementedError` on the current `None` train states / `(N,…)` eval
states have dim0 = B·H·W not B.

- [ ] **Step 3: Implement backbone-owned state conversion**

In `code/event_ssm/backbone/resnet_mamba.py`, add two module-level helpers (after the imports) and rewrite `forward`:

```python
def _state_to_bmajor(state, B, hw):
    """Eval per-layer state [(conv:(N,di,dc), ssm:(N,di,ds)), ...] -> dim0=B for RVT storage."""
    out = []
    for conv, ssm in state:
        out.append((conv.reshape(B, hw, *conv.shape[1:]),
                    ssm.reshape(B, hw, *ssm.shape[1:])))
    return out


def _state_from_bmajor(state_b, B, hw):
    """RVT-stored dim0=B state -> per-layer (N=B*hw, ...) for the step kernels. None passes through."""
    if state_b is None:
        return None
    out = []
    for conv_b, ssm_b in state_b:
        out.append((conv_b.reshape(B * hw, *conv_b.shape[2:]),
                    ssm_b.reshape(B * hw, *ssm_b.shape[2:])))
    return out
```

Replace the body of `forward` (keep the signature) with:

```python
    def forward(self, x, prev_states=None, token_mask=None, train_step: bool = True):
        L, B, C, H, W = x.shape
        num_stages = len(self.temporal)
        if prev_states is None:
            prev_states = [None] * num_stages
        spat = self.spatial(x.reshape(L * B, C, H, W))     # {1..N}: (L*B, c, h, w)
        feats, new_states = {}, []
        for i in range(num_stages):
            stage = i + 1
            fmap = spat[stage]
            c, h, w = fmap.shape[1], fmap.shape[2], fmap.shape[3]
            seq = fmap.reshape(L, B, c, h, w)
            seq, dims = MambaTemporalBlock.fold(seq)        # (B*h*w, L, c)
            if self.training:
                seq, _ = self.temporal[i](seq, None)        # zero-init per clip (beta deferred)
                st_rvt = seq.new_zeros(B, 1)                # placeholder: None-free, dim0=B
            else:
                prev = _state_from_bmajor(prev_states[i], B, h * w)
                seq, st = self.temporal[i](seq, prev)       # st: per-layer (conv,ssm) in (N,...)
                st_rvt = _state_to_bmajor(st, B, h * w)
            feats[stage] = MambaTemporalBlock.unfold(seq, dims)   # (L, B, c, h, w)
            new_states.append(st_rvt)
        return feats, new_states
```

- [ ] **Step 4: Run to verify it passes (including the Task-1 tests)**

Run (user):
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python -m pytest tests/test_resnet_mamba.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add code/event_ssm/backbone/resnet_mamba.py code/event_ssm/tests/test_resnet_mamba.py
git commit -m "feat(stage4): RVT-compatible backbone state (dim0=B, None-free; train zero placeholder)"
```

---

## Task 3: Register patches the detector's bound builder

**Files:**
- Modify: `code/event_ssm/integration/register.py`
- Test: `code/event_ssm/tests/test_register.py`

- [ ] **Step 1: Write the failing test**

Append to `code/event_ssm/tests/test_register.py`:

```python
def test_register_patches_detector_binding():
    """YoloXDetector did `from ...recurrent_backbone import build_recurrent_backbone` (local bind);
    after register, the detector module's own name must dispatch to ResNetMamba."""
    from omegaconf import OmegaConf
    from event_ssm.integration.register import register_resnet_mamba
    register_resnet_mamba()
    import models.detection.yolox_extension.models.detector as det
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
    cfg = OmegaConf.create({"name": "ResNetMamba", "input_channels": 20,
                            "pretrained": False, "num_layers_per_stage": 1})
    bb = det.build_recurrent_backbone(cfg)
    assert isinstance(bb, ResNetMambaBackbone)
```

- [ ] **Step 2: Run to verify it fails**

Run (user):
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python -m pytest tests/test_register.py::test_register_patches_detector_binding -v
```
Expected: FAIL — `det.build_recurrent_backbone` is still the original (raises `NotImplementedError` for ResNetMamba).

- [ ] **Step 3: Patch both modules in register**

Append to `register_resnet_mamba()` in `code/event_ssm/integration/register.py`, immediately after
`rb.build_recurrent_backbone = patched`:

```python
    # YoloXDetector did `from ...recurrent_backbone import build_recurrent_backbone` (a local
    # name bind), so patch that module's name too — robust to import order.
    import models.detection.yolox_extension.models.detector as det
    det.build_recurrent_backbone = patched
```

- [ ] **Step 4: Run to verify it passes (and the original routing test still passes)**

Run (user):
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python -m pytest tests/test_register.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add code/event_ssm/integration/register.py code/event_ssm/tests/test_register.py
git commit -m "feat(stage4): register also patches the detector module's bound builder"
```

---

## Task 4: Hydra model config for the drop-in backbone

**Files:**
- Create: `external/ssms_event_cameras/RVT/config/model/resnet_mamba_yolox/default.yaml`
- Test: `code/event_ssm/tests/test_register.py` (append a config-load check)

- [ ] **Step 1: Write the failing config-load test**

Append to `code/event_ssm/tests/test_register.py`:

```python
def test_resnet_mamba_config_present_and_valid():
    import pathlib
    from omegaconf import OmegaConf
    rvt = pathlib.Path(__file__).resolve().parents[3] / "external/ssms_event_cameras/RVT"
    cfg = OmegaConf.load(rvt / "config/model/resnet_mamba_yolox/default.yaml")
    assert cfg.model.backbone.name == "ResNetMamba"
    assert cfg.model.backbone.input_channels == 20
    assert list(cfg.model.fpn.in_stages) == [2, 3, 4]
    assert cfg.model.head.name == "YoloX"
```

- [ ] **Step 2: Run to verify it fails**

Run (user):
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python -m pytest tests/test_register.py::test_resnet_mamba_config_present_and_valid -v
```
Expected: FAIL — file does not exist.

- [ ] **Step 3: Create the config (fpn/head/postprocess copied verbatim from `maxvit_yolox/default.yaml`)**

Create `external/ssms_event_cameras/RVT/config/model/resnet_mamba_yolox/default.yaml`:

```yaml
# @package _global_
defaults:
  - override /model: rnndet

model:
  backbone:
    name: ResNetMamba
    compile:
      enable: False
      args:
        mode: reduce-overhead
    input_channels: 20         # stacked_histogram: 2 pol x 10 bins (matches baseline gen1.yaml)
    pretrained: True
    d_state: 16
    num_layers_per_stage: 1
  fpn:
    name: PAFPN
    compile:
      enable: False
      args:
        mode: reduce-overhead
    depth: 0.67                # round(depth * 3) == num bottleneck blocks
    in_stages: [2, 3, 4]
    depthwise: False
    act: "silu"
  head:
    name: YoloX
    compile:
      enable: False
      args:
        mode: reduce-overhead
    depthwise: False
    act: "silu"
  postprocess:
    confidence_threshold: 0.1
    nms_threshold: 0.45
```

(`num_classes` is injected at runtime by `config/modifier.py` for Gen1 = 2 — do not hardcode it, matching the
baseline `maxvit_yolox` config.)

- [ ] **Step 4: Run to verify it passes**

Run (user):
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python -m pytest tests/test_register.py::test_resnet_mamba_config_present_and_valid -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add external/ssms_event_cameras/RVT/config/model/resnet_mamba_yolox/default.yaml code/event_ssm/tests/test_register.py
git commit -m "feat(stage4): add resnet_mamba_yolox Hydra model config (mirrors maxvit_yolox)"
```

---

## Task 5: Synthetic full-model smoke + visual proof

**Files:**
- Create: `code/event_ssm/proofs/proof_integration.py`
- Output: `code/event_ssm/proofs/out/u4_integration.md`

- [ ] **Step 1: Write the proof/smoke script**

Create `code/event_ssm/proofs/proof_integration.py`:

```python
"""Stage 4 proof: the unmodified RVT YoloXDetector, built with our drop-in ResNetMamba backbone,
assembles and runs one TRAIN step + one EVAL step (with RVT-style state detach/reset) on a
synthetic (L,B,20,256,320) clip. Writes a shape/loss/param table."""
import sys, pathlib, torch
REPO = pathlib.Path(__file__).resolve().parents[2]
for p in (REPO / "code", REPO / "external/ssms_event_cameras/RVT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from omegaconf import OmegaConf
from event_ssm.integration.register import register_resnet_mamba
register_resnet_mamba()
from models.detection.yolox_extension.models.detector import YoloXDetector
from modules.utils.detection import RNNStates

OUT = pathlib.Path(__file__).parent / "out"; OUT.mkdir(parents=True, exist_ok=True)

model_cfg = OmegaConf.create({
    "backbone": {"name": "ResNetMamba", "input_channels": 20, "pretrained": False,
                 "d_state": 16, "num_layers_per_stage": 1},
    "fpn": {"name": "PAFPN", "depth": 0.67, "in_stages": [2, 3, 4],
            "depthwise": False, "act": "silu", "compile": {"enable": False}},
    "head": {"name": "YoloX", "num_classes": 2, "depthwise": False, "act": "silu",
             "compile": {"enable": False}},   # num_classes=2 == config/modifier.py for gen1
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
assert feats[2].shape == (L, B, 128, 32, 40)
sel = {k: v[-1] for k, v in feats.items() if k in (2, 3, 4)}      # last window: (B,c,h,w)
targets = torch.zeros(B, 3, 5, device="cuda")                     # (B, max_objs, 5)=[cls,cx,cy,w,h]
targets[:, 0] = torch.tensor([0., 160., 128., 40., 30.], device="cuda")
_, losses = model.forward_detect(backbone_features=sel, targets=targets)
if isinstance(losses, dict):
    loss = losses.get("total_loss") or sum(v for v in losses.values()
                                           if torch.is_tensor(v) and v.requires_grad)
else:
    loss = losses
loss.backward()
assert torch.isfinite(loss).all()

# ---- EVAL step (carry + reset state, RVT-style) ----
model.eval()
with torch.no_grad():
    _, st1 = model.forward_backbone(x, previous_states=None, train_step=False)
    st1 = RNNStates.recursive_detach(st1)
    feats_e, st2 = model.forward_backbone(x, previous_states=st1, train_step=False)
    RNNStates.recursive_reset(RNNStates.recursive_detach(st2), indices_or_bool_tensor=[0])
    sel_e = {k: v[-1] for k, v in feats_e.items() if k in (2, 3, 4)}
    out_e, _ = model.forward_detect(backbone_features=sel_e)
assert out_e.shape[-1] == 7 and out_e.shape[1] == 32*40 + 16*20 + 8*10   # 1680

lines = [
    "# Stage 4 - full-model integration proof", "",
    "| metric | value |", "|---|---|",
    f"| params total | {n_total:.2f} M |",
    f"| params backbone / fpn / head | {n_bb:.2f} / {n_fpn:.2f} / {n_head:.2f} M |",
    f"| train feats[2] | {tuple(feats[2].shape)} |",
    f"| train loss (finite) | {float(loss):.4f} |",
    f"| eval output | {tuple(out_e.shape)} |",
    "| train step | PASS |",
    "| eval step + state detach/reset | PASS |",
]
(OUT / "u4_integration.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print("wrote", OUT / "u4_integration.md")
```

- [ ] **Step 2: Run the smoke (the verification)**

Run (user):
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python proofs/proof_integration.py
```
Expected: prints the table with `train step | PASS` and `eval step … | PASS`, `eval output (2, 1680, 7)`, total
params ~20–25M; writes `proofs/out/u4_integration.md`. If `losses` has no `total_loss` key, the script falls back to
summing grad-requiring tensors (works regardless of the exact YOLOX key).

- [ ] **Step 3: Commit**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add code/event_ssm/proofs/proof_integration.py code/event_ssm/proofs/out/u4_integration.md
git commit -m "feat(stage4): synthetic full-model integration smoke + proof table"
```

---

## Task 6: Full regression + Stage-4 doc status

**Files:**
- Modify: `stages/Stage_04_Integration.md` (flip status to done), `code/event_ssm/proofs/proof_backbone.py` (print note)

- [ ] **Step 1: Run the full suite to confirm no regressions**

Run (user):
```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python -m pytest tests/ -v
```
Expected: all pass (Stage-3 tests + the 3 new Stage-4 tests). `test_mamba_temporal.py` and `test_resnet_spatial.py`
unaffected.

- [ ] **Step 2: Update `proof_backbone.py` shape comment to the new `(L,B,…)` contract**

In `code/event_ssm/proofs/proof_backbone.py`, the printed stage shapes are now `(L,B,c,h,w)`; update the header
docstring/comment to say "per-stage feature `(L,B,c,h,w)`" (no behavioural change).

- [ ] **Step 3: Mark Stage 4 done in the plan doc**

In `stages/Stage_04_Integration.md`, change the STATUS banner to note the full-model smoke passed and the backbone
is RVT-contract-compatible (features `(L,B,c,h,w)`, dim0=B None-free states); β still deferred to Stage 6.

- [ ] **Step 4: Commit**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add stages/Stage_04_Integration.md code/event_ssm/proofs/proof_backbone.py
git commit -m "docs(stage4): mark integration complete; backbone is RVT-contract-compatible"
```

---

## Notes / Out of Scope
- **β (cross-clip training state)** is deferred to Stage 6 — training zero-inits per clip via the `(B,1)` placeholder.
  Documented in the spec §Unit 6 and `stages/Stage_04_Integration.md`.
- Real Gen1 data / overfit (Stage 5), short training (Stage 6), full Hydra training entrypoint wiring + SLURM (Stage 6/7).
- After code changes land, optionally run `graphify update .` to refresh the knowledge graph (AST-only).
