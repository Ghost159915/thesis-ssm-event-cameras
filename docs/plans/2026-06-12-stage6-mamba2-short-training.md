# Stage 6 — Mamba-2 Unified TBPTT + Short Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Stage-3 dual-path temporal scan with a single Mamba-2 stateful chunk-scan that carries (detached) state across sub-sequences in **both** training (TBPTT, matching S5-RVT) and eval; drop the dead stage-1 temporal block (Finding §8); then produce the artifacts for a short Gen1 training run (handed to the user).

**Architecture:** The temporal block becomes `Mamba2`. A new `mamba2_scan_time(layer, x, state)` replicates Mamba-2's forward internals but calls `mamba_chunk_scan_combined(initial_states=…, return_final_states=True)` so one path serves train + eval. RVT's `RNNStates` already detaches/resets state across sub-sequences, so the backbone just threads `prev_states` in and returns `new_states` in both modes. Temporal blocks are built only for FPN-consumed stages (2/3/4).

**Tech Stack:** PyTorch, `mamba-ssm==2.3.2.post1` (`Mamba2`, `mamba_chunk_scan_combined`), `causal_conv1d`, einops, the reused RVT (`external/ssms_event_cameras/RVT`) PAFPN/YOLOX/Lightning stack, Hydra, Prophesee Gen1, RTX 5070 Ti (`events_signals` env), Katana SLURM.

**Spec:** `docs/specs/2026-06-12-stage6-mamba2-short-training-design.md`.

**Env / test invocation (all tasks):**
```bash
PY=/home/ghost/miniforge3/envs/events_signals/bin/python   # CUDA + mamba_ssm; pytest.ini disables ROS plugins
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
$PY -m pytest <args>
```
All Mamba-2 kernels are **CUDA-only** — every test/proof runs on the GPU (`.cuda()`).

**Key shapes (reference):**
- Backbone temporal stages = 2/3/4 → `d_model` = 128/256/512; with `headdim=64`, `expand=2` → `d_inner` 256/512/1024 → `nheads` 4/8/16. `ngroups=1`, `d_state=64`, `d_conv=4`.
- Per-layer state = `(conv_state, ssm_state)`:
  - `conv_state`: `(N, d_conv-1, conv_dim)` — last `d_conv-1` pre-conv `xBC` frames (left context); `conv_dim = d_ssm + 2*ngroups*d_state`.
  - `ssm_state`: `(N, nheads, headdim, d_state)`.
  - `N = B*H*W` (one independent time-sequence per spatial location).

---

## Phase A — code (Claude executes + smokes)

### Task 1: Unified Mamba-2 stateful scan + equivalence gate

**Files:**
- Rewrite: `code/event_ssm/temporal/_scan.py`
- Test: `code/event_ssm/tests/test_scan_equivalence.py` (create)

- [ ] **Step 1: Write the failing equivalence test**

`code/event_ssm/tests/test_scan_equivalence.py`:
```python
import torch
import pytest
from mamba_ssm import Mamba2
from event_ssm.temporal._scan import mamba2_scan_time

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")


@pytest.mark.parametrize("d_model", [128, 256])
def test_split_equals_full_scan(d_model):
    """Carried-state scan over two sub-sequences == one full scan (TBPTT correctness)."""
    torch.manual_seed(0)
    layer = Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=64).cuda().float().eval()
    N, L = 8, 10
    x = torch.randn(N, L, d_model, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        y_full, st_full = mamba2_scan_time(layer, x, None)
        y1, st1 = mamba2_scan_time(layer, x[:, :5], None)
        y2, st2 = mamba2_scan_time(layer, x[:, 5:], st1)
    y_split = torch.cat([y1, y2], dim=1)

    assert y_full.shape == (N, L, d_model)
    max_diff = (y_full - y_split).abs().max().item()
    assert max_diff < 2e-3, f"carried-split vs full max|diff|={max_diff:.2e}"


def test_state_shapes():
    layer = Mamba2(d_model=128, d_state=64, d_conv=4, expand=2, headdim=64).cuda().float().eval()
    x = torch.randn(8, 6, 128, device="cuda")
    with torch.no_grad():
        _, (conv_state, ssm_state) = mamba2_scan_time(layer, x, None)
    assert conv_state.shape == (8, 4 - 1, layer.d_ssm + 2 * 64)   # (N, d_conv-1, conv_dim)
    assert ssm_state.shape == (8, layer.nheads, 64, 64)           # (N, nheads, headdim, d_state)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_scan_equivalence.py -v`
Expected: FAIL with `ImportError`/`AttributeError` — `mamba2_scan_time` not defined.

- [ ] **Step 3: Write the unified scan**

Replace the entire contents of `code/event_ssm/temporal/_scan.py`:
```python
"""Unified stateful temporal scan for Mamba-2 (Stage 6) — replaces the Stage-3 dual-path.

Mamba-2's official chunk-scan kernel `mamba_chunk_scan_combined` accepts an initial SSM
state and returns the final state on the *trainable* path, so a SINGLE path serves both
TRAINING (carry detached state across sub-sequences -> TBPTT, matching the S5-RVT baseline)
and EVAL/INFERENCE (carry full state -> streaming memory). RVT's RNNStates performs the
detach (between TBPTT windows) and reset (at sequence starts).

State per layer = (conv_state, ssm_state):
  conv_state: (N, d_conv-1, conv_dim)        last (d_conv-1) pre-conv xBC frames (left context)
  ssm_state : (N, nheads, headdim, d_state)  final SSM state from the chunk scan
N = B*H*W rows. Mamba-2's public forward does NOT expose initial_states, so we replicate the
relevant forward internals (in_proj -> causal conv -> chunk scan -> RMSNormGated -> out_proj).
"""
import torch
from einops import rearrange
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pure-PyTorch fallback (slower, used only if the kernel is unavailable)
    causal_conv1d_fn = None


def mamba2_scan_time(layer: Mamba2, x: torch.Tensor, state=None):
    """Scan one Mamba-2 `layer` over L=TIME for N rows. x:(N, L, C).
    Carries (conv_state, ssm_state) across calls in BOTH train and eval.
    Returns (y:(N,L,C), (conv_state, ssm_state)); the returned state is detached (TBPTT boundary)."""
    assert layer.ngroups == 1, "mamba2_scan_time assumes ngroups=1"
    N, L, _ = x.shape
    d_ssm, d_state, d_conv = layer.d_ssm, layer.d_state, layer.d_conv
    conv_dim = d_ssm + 2 * layer.ngroups * d_state
    prev_conv, prev_ssm = (None, None) if state is None else state

    zxbcdt = layer.in_proj(x)                                        # (N, L, d_in_proj)
    z, xBC, dt = torch.split(zxbcdt, [d_ssm, conv_dim, layer.nheads], dim=-1)

    # --- causal depthwise conv over time, seeded with carried left context ---
    if prev_conv is None:
        prev_conv = xBC.new_zeros(N, d_conv - 1, conv_dim)           # zero left-pad == full-scan t=0 behaviour
    xBC_ext = torch.cat([prev_conv, xBC], dim=1)                     # (N, d_conv-1+L, conv_dim)
    new_conv = xBC[:, -(d_conv - 1):].detach()                      # carry last d_conv-1 input frames
    xBC_t = rearrange(xBC_ext, "n l d -> n d l")
    if causal_conv1d_fn is not None:
        conv_out = causal_conv1d_fn(
            xBC_t, rearrange(layer.conv1d.weight, "d 1 w -> d w"),
            bias=layer.conv1d.bias, activation=layer.activation,
        )                                                            # (N, conv_dim, d_conv-1+L)
    else:
        conv_out = layer.act(layer.conv1d(xBC_t)[..., : xBC_t.shape[-1]])
    xBC = rearrange(conv_out[..., (d_conv - 1):], "n d l -> n l d")  # drop warm-up -> (N, L, conv_dim)

    x_ssm, B, C = torch.split(xBC, [d_ssm, d_state, d_state], dim=-1)
    A = -torch.exp(layer.A_log.float())                              # (nheads,)
    y, last_state = mamba_chunk_scan_combined(
        rearrange(x_ssm, "n l (h p) -> n l h p", p=layer.headdim),
        dt, A,
        rearrange(B, "n l (g s) -> n l g s", g=layer.ngroups),
        rearrange(C, "n l (g s) -> n l g s", g=layer.ngroups),
        chunk_size=layer.chunk_size, D=layer.D, z=None,
        dt_bias=layer.dt_bias, dt_softplus=True,
        initial_states=prev_ssm,                                     # (N, nheads, headdim, d_state) or None
        return_final_states=True,
    )
    y = rearrange(y, "n l h p -> n l (h p)")
    y = layer.norm(y, z)                                             # RMSNormGated(y, gate=z)
    out = layer.out_proj(y)
    return out, (new_conv, last_state.detach())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_scan_equivalence.py -v`
Expected: PASS (both params; `max|diff| < 2e-3`).

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/temporal/_scan.py code/event_ssm/tests/test_scan_equivalence.py
git commit -m "feat(stage6): unified Mamba-2 stateful scan + TBPTT equivalence test"
```

---

### Task 2: Mamba-2 temporal block

**Files:**
- Modify: `code/event_ssm/temporal/mamba_temporal.py`
- Modify: `code/event_ssm/tests/test_mamba_temporal.py`

- [ ] **Step 1: Update the block test (failing)**

In `code/event_ssm/tests/test_mamba_temporal.py`, replace the Mamba-1 shape/state assertions with Mamba-2 ones and add a block-level equivalence check. Add/replace these tests (keep the existing `fold`/`unfold` round-trip test unchanged):
```python
import torch, pytest
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")


def test_block_forward_shape_and_state():
    blk = MambaTemporalBlock(d_model=128, num_layers=1).cuda().float().eval()
    x = torch.randn(8, 6, 128, device="cuda")
    with torch.no_grad():
        y, state = blk(x, None)
    assert y.shape == (8, 6, 128)
    assert len(state) == 1                                   # one layer -> one (conv,ssm) tuple
    conv_state, ssm_state = state[0]
    assert ssm_state.shape == (8, blk.layers[0].nheads, 64, 64)


def test_block_split_equals_full():
    blk = MambaTemporalBlock(d_model=128, num_layers=2).cuda().float().eval()
    x = torch.randn(8, 10, 128, device="cuda")
    with torch.no_grad():
        y_full, _ = blk(x, None)
        y1, s1 = blk(x[:, :5], None)
        y2, _ = blk(x[:, 5:], s1)
    assert (y_full - torch.cat([y1, y2], 1)).abs().max().item() < 3e-3
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_mamba_temporal.py -v`
Expected: FAIL (block still builds `Mamba`/uses `mamba_scan_time`; state shape mismatch).

- [ ] **Step 3: Update the block to Mamba-2**

Replace the imports + class body in `code/event_ssm/temporal/mamba_temporal.py` (keep `fold`/`unfold` exactly as-is):
```python
import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from einops import rearrange
from event_ssm.temporal._scan import mamba2_scan_time


class MambaTemporalBlock(nn.Module):
    """num_layers causal Mamba-2 block(s) over the TIME axis, per spatial location.
    forward expects x:(N, L, C) with N=B*H*W and L=time; carries (conv,ssm) state across
    clips in BOTH train and eval (unified Mamba-2 chunk scan; see temporal/_scan.py)."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
            for _ in range(num_layers)
        )

    def forward(self, x, state=None):
        if state is None:
            state = [None] * len(self.layers)
        new_state = []
        for layer, st in zip(self.layers, state):
            x, st2 = mamba2_scan_time(layer, x, st)
            new_state.append(st2)
        return x, new_state

    @staticmethod
    def fold(x):  # (L,B,C,H,W) -> ((B*H*W, L, C), dims)
        L, B, C, H, W = x.shape
        return rearrange(x, "L B C H W -> (B H W) L C"), (L, B, C, H, W)

    @staticmethod
    def unfold(x, dims):  # (B*H*W,L,C) -> (L,B,C,H,W)
        L, B, C, H, W = dims
        return rearrange(x, "(B H W) L C -> L B C H W", B=B, H=H, W=W)
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_mamba_temporal.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/temporal/mamba_temporal.py code/event_ssm/tests/test_mamba_temporal.py
git commit -m "feat(stage6): Mamba-2 temporal block (unified stateful path)"
```

---

### Task 3: Backbone — Finding §8 (temporal on FPN stages only) + unified state threading

**Files:**
- Modify: `code/event_ssm/backbone/resnet_mamba.py`
- Modify: `code/event_ssm/tests/test_resnet_mamba.py`

- [ ] **Step 1: Write/adjust failing tests**

Add to `code/event_ssm/tests/test_resnet_mamba.py`:
```python
import torch, pytest
from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")


def _bb():
    return ResNetMambaBackbone(in_channels=20, pretrained=False, d_state=64,
                               temporal_stages=(2, 3, 4)).cuda().float()


def test_temporal_only_on_fpn_stages():
    bb = _bb()
    assert set(bb.temporal.keys()) == {"2", "3", "4"}        # no stage-1 temporal (Finding §8)


def test_forward_shapes_and_states_train_and_eval():
    bb = _bb()
    x = torch.randn(3, 2, 20, 256, 320, device="cuda")        # (L,B,C,H,W)
    for mode in ("train", "eval"):
        getattr(bb, mode)()
        feats, states = bb(x, None)
        assert set(feats.keys()) == {1, 2, 3, 4}
        assert feats[2].shape[0] == 3 and feats[2].shape[1] == 2   # (L,B,c,h,w)
        assert len(states) == 4
        # temporal stages carry a real (conv,ssm) state in BOTH modes; stage-1 is a placeholder
        for s in (2, 3, 4):
            assert isinstance(states[s - 1], list) and len(states[s - 1]) == 1


def test_state_carry_changes_output():
    bb = _bb().eval()
    x = torch.randn(3, 2, 20, 256, 320, device="cuda")
    with torch.no_grad():
        f0, st = bb(x, None)
        f1, _ = bb(x, st)                                     # carried memory must change stage-4 output
    assert (f0[4] - f1[4]).abs().max().item() > 1e-5
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_resnet_mamba.py -v`
Expected: FAIL (`temporal` is a `ModuleList`, no `temporal_stages` arg, training zero-inits state).

- [ ] **Step 3: Rewrite the backbone**

Replace `code/event_ssm/backbone/resnet_mamba.py` (the `_state_to_bmajor`/`_state_from_bmajor` helpers stay — they are shape-agnostic and already generalise to Mamba-2 state):
```python
import torch
import torch.nn as nn
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock


def _state_to_bmajor(state, B, hw):
    """Per-layer [(conv:(N,..), ssm:(N,..)), ...] -> dim0=B for RVT RNNStates storage."""
    out = []
    for conv, ssm in state:
        out.append((conv.reshape(B, hw, *conv.shape[1:]),
                    ssm.reshape(B, hw, *ssm.shape[1:])))
    return out


def _state_from_bmajor(state_b, B, hw):
    """RVT dim0=B state -> per-layer (N=B*hw, ...) for the scan. None passes through."""
    if state_b is None:
        return None
    out = []
    for conv_b, ssm_b in state_b:
        out.append((conv_b.reshape(B * hw, *conv_b.shape[2:]),
                    ssm_b.reshape(B * hw, *ssm_b.shape[2:])))
    return out


class ResNetMambaBackbone(nn.Module):
    """Interleaved ResNet-18 conv (spatial) + Mamba-2 (temporal) on FPN-consumed stages only.
    forward(x:(L,B,20,H,W), prev_states) -> (features dict{1..N}, states list[N]).
    Mirrors RVT's recurrent-backbone contract. State is carried (detached) across sub-sequences
    in BOTH train and eval (TBPTT); RVT's RNNStates does the detach/reset.

    `token_mask`/`train_step` are accepted only for RVT signature-compatibility (unused)."""

    def __init__(self, in_channels: int = 20, pretrained: bool = True, d_state: int = 64,
                 num_layers_per_stage: int = 1, temporal_stages=(2, 3, 4)):
        super().__init__()
        self.spatial = ResNetSpatialStages(in_channels, pretrained)
        self.temporal_stages = tuple(temporal_stages)
        self.temporal = nn.ModuleDict({
            str(s): MambaTemporalBlock(d_model=self.spatial.stage_dims[s - 1],
                                       d_state=d_state, num_layers=num_layers_per_stage)
            for s in self.temporal_stages
        })

    def get_stage_dims(self, stages):           # stages 1-indexed, e.g. (2,3,4)
        return tuple(self.spatial.stage_dims[s - 1] for s in stages)

    def get_strides(self, stages):
        return tuple(self.spatial.strides[s - 1] for s in stages)

    def forward(self, x, prev_states=None, token_mask=None, train_step: bool = True):
        L, B, C, H, W = x.shape
        num_stages = len(self.spatial.stage_dims)
        if prev_states is None:
            prev_states = [None] * num_stages
        spat = self.spatial(x.reshape(L * B, C, H, W))          # {1..N}: (L*B, c, h, w)
        feats, new_states = {}, []
        for i in range(num_stages):
            stage = i + 1
            fmap = spat[stage]
            c, h, w = fmap.shape[1], fmap.shape[2], fmap.shape[3]
            seq = fmap.reshape(L, B, c, h, w)
            if str(stage) in self.temporal:
                folded, dims = MambaTemporalBlock.fold(seq)     # (B*h*w, L, c)
                prev = _state_from_bmajor(prev_states[i], B, h * w)
                folded, st = self.temporal[str(stage)](folded, prev)
                feats[stage] = MambaTemporalBlock.unfold(folded, dims)
                new_states.append(_state_to_bmajor(st, B, h * w))
            else:
                feats[stage] = seq                              # no temporal (not FPN-fed)
                new_states.append(seq.new_zeros(B, 1))          # None-free placeholder, dim0=B
        return feats, new_states
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_resnet_mamba.py -v`
Expected: PASS (all three new tests).

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/backbone/resnet_mamba.py code/event_ssm/tests/test_resnet_mamba.py
git commit -m "feat(stage6): backbone temporal on FPN stages only + unified state threading"
```

---

### Task 4: Builder/register — thread temporal_stages + d_state default

**Files:**
- Modify: `code/event_ssm/integration/register.py:22-29` (the `patched` builder)
- Modify: `code/event_ssm/tests/test_register.py`

- [ ] **Step 1: Write failing test**

Add to `code/event_ssm/tests/test_register.py`:
```python
def test_builder_temporal_stages_from_fpn(monkeypatch):
    """The builder builds temporal blocks only for the FPN's in_stages."""
    from omegaconf import OmegaConf
    from event_ssm.integration.register import register_backbone_builder
    import models.detection.recurrent_backbone as rb
    register_backbone_builder()
    cfg = OmegaConf.create({"name": "ResNetMamba", "input_channels": 20, "pretrained": False,
                            "d_state": 64, "num_layers_per_stage": 1, "in_stages": [2, 3, 4]})
    bb = rb.build_recurrent_backbone(cfg)
    assert set(bb.temporal.keys()) == {"2", "3", "4"}
    assert bb.spatial.stage_dims[1] == bb.get_stage_dims((2,))[0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_register.py::test_builder_temporal_stages_from_fpn -v`
Expected: FAIL (builder ignores `in_stages`; default `temporal_stages=(2,3,4)` may still pass — if so, change the test cfg `in_stages` to `[3,4]` and assert keys `{"3","4"}` to prove threading).

- [ ] **Step 3: Update the builder**

In `code/event_ssm/integration/register.py`, replace the `patched` function body inside `register_backbone_builder`:
```python
    def patched(backbone_cfg):
        if backbone_cfg.name == "ResNetMamba":
            in_stages = backbone_cfg.get("in_stages", None)
            temporal_stages = tuple(in_stages) if in_stages is not None else (2, 3, 4)
            return ResNetMambaBackbone(
                in_channels=backbone_cfg.input_channels,
                pretrained=backbone_cfg.get("pretrained", True),
                d_state=backbone_cfg.get("d_state", 64),
                num_layers_per_stage=backbone_cfg.get("num_layers_per_stage", 1),
                temporal_stages=temporal_stages,
            )
        return orig(backbone_cfg)
```
Note: the YOLOX detector passes the FPN's `in_stages` onto the backbone config, or the backbone cfg carries it directly (`configs/resnet_mamba.yaml` sets `fpn.in_stages`). If the builder only receives `backbone_cfg`, mirror `fpn.in_stages` into the backbone block in the config (Task 5) so `backbone_cfg.in_stages` exists; the `.get(..., None)` fallback keeps it safe.

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_register.py -v`
Expected: PASS (all register tests, including the existing config-modifier test).

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/integration/register.py code/event_ssm/tests/test_register.py
git commit -m "feat(stage6): builder threads temporal_stages (Finding §8) + d_state=64 default"
```

---

### Task 5: Config — d_state=64 + temporal_stages wiring

**Files:**
- Modify: `code/event_ssm/configs/resnet_mamba.yaml`
- Modify: `code/event_ssm/configs/experiment/gen1/resnet_mamba.yaml` (if it duplicates backbone keys)

- [ ] **Step 1: Update the reference config**

Edit `code/event_ssm/configs/resnet_mamba.yaml` so the backbone block carries the Mamba-2 defaults and the FPN stages it should mirror onto the backbone:
```yaml
backbone:
  name: ResNetMamba
  input_channels: 20   # stacked_histogram: 2 polarities x 10 bins
  pretrained: true
  d_state: 64          # Mamba-2 (was 16 for Mamba-1); VRAM lever — lower to 32/16 if bs/seq grows
  num_layers_per_stage: 1
  in_stages: [2, 3, 4] # mirror of fpn.in_stages -> temporal built only here (Finding §8)
fpn:
  name: PAFPN
  in_stages: [2, 3, 4]
  depth: 0.33
head:
  name: YoloX
  num_classes: 2
```

- [ ] **Step 2: Verify Hydra still composes**

Run: `$PY -m pytest tests/test_register.py::test_config_modifier_injects_hw_and_num_classes -v`
Expected: PASS (the real `train` config still composes with the modifier patch; `in_res_hw=(256,320)`, `num_classes=2`).

- [ ] **Step 3: Commit**

```bash
git add code/event_ssm/configs/resnet_mamba.yaml code/event_ssm/configs/experiment/gen1/resnet_mamba.yaml
git commit -m "feat(stage6): config d_state=64 + temporal_stages mirror of fpn.in_stages"
```

---

### Task 6: Equivalence visual proof (per-stage proof artifact)

**Files:**
- Create: `code/event_ssm/proofs/proof_equivalence.py`
- Output: `code/event_ssm/proofs/out/scan_equivalence.png` + `results/stage6/equivalence.md`

- [ ] **Step 1: Write the proof script**

`code/event_ssm/proofs/proof_equivalence.py` — for each temporal stage dim (128/256/512), run the carried-split vs full scan and plot per-timestep `max|diff|`:
```python
"""Visual proof: Mamba-2 carried-state scan == full scan (TBPTT correctness gate, Stage-6 A4.1)."""
import torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mamba_ssm import Mamba2
from event_ssm.temporal._scan import mamba2_scan_time

OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
rows = []
for d_model in (128, 256, 512):
    torch.manual_seed(0)
    layer = Mamba2(d_model=d_model, d_state=64, expand=2, headdim=64).cuda().float().eval()
    N, L, split = 16, 12, 6
    x = torch.randn(N, L, d_model, device="cuda")
    with torch.no_grad():
        y_full, _ = mamba2_scan_time(layer, x, None)
        y1, s1 = mamba2_scan_time(layer, x[:, :split], None)
        y2, _ = mamba2_scan_time(layer, x[:, split:], s1)
    diff = (y_full - torch.cat([y1, y2], 1)).abs().amax(dim=(0, 2)).cpu()   # per-timestep
    ax.plot(range(L), diff, marker="o", label=f"d_model={d_model}")
    rows.append((d_model, diff.max().item()))
ax.axvline(split - 0.5, ls="--", c="grey"); ax.set_yscale("log")
ax.set_xlabel("timestep"); ax.set_ylabel("max|full - carried-split|"); ax.legend()
ax.set_title("Mamba-2 TBPTT carried-state equivalence")
fig.tight_layout(); fig.savefig(OUT / "scan_equivalence.png", dpi=120)
res = Path(__file__).resolve().parents[3] / "results/stage6"; res.mkdir(parents=True, exist_ok=True)
(res / "equivalence.md").write_text(
    "# Stage 6 — scan equivalence\n\n| d_model | max|diff| |\n|---|---|\n" +
    "\n".join(f"| {d} | {v:.2e} |" for d, v in rows) + "\n")
print("equivalence:", rows)
```

- [ ] **Step 2: Run the proof**

Run: `$PY proofs/proof_equivalence.py`
Expected: prints `max|diff|` per stage all `< 2e-3`; writes `out/scan_equivalence.png` + `results/stage6/equivalence.md`.

- [ ] **Step 3: Commit**

```bash
git add code/event_ssm/proofs/proof_equivalence.py code/event_ssm/proofs/out/scan_equivalence.png
git add -f results/stage6/equivalence.md
git commit -m "proof(stage6): TBPTT carried-state equivalence (visual)"
```

---

### Task 7: Re-run the overfit smoke on the Mamba-2 backbone

**Files:**
- Use (no edit expected): `code/event_ssm/proofs/smoke_overfit.py`
- Output: `results/smoke_test/overfit_loss_curve.png` (overwrite)

- [ ] **Step 1: Build the smoke dataset (if not present)**

Run: `$PY integration/make_smoke_dataset.py`
Expected: `data/gen1_smoke/{train,val,test}` symlink tree exists (idempotent).

- [ ] **Step 2: Run the overfit smoke**

Run: `$PY proofs/smoke_overfit.py`
Expected: monotonic loss decrease on a fixed real Gen1 batch, **≥3×** reduction, no NaN; loss curve written. (If the run errors on a Mamba-2 state shape, fix the offending proof helper — not the scan — and re-run.)

- [ ] **Step 3: Commit the refreshed curve**

```bash
git add -f results/smoke_test/overfit_loss_curve.png results/smoke_test/smoke_results.md
git commit -m "proof(stage6): overfit smoke re-passes on Mamba-2 backbone (>=3x)"
```

---

### Task 8: Re-run health + param-count delta

**Files:**
- Modify if needed: `code/event_ssm/proofs/smoke_health.py` (drop the documented dead-`temporal[0]` exclusion — there is no longer a dead block; add a param-count table)
- Output: `results/smoke_test/smoke_results.md`, `results/stage6/params.md`

- [ ] **Step 1: Update the grad-flow exclusion**

In `smoke_health.py`, remove the special-case that excluded `backbone.temporal.0.` from the "real-missing" grad check (Finding §8 removed that block). The deterministic-path grad check must now find **zero** missing grads in the backbone temporal blocks.

- [ ] **Step 2: Add a param-count table**

Append to `smoke_health.py` a section that prints backbone / temporal / FPN / head param counts and writes `results/stage6/params.md` (compare to the Stage-5 ~19.26M; record the Mamba-1→Mamba-2 + dead-temporal-removed delta).

- [ ] **Step 3: Run health**

Run: `$PY proofs/smoke_health.py`
Expected: grad-flow PASS with **no dead temporal**; VRAM sweep {1,2,4} (note: carried state at `d_state=64` raises VRAM vs Stage 5 — record it); eval step latency; param table written.

- [ ] **Step 4: Commit**

```bash
git add code/event_ssm/proofs/smoke_health.py
git add -f results/smoke_test/smoke_results.md results/stage6/params.md
git commit -m "proof(stage6): health re-run (no dead temporal) + param-count delta"
```

---

### Task 9: Full test suite green

- [ ] **Step 1: Run the whole suite**

Run: `$PY -m pytest tests/ -q`
Expected: all pass (Stage-5 had 21; expect ≥21 with the new equivalence/state tests). Fix any stragglers (most likely lingering Mamba-1 shape assertions in `test_mamba_temporal.py`/`test_resnet_mamba.py`).

- [ ] **Step 2: Commit any fixes**

```bash
git add code/event_ssm/tests
git commit -m "test(stage6): suite green on Mamba-2 unified path"
```

---

## Phase B — short-training artifacts (training handed to the user)

### Task 10: Fixed-seed 10%-recording train subset builder

**Files:**
- Create: `code/event_ssm/integration/make_train_subset.py`
- Test: `code/event_ssm/tests/test_train_subset.py`

- [ ] **Step 1: Write failing test**

`code/event_ssm/tests/test_train_subset.py`:
```python
from pathlib import Path
from event_ssm.integration.make_train_subset import select_recordings


def test_deterministic_ten_percent(tmp_path):
    src = tmp_path / "train"; src.mkdir()
    for i in range(50):
        (src / f"rec_{i:03d}").mkdir()
    a = select_recordings(src, frac=0.10, seed=1234)
    b = select_recordings(src, frac=0.10, seed=1234)
    assert a == b                                   # deterministic
    assert len(a) == 5                              # 10% of 50
    assert select_recordings(src, frac=0.10, seed=1) != a   # seed matters
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_train_subset.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the builder**

`code/event_ssm/integration/make_train_subset.py`:
```python
"""Sample a fixed-seed random 10% of Gen1 train RECORDINGS (ISSUE-10) and log the list.
Windows within each recording stay contiguous. Builds a symlink subtree the RVT data module reads."""
import argparse, random
from pathlib import Path


def select_recordings(train_dir: Path, frac: float = 0.10, seed: int = 1234):
    recs = sorted(p.name for p in Path(train_dir).iterdir() if p.is_dir())
    k = max(1, round(len(recs) * frac))
    return sorted(random.Random(seed).sample(recs, k))


def build_subset(src_train: Path, dest_root: Path, frac=0.10, seed=1234):
    chosen = select_recordings(src_train, frac, seed)
    for split in ("train", "val", "test"):           # val/test point at the real full splits
        (dest_root / split).mkdir(parents=True, exist_ok=True)
    for r in chosen:
        link = dest_root / "train" / r
        if not link.exists():
            link.symlink_to((src_train / r).resolve())
    (dest_root / "train_subset_recordings.txt").write_text("\n".join(chosen) + "\n")
    print(f"selected {len(chosen)} recordings (seed={seed}); logged to train_subset_recordings.txt")
    return chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-train", type=Path, required=True, help="full Gen1 train split dir")
    ap.add_argument("--dest", type=Path, default=Path("data/gen1_subset10"))
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    build_subset(a.src_train, a.dest, a.frac, a.seed)
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_train_subset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/integration/make_train_subset.py code/event_ssm/tests/test_train_subset.py
git commit -m "feat(stage6): fixed-seed 10%-recording train-subset builder (ISSUE-10)"
```

---

### Task 11: Run artifacts — Katana SLURM + local command + monitoring checklist

**Files:**
- Create: `code/event_ssm/scripts/stage6_short_train.slurm`
- Create: `code/event_ssm/scripts/stage6_run_local.sh`
- Create: `code/event_ssm/scripts/STAGE6_RUN.md` (run + monitoring checklist)

- [ ] **Step 1: Local run command**

`code/event_ssm/scripts/stage6_run_local.sh` — activates `events_signals`, sets `PYTHONPATH` to `code/`+RVT, and launches the real RVT `train.py` with the resnet_mamba experiment + short overrides (bf16, no GradScaler — ISSUE-09):
```bash
#!/usr/bin/env bash
set -euo pipefail
source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
cd "$REPO/external/ssms_event_cameras/RVT"
python train.py model=rnndet +experiment/gen1=resnet_mamba \
  dataset.path="$REPO/data/gen1_subset10" \
  training.max_epochs=1 hardware.gpus=0 \
  batch_size.train=8 batch_size.eval=8 \
  +callbacks.checkpoint.save_top_k=1 wandb.mode=offline
```
(`+experiment/gen1=resnet_mamba` registers our backbone via the harness `register_resnet_mamba()` call that `train.py`'s startup must perform — confirm the experiment config triggers it; otherwise add `register_resnet_mamba()` import shim, mirroring the smoke harness `setup_paths()`.)

- [ ] **Step 2: Katana SLURM script**

`code/event_ssm/scripts/stage6_short_train.slurm` — single-GPU CUDA, module loads, the same `train.py` invocation; placeholders for the cluster's conda/module names with a comment to fill them:
```bash
#!/bin/bash
#SBATCH --job-name=ssm_stage6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=stage6_%j.out
set -euo pipefail
# module load cuda/12.8   # <-- set to Katana's CUDA >=12.8 (Blackwell/cu128 stack)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
REPO="$SLURM_SUBMIT_DIR"
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
cd "$REPO/external/ssms_event_cameras/RVT"
srun python train.py model=rnndet +experiment/gen1=resnet_mamba \
  dataset.path="$REPO/data/gen1_subset10" \
  training.max_epochs=1 hardware.gpus=0 batch_size.train=8 batch_size.eval=8 \
  wandb.mode=offline
```

- [ ] **Step 3: Run + monitoring checklist doc**

`code/event_ssm/scripts/STAGE6_RUN.md` — prerequisites (download full Gen1 **train** split; build the 10% subset via Task 10), the two launch options, and the **paste-back checklist**: total + cls/obj/iou sub-losses (trend down), LR schedule, grad-norm, VRAM, throughput (it/s), any NaN/divergence, and a rough val-mAP from the validation epoch.

- [ ] **Step 4: Syntax-check the scripts**

Run: `bash -n code/event_ssm/scripts/stage6_short_train.slurm && bash -n code/event_ssm/scripts/stage6_run_local.sh`
Expected: no output (valid syntax).

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/scripts
git commit -m "feat(stage6): short-training run artifacts (local + Katana SLURM + monitoring checklist)"
```

---

### Task 12: Stage report

**Files:**
- Create: `reports/Stage_06_Mamba2_Training_Report.md`

- [ ] **Step 1: Write the report**

Mirror the Stage-5 report structure: what the stage set out to do; ADDED/CHANGED/REMOVED tables; **proof of working** (equivalence plot, overfit curve, health/param table); what went well / wrong; the train/eval-parity resolution narrative (dual-path → unified, §4 of the spec); deferred/open items; **a "Phase B — handed to user" section left to be filled after the training run**; commit list.

- [ ] **Step 2: Commit**

```bash
git add reports/Stage_06_Mamba2_Training_Report.md
git commit -m "docs(stage6): stage report (Mamba-2 unified TBPTT; Phase A complete)"
```

---

## Handoff (after Phase A is green)

The user downloads the **full Gen1 train split**, runs Task 10's subset builder, then launches the short training (local or SLURM, Task 11) and pastes back the monitoring checklist. Claude then produces the loss/LR/grad curves and completes the Phase-B section of the Stage-6 report. **Per the terminal policy, Claude does not run full training or dataset downloads.**

---

## Self-review notes (author)

- **Spec coverage:** §2 decision → Tasks 1–2 (Mamba-2); §5 A1 → Task 2; A2 unified scan → Task 1; A3 Finding §8 → Tasks 3–5; A4.1 equivalence → Tasks 1 + 6; A4.2 overfit → Task 7; A4.3 health/params → Task 8; §6 Phase B → Tasks 10–11; success criteria → Tasks 6–9 (Phase A) + Task 11 checklist (Phase B); report (session grant) → Task 12.
- **Type consistency:** state tuple `(conv_state, ssm_state)` and shapes `(N, d_conv-1, conv_dim)` / `(N, nheads, headdim, d_state)` are used identically across `_scan.py`, the block tests, the backbone reshapers, and the proof. `mamba2_scan_time` / `MambaTemporalBlock(d_model, d_state=64, headdim=64, num_layers)` / `ResNetMambaBackbone(..., temporal_stages)` signatures match every call site.
- **Open risk to watch at execution:** VRAM from carried state at `d_state=64` (Task 8 measures it; `d_state` and `batch_size` are the documented levers); the `+experiment/gen1=resnet_mamba` config must actually invoke `register_resnet_mamba()` at `train.py` startup (Task 11 Step 1 note).
```
