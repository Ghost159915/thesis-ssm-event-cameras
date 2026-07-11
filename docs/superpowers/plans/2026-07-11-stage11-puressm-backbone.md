# Stage 11 — PureSSM BiMamba Spatial Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `BiMambaSpatialStages` — a fully-pure, stateless, 4-stage bidirectional-Mamba spatial backbone that is a drop-in duck-type replacement for `ResNetSpatialStages` — plus the latency/VRAM probe and untrained-ERF figure that gate Stage 12.

**Architecture:** New package `code/event_ssm/spatial/` with three files: a bidirectional Mamba-2 scan core (`_scan2d.py`, extending the trusted `temporal/_scan.py` kernel pattern — shared in/out projections, per-direction conv/A/dt/D, sum before one shared gate), a 2D block (`bimamba_block.py`: zero-init DWConv3×3 local mix + axis-alternating bidirectional scan, pre-norm residual), and the 4-stage pyramid (`bimamba_spatial.py`: conv stem stride 4, depths [2,2,8,2], dims 64/128/256/512, conv downsamplers, no positional embeddings). One minimal modification to `ResNetMambaBackbone` (optional `spatial=` injection) assembles the full recurrent backbone for the probe.

**Tech Stack:** PyTorch 2.11.0+cu128 (Blackwell sm_120), mamba-ssm 2.3.2.post1 (`mamba_chunk_scan_combined`, `RMSNormGated`), causal-conv1d 1.6.2.post1, einops 0.8.2, pytest, matplotlib.

**Authoritative spec:** `docs/superpowers/specs/2026-07-11-puressm-backbone-design.md`

## Global Constraints

- `PY=/home/ghost/miniforge3/envs/events_signals/bin/python` — all commands use this interpreter; tests are CUDA-only (mamba kernels), run on the idle RTX 5070 Ti.
- **Never `pip install` anything in this stage.** timm is absent — DropPath is vendored below. No edits under `external/`.
- Spatial SSM hyperparameters (spec §4.2): `d_state=16, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256, d_ssm=d_inner`. Temporal path keeps `d_state=64` and is NOT modified.
- Duck-type contract (spec §3): `stage_dims=(64,128,256,512)`, `strides=(4,8,16,32)`, `forward((N,20,256,320)) → {1:(N,64,64,80), 2:(N,128,32,40), 3:(N,256,16,20), 4:(N,512,8,10)}`; stateless; no BatchNorm anywhere.
- Param gate: spatial module total params in **[8e6, 16e6]**.
- Probe gates (spec U3): projected full pipeline ≥ 51 Hz (backbone streaming p50 ≤ 12.5 ms; Stage-10 neck+head = 7.2 ms) and train-shape step < 16 GB. **Gate failure = STOP and escalate to user (fallback ladder is a user decision).**
- Visual proofs to `code/event_ssm/proofs/out/`. Before writing figure code, load the `dataviz` skill; repo palette: ours `#2a78d6`, baseline `#1baf7a` (Stage-10 convention).
- Commits: conventional style (`feat(stage11): …`), no assistant names, commit after every task.
- Work on branch `stage11-puressm-backbone` (created in Task 1, merged via superpowers:finishing-a-development-branch at the end).

## File Structure

```
code/event_ssm/spatial/
├── __init__.py            # exports BiMamba1DScan, BiMamba2DBlock, BiMambaSpatialStages
├── _scan2d.py             # BiMamba1DScan: bidirectional Mamba-2 scan over a flattened axis
├── bimamba_block.py       # DropPath (vendored), LayerNorm2d, BiMamba2DBlock
└── bimamba_spatial.py     # BiMambaSpatialStages (duck-type of ResNetSpatialStages)
code/event_ssm/backbone/resnet_mamba.py   # MODIFY: optional `spatial=` injection param
code/event_ssm/tests/test_spatial_scan2d.py
code/event_ssm/tests/test_bimamba_spatial.py
code/event_ssm/scripts/stage11_probe.py   # U3 gate script
code/event_ssm/scripts/stage11_erf.py     # U5 untrained-ERF figure
```

---

### Task 1: `BiMamba1DScan` — module skeleton with shape test

**Files:**
- Create: `code/event_ssm/spatial/__init__.py`, `code/event_ssm/spatial/_scan2d.py`
- Test: `code/event_ssm/tests/test_spatial_scan2d.py`

**Interfaces:**
- Produces: `BiMamba1DScan(d_model, d_state=16, d_conv=4, expand=2, headdim=64, chunk_size=256)`; `forward(x: (N, S, d_model)) -> (N, S, d_model)`. Attributes used by later tasks/tests: `.d_inner`, `.nheads`, `.conv_dim`, and per-direction parameter pairs `conv1d_fwd/conv1d_bwd`, `A_log_fwd/A_log_bwd`, `dt_bias_fwd/dt_bias_bwd`, `D_fwd/D_bwd`.

- [ ] **Step 1: Verify the kernel imports exist in this env (no code yet)**

Run:
```bash
PY=/home/ghost/miniforge3/envs/events_signals/bin/python
$PY -c "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, ssd_chunk_scan_combined_ref; from mamba_ssm.ops.triton.layernorm_gated import RMSNormGated; from causal_conv1d import causal_conv1d_fn; print('kernel imports ok')"
```
Expected: `kernel imports ok`. If `RMSNormGated` fails to import from that path, grep for it (`grep -rn "class RMSNormGated" /home/ghost/miniforge3/envs/events_signals/lib/python3.11/site-packages/mamba_ssm/`) and use the found path in Step 4 — do not proceed on a guessed import.

- [ ] **Step 2: Create branch; write the failing shape test**

```bash
git checkout -b stage11-puressm-backbone
```

`code/event_ssm/tests/test_spatial_scan2d.py`:
```python
# Stage 11 U1: bidirectional Mamba-2 scan core (spec §4.2, §6 U1)
import pytest
import torch


def _make(d_model=64, device="cuda", dtype=torch.float32):
    from event_ssm.spatial import BiMamba1DScan
    torch.manual_seed(0)
    return BiMamba1DScan(d_model=d_model).to(device=device, dtype=dtype)


def test_scan_output_shape(device):
    m = _make(device=device)
    x = torch.randn(3, 100, 64, device=device)
    y = m(x)
    assert y.shape == (3, 100, 64)
    assert y.dtype == x.dtype


def test_scan_all_stage_lengths(device):
    # exactly the four flatten lengths the pyramid produces (spec §4.3 + risk table)
    m = _make(device=device)
    for s in (5120, 1280, 320, 80):
        y = m(torch.randn(1, s, 64, device=device))
        assert y.shape == (1, s, 64)
```

- [ ] **Step 3: Run to verify failure**

Run: `$PY -m pytest code/event_ssm/tests/test_spatial_scan2d.py -v`
Expected: FAIL — `ImportError: cannot import name 'BiMamba1DScan'`.

- [ ] **Step 4: Implement `_scan2d.py` and `__init__.py`**

`code/event_ssm/spatial/_scan2d.py`:
```python
"""Bidirectional Mamba-2 scan core for SPATIAL token sequences (Stage 11, spec U1).

Design (spec §4.2, locked 2026-07-11): one SHARED in_proj / RMSNormGated gate / out_proj,
PER-DIRECTION depthwise conv, A_log, dt_bias, D (Vim/VMamba both learn separate direction
decay params; sharing them ties the two directions' lengthscales). Both directions run the
same sm_120-verified kernels as temporal/_scan.py (causal_conv1d_fn + mamba_chunk_scan_combined);
the backward direction is realised by flipping the sequence through the same left-causal kernels.

Unlike the temporal scan there is NO carried state: a frame's spatial sequence is complete
(initial_states=None, return_final_states=False) — the module is stateless by construction,
which is what keeps bidirectionality causally safe (space within one frame, never time).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.ops.triton.layernorm_gated import RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pure-PyTorch fallback, mirrors temporal/_scan.py
    causal_conv1d_fn = None


def _init_dt_bias(nheads, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
    """Mamba-2's dt_bias init (inverse-softplus of log-uniform dt), per direction."""
    dt = torch.exp(torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
    dt = torch.clamp(dt, min=dt_init_floor)
    return dt + torch.log(-torch.expm1(-dt))          # inv_softplus(dt)


def _init_a_log(nheads, a_init_range=(1, 16)):
    return torch.log(torch.empty(nheads).uniform_(*a_init_range))


class BiMamba1DScan(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, chunk_size: int = 256):
        super().__init__()
        assert (d_model * expand) % headdim == 0, (
            f"d_model*expand ({d_model * expand}) must be divisible by headdim ({headdim})")
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        self.headdim, self.nheads = headdim, self.d_inner // headdim
        self.conv_dim = self.d_inner + 2 * d_state    # ngroups=1
        self.chunk_size = chunk_size

        # shared: z | xBC | dt  (same split as temporal/_scan.py:51-52)
        self.in_proj = nn.Linear(d_model, self.d_inner + self.conv_dim + self.nheads, bias=False)
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False,
                                 group_size=self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # per-direction params
        for tag in ("fwd", "bwd"):
            conv = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv,
                             groups=self.conv_dim, padding=d_conv - 1)
            setattr(self, f"conv1d_{tag}", conv)
            setattr(self, f"A_log_{tag}", nn.Parameter(_init_a_log(self.nheads)))
            setattr(self, f"dt_bias_{tag}", nn.Parameter(_init_dt_bias(self.nheads)))
            setattr(self, f"D_{tag}", nn.Parameter(torch.ones(self.nheads)))

    def _direction(self, xBC, dt, conv1d, A_log, dt_bias, D):
        """One directional scan on an already-oriented (N,S,·) sequence."""
        xBC_t = rearrange(xBC, "n s d -> n d s")
        if causal_conv1d_fn is not None:
            conv_out = causal_conv1d_fn(xBC_t, rearrange(conv1d.weight, "d 1 w -> d w"),
                                        bias=conv1d.bias, activation="silu")
        else:
            conv_out = F.silu(conv1d(xBC_t)[..., : xBC_t.shape[-1]])
        xBC = rearrange(conv_out, "n d s -> n s d")
        x_ssm, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x_ssm, "n s (h p) -> n s h p", p=self.headdim),
            dt, -torch.exp(A_log.float()),
            rearrange(B, "n s (g d) -> n s g d", g=1),
            rearrange(C, "n s (g d) -> n s g d", g=1),
            chunk_size=self.chunk_size, D=D, z=None,
            dt_bias=dt_bias, dt_softplus=True,
            initial_states=None, return_final_states=False,
        )
        return rearrange(y, "n s h p -> n s (h p)")

    def forward(self, x):                              # (N, S, d_model)
        z, xBC, dt = torch.split(self.in_proj(x),
                                 [self.d_inner, self.conv_dim, self.nheads], dim=-1)
        y_fwd = self._direction(xBC, dt,
                                self.conv1d_fwd, self.A_log_fwd, self.dt_bias_fwd, self.D_fwd)
        y_bwd = self._direction(xBC.flip(1), dt.flip(1),
                                self.conv1d_bwd, self.A_log_bwd, self.dt_bias_bwd, self.D_bwd
                                ).flip(1)
        y = self.norm(y_fwd + y_bwd, z)                # ONE shared gate over the summed directions
        return self.out_proj(y)
```

`code/event_ssm/spatial/__init__.py`:
```python
from event_ssm.spatial._scan2d import BiMamba1DScan

__all__ = ["BiMamba1DScan"]
```

- [ ] **Step 5: Run to verify pass**

Run: `$PY -m pytest code/event_ssm/tests/test_spatial_scan2d.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/spatial/ code/event_ssm/tests/test_spatial_scan2d.py
git commit -m "feat(stage11): BiMamba1DScan bidirectional spatial scan core (shared proj/gate, per-direction conv/A/dt/D)"
```

---

### Task 2: `BiMamba1DScan` correctness battery + U1 visual proof

**Files:**
- Modify: `code/event_ssm/tests/test_spatial_scan2d.py` (append tests)
- Create: `code/event_ssm/scripts/stage11_u1_proof.py`

**Interfaces:**
- Consumes: `BiMamba1DScan` from Task 1 (exact signature above).
- Produces: `proofs/out/u1_scan_equivalence.png`; confidence that both directions are live and kernel-vs-reference agree.

- [ ] **Step 1: Append the four correctness tests (write first, watch them fail only if implementation is wrong — they may pass immediately; that is acceptable here because Task 1's implementation already exists; a test that fails unexpectedly means STOP and debug via superpowers:systematic-debugging)**

Append to `code/event_ssm/tests/test_spatial_scan2d.py`:
```python
def test_kernel_matches_reference(device):
    # fwd-direction chunk-scan kernel vs pure-PyTorch reference (spec §6 U1, gate ~1e-3 fp32)
    from mamba_ssm.ops.triton.ssd_combined import (mamba_chunk_scan_combined,
                                                   ssd_chunk_scan_combined_ref)
    torch.manual_seed(1)
    n, s, h, p, dstate, chunk = 2, 320, 2, 64, 16, 256
    x = torch.randn(n, s, h, p, device=device)
    dt = torch.nn.functional.softplus(torch.randn(n, s, h, device=device))
    A = -torch.exp(torch.randn(h, device=device))
    B = torch.randn(n, s, 1, dstate, device=device)
    C = torch.randn(n, s, 1, dstate, device=device)
    y_kernel = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=chunk, D=None,
                                         dt_softplus=False)
    y_ref = ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size=chunk, D=None)
    assert torch.allclose(y_kernel, y_ref, atol=1e-3, rtol=1e-3), \
        f"max err {(y_kernel - y_ref).abs().max().item():.2e}"


def test_flip_equivariance_with_mirrored_params(device):
    # copy fwd params into bwd -> module must commute with sequence flip (spec §6 U1)
    m = _make(device=device)
    with torch.no_grad():
        m.conv1d_bwd.weight.copy_(m.conv1d_fwd.weight)
        m.conv1d_bwd.bias.copy_(m.conv1d_fwd.bias)
        m.A_log_bwd.copy_(m.A_log_fwd)
        m.dt_bias_bwd.copy_(m.dt_bias_fwd)
        m.D_bwd.copy_(m.D_fwd)
    x = torch.randn(2, 200, 64, device=device)
    y1 = m(x.flip(1))
    y2 = m(x).flip(1)
    assert torch.allclose(y1, y2, atol=1e-4, rtol=1e-4)


def test_backward_direction_is_live(device):
    # perturbing the LAST token must change the FIRST output (only the bwd path can do that)
    m = _make(device=device)
    x = torch.randn(1, 200, 64, device=device)
    x2 = x.clone()
    x2[:, -1] += 1.0
    assert not torch.allclose(m(x)[:, 0], m(x2)[:, 0]), \
        "output[0] insensitive to input[-1]: backward direction dead"


def test_gradients_reach_both_directions(device):
    m = _make(device=device)
    m(torch.randn(2, 96, 64, device=device)).square().mean().backward()
    for tag in ("fwd", "bwd"):
        for name in (f"A_log_{tag}", f"dt_bias_{tag}", f"D_{tag}"):
            g = getattr(m, name).grad
            assert g is not None and g.abs().sum() > 0, f"no gradient into {name}"
        assert getattr(m, f"conv1d_{tag}").weight.grad.abs().sum() > 0


def test_bf16_autocast_no_nan(device):
    # VMamba documents fp16 scan instability; bf16 is the repo-verified regime — verify explicitly
    m = _make(device=device)
    x = 50.0 * torch.randn(2, 1280, 64, device=device)   # deliberately large activations
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x)
    assert torch.isfinite(y.float()).all()
```

- [ ] **Step 2: Run the battery**

Run: `$PY -m pytest code/event_ssm/tests/test_spatial_scan2d.py -v`
Expected: 7 passed. Any failure → superpowers:systematic-debugging before proceeding (likely suspects: reference-function signature drift → inspect `ssd_combined.py` source; flip test tolerance under TF32 → set `torch.backends.cuda.matmul.allow_tf32 = False` inside the test).

- [ ] **Step 3: U1 visual proof script** (load `dataviz` skill before writing figure code)

`code/event_ssm/scripts/stage11_u1_proof.py`:
```python
#!/usr/bin/env python
"""U1 proof: kernel-vs-reference error histogram + per-direction output-norm bars."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from event_ssm.spatial import BiMamba1DScan  # noqa: E402
from mamba_ssm.ops.triton.ssd_combined import (mamba_chunk_scan_combined,  # noqa: E402
                                               ssd_chunk_scan_combined_ref)

OUT = REPO / "code" / "event_ssm" / "proofs" / "out"
OUT.mkdir(parents=True, exist_ok=True)
OURS = "#2a78d6"

torch.manual_seed(0)
n, s, h, p, dstate = 2, 1280, 2, 64, 16
x = torch.randn(n, s, h, p, device="cuda")
dt = F.softplus(torch.randn(n, s, h, device="cuda"))
A = -torch.exp(torch.randn(h, device="cuda"))
B = torch.randn(n, s, 1, dstate, device="cuda")
C = torch.randn(n, s, 1, dstate, device="cuda")
err = (mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=256, D=None, dt_softplus=False)
       - ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size=256, D=None)).abs()

m = BiMamba1DScan(64).cuda()
xin = torch.randn(1, 1280, 64, device="cuda")
z, xBC, dtm = torch.split(m.in_proj(xin), [m.d_inner, m.conv_dim, m.nheads], dim=-1)
y_f = m._direction(xBC, dtm, m.conv1d_fwd, m.A_log_fwd, m.dt_bias_fwd, m.D_fwd)
y_b = m._direction(xBC.flip(1), dtm.flip(1), m.conv1d_bwd, m.A_log_bwd,
                   m.dt_bias_bwd, m.D_bwd).flip(1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
ax1.hist(err.flatten().float().cpu().numpy(), bins=60, color=OURS)
ax1.set_yscale("log")
ax1.set_xlabel("|kernel − reference|")
ax1.set_title(f"chunk-scan vs reference (max {err.max():.1e})")
ax2.bar(["forward", "backward"],
        [y_f.norm().item(), y_b.norm().item()], color=OURS)
ax2.set_title("per-direction output norm (both live)")
fig.tight_layout()
fig.savefig(OUT / "u1_scan_equivalence.png", dpi=160)
print(f"wrote {OUT / 'u1_scan_equivalence.png'}; kernel-ref max err {err.max():.2e}")
```

- [ ] **Step 4: Run it; eyeball the figure**

Run: `$PY code/event_ssm/scripts/stage11_u1_proof.py`
Expected: prints `wrote .../u1_scan_equivalence.png; kernel-ref max err ~1e-4..1e-3`; both bars non-zero.

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/tests/test_spatial_scan2d.py code/event_ssm/scripts/stage11_u1_proof.py code/event_ssm/proofs/out/u1_scan_equivalence.png
git commit -m "test(stage11): scan correctness battery (reference equivalence, flip equivariance, direction liveness, bf16) + U1 proof figure"
```

---

### Task 3: `BiMamba2DBlock` (+ vendored DropPath, LayerNorm2d)

**Files:**
- Create: `code/event_ssm/spatial/bimamba_block.py`
- Modify: `code/event_ssm/spatial/__init__.py`
- Test: `code/event_ssm/tests/test_bimamba_spatial.py`

**Interfaces:**
- Consumes: `BiMamba1DScan` (Task 1).
- Produces: `BiMamba2DBlock(d_model, axis, d_state=16, d_conv=4, expand=2, headdim=64, drop_path=0.0)` with `axis ∈ {"row","col"}`; `forward(x: (N,C,H,W)) -> (N,C,H,W)`. Also `DropPath(p)`, `LayerNorm2d(C)` (reused by Task 4).

- [ ] **Step 1: Write the failing tests**

`code/event_ssm/tests/test_bimamba_spatial.py`:
```python
# Stage 11 U2: BiMamba 2D block + 4-stage pyramid (spec §4, §6 U2)
import pytest
import torch


def test_block_shape_row_and_col(device):
    from event_ssm.spatial import BiMamba2DBlock
    torch.manual_seed(0)
    for axis in ("row", "col"):
        blk = BiMamba2DBlock(64, axis=axis).to(device)
        y = blk(torch.randn(2, 64, 16, 20, device=device))
        assert y.shape == (2, 64, 16, 20), axis


def test_block_rejects_bad_axis():
    from event_ssm.spatial import BiMamba2DBlock
    with pytest.raises(AssertionError):
        BiMamba2DBlock(64, axis="diag")


def test_dwconv_zero_init_starts_as_identity_mix(device):
    # local-mix residual is zero-initialised -> at init the block output equals
    # the pure scan path's output (dwconv contributes exactly nothing)
    from event_ssm.spatial import BiMamba2DBlock
    torch.manual_seed(0)
    blk = BiMamba2DBlock(64, axis="row").to(device)
    assert blk.dwconv.weight.abs().sum() == 0 and blk.dwconv.bias.abs().sum() == 0


def test_col_axis_mixes_along_columns(device):
    # a col-axis block must propagate a point perturbation within its column
    # far more than a row-axis block does at init
    from event_ssm.spatial import BiMamba2DBlock
    torch.manual_seed(0)
    blk = BiMamba2DBlock(64, axis="col").to(device).eval()
    x = torch.randn(1, 64, 16, 20, device=device)
    x2 = x.clone()
    x2[..., 2, 7] += 5.0                       # perturb (h=2, w=7)
    d = (blk(x2) - blk(x)).abs().sum(dim=1)[0]  # (H, W)
    col_effect = d[:, 7].sum() - d[2, 7]
    row_effect = d[2, :].sum() - d[2, 7]
    assert col_effect > row_effect, "col-axis block did not mix along its column"
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest code/event_ssm/tests/test_bimamba_spatial.py -v`
Expected: FAIL — `ImportError: cannot import name 'BiMamba2DBlock'`.

- [ ] **Step 3: Implement `bimamba_block.py`**

```python
"""BiMamba 2D block (Stage 11, spec §4.2): zero-init DWConv3x3 local mix + pre-norm
bidirectional scan along ONE axis (row- or column-major flatten; the pyramid alternates
axes across blocks — Mamba-ND finding). timm is absent from this env: DropPath is vendored."""
import torch
import torch.nn as nn
from einops import rearrange

from event_ssm.spatial._scan2d import BiMamba1DScan


class DropPath(nn.Module):
    """Per-sample stochastic depth (vendored; timm not installable under --no-deps policy)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels for (N,C,H,W) maps (no BatchNorm anywhere: spec §4.1)."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class BiMamba2DBlock(nn.Module):
    def __init__(self, d_model: int, axis: str, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, drop_path: float = 0.0):
        super().__init__()
        assert axis in ("row", "col"), f"axis must be 'row' or 'col', got {axis!r}"
        self.axis = axis
        # local mix: zero-init depthwise 3x3 -> block starts as pure global scan (stable from-scratch start)
        self.dwconv = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
        nn.init.zeros_(self.dwconv.weight)
        nn.init.zeros_(self.dwconv.bias)
        self.norm = nn.LayerNorm(d_model)
        self.scan = BiMamba1DScan(d_model, d_state=d_state, d_conv=d_conv,
                                  expand=expand, headdim=headdim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):                                  # (N, C, H, W)
        x = x + self.dwconv(x)
        n, c, h, w = x.shape
        t = self.norm(rearrange(x, "n c h w -> n h w c"))
        if self.axis == "row":
            y = self.scan(rearrange(t, "n h w c -> n (h w) c"))
            t = rearrange(y, "n (h w) c -> n h w c", h=h, w=w)
        else:
            y = self.scan(rearrange(t, "n h w c -> n (w h) c"))
            t = rearrange(y, "n (w h) c -> n h w c", h=h, w=w)
        return x + self.drop_path(rearrange(t, "n h w c -> n c h w"))
```

Update `code/event_ssm/spatial/__init__.py`:
```python
from event_ssm.spatial._scan2d import BiMamba1DScan
from event_ssm.spatial.bimamba_block import BiMamba2DBlock, DropPath, LayerNorm2d

__all__ = ["BiMamba1DScan", "BiMamba2DBlock", "DropPath", "LayerNorm2d"]
```

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m pytest code/event_ssm/tests/test_bimamba_spatial.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/spatial/ code/event_ssm/tests/test_bimamba_spatial.py
git commit -m "feat(stage11): BiMamba2DBlock with axis-alternating scan, zero-init DWConv local mix, vendored DropPath/LayerNorm2d"
```

---

### Task 4: `BiMambaSpatialStages` — the duck-type pyramid

**Files:**
- Create: `code/event_ssm/spatial/bimamba_spatial.py`
- Modify: `code/event_ssm/spatial/__init__.py`
- Test: `code/event_ssm/tests/test_bimamba_spatial.py` (append)

**Interfaces:**
- Consumes: `BiMamba2DBlock`, `LayerNorm2d` (Task 3).
- Produces: `BiMambaSpatialStages(in_channels=20, depths=(2,2,8,2), d_state=16, d_conv=4, expand=2, headdim=64, drop_path_rate=0.1)` — class attrs `stage_dims=(64,128,256,512)`, `strides=(4,8,16,32)`; `forward((N,20,H,W)) -> dict{1..4}`. This is the exact duck type `ResNetMambaBackbone` consumes (Task 5).

- [ ] **Step 1: Append the failing tests**

Append to `code/event_ssm/tests/test_bimamba_spatial.py`:
```python
def _stages(device, **kw):
    from event_ssm.spatial import BiMambaSpatialStages
    torch.manual_seed(0)
    return BiMambaSpatialStages(**kw).to(device)


def test_duck_type_matches_resnet_spatial(device):
    from event_ssm.backbone.resnet_spatial import ResNetSpatialStages
    from event_ssm.spatial import BiMambaSpatialStages
    assert BiMambaSpatialStages.stage_dims == ResNetSpatialStages.stage_dims
    assert BiMambaSpatialStages.strides == ResNetSpatialStages.strides


def test_forward_shapes_padded_gen1(device):
    m = _stages(device)
    feats = m(torch.randn(2, 20, 256, 320, device=device))
    assert set(feats.keys()) == {1, 2, 3, 4}
    assert feats[1].shape == (2, 64, 64, 80)
    assert feats[2].shape == (2, 128, 32, 40)
    assert feats[3].shape == (2, 256, 16, 20)
    assert feats[4].shape == (2, 512, 8, 10)


def test_stateless_and_deterministic_in_eval(device):
    m = _stages(device).eval()
    x = torch.randn(1, 20, 256, 320, device=device)
    with torch.no_grad():
        a, b = m(x), m(x)
    for k in a:
        assert torch.equal(a[k], b[k]), f"stage {k} not deterministic/stateless"


def test_param_budget_gate(device):
    m = _stages(device)
    p = sum(t.numel() for t in m.parameters())
    assert 8e6 <= p <= 16e6, f"spatial params {p/1e6:.2f} M outside the 8-16 M gate (spec §4.3)"


def test_axes_alternate_within_every_stage(device):
    m = _stages(device)
    for stage in m.stages:
        axes = [blk.axis for blk in stage]
        assert axes == ["row" if j % 2 == 0 else "col" for j in range(len(axes))]


def test_no_batchnorm_anywhere(device):
    m = _stages(device)
    assert not any(isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
                   for mod in m.modules())
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest code/event_ssm/tests/test_bimamba_spatial.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'BiMambaSpatialStages'`.

- [ ] **Step 3: Implement `bimamba_spatial.py`**

```python
"""BiMambaSpatialStages (Stage 11, spec §4): fully-pure 4-stage BiMamba pyramid.
Duck-type drop-in for ResNetSpatialStages (backbone/resnet_spatial.py): same stage_dims,
strides, and forward contract, so ResNetMambaBackbone's temporal path, state handling,
and the RVT wiring need zero changes. Stateless: runs on time-folded (L*B) frames.

Conv appears ONLY as stem/downsampling (position information — no positional embeddings
needed, VMamba evidence) and the in-block zero-init DWConv3x3. All token MIXING is SSM
(spiking-fork property, spec §10). No BatchNorm (LayerNorm family only)."""
import torch
import torch.nn as nn

from event_ssm.spatial.bimamba_block import BiMamba2DBlock, LayerNorm2d


class BiMambaSpatialStages(nn.Module):
    stage_dims = (64, 128, 256, 512)
    strides = (4, 8, 16, 32)

    def __init__(self, in_channels: int = 20, depths=(2, 2, 8, 2), d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, headdim: int = 64,
                 drop_path_rate: float = 0.1):
        super().__init__()
        dims = self.stage_dims
        self.depths = tuple(depths)
        self.stem = nn.Sequential(                       # stride 4 (two 3x3 s2 convs)
            nn.Conv2d(in_channels, dims[0] // 2, 3, 2, 1),
            LayerNorm2d(dims[0] // 2), nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, 2, 1),
            LayerNorm2d(dims[0]),
        )
        self.downsamples = nn.ModuleList(
            nn.Sequential(nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1), LayerNorm2d(dims[i + 1]))
            for i in range(3)
        )
        dp = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages, k = nn.ModuleList(), 0
        for i, depth in enumerate(depths):
            self.stages.append(nn.Sequential(*[
                BiMamba2DBlock(dims[i], axis=("row" if j % 2 == 0 else "col"),
                               d_state=d_state, d_conv=d_conv, expand=expand,
                               headdim=headdim, drop_path=dp[k + j])
                for j in range(depth)
            ]))
            k += depth

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        feats = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats[i + 1] = x
            if i < 3:
                x = self.downsamples[i](x)
        return feats
```

Update `code/event_ssm/spatial/__init__.py`:
```python
from event_ssm.spatial._scan2d import BiMamba1DScan
from event_ssm.spatial.bimamba_block import BiMamba2DBlock, DropPath, LayerNorm2d
from event_ssm.spatial.bimamba_spatial import BiMambaSpatialStages

__all__ = ["BiMamba1DScan", "BiMamba2DBlock", "DropPath", "LayerNorm2d",
           "BiMambaSpatialStages"]
```

- [ ] **Step 4: Run to verify pass; check the printed param count**

Run: `$PY -m pytest code/event_ssm/tests/test_bimamba_spatial.py -v`
Expected: 10 passed. If `test_param_budget_gate` fails LOW (<8 M), increase stage-3 depth 8→10 and rerun; if HIGH (>16 M), decrease 8→6. Record the final depths in the commit message — this is the only sanctioned tuning knob in this task.

- [ ] **Step 5: U2 visual proof — scan-order figure** (load `dataviz` skill first)

Create `code/event_ssm/scripts/stage11_u2_proof.py`:
```python
#!/usr/bin/env python
"""U2 proof: render the actual row/col scan orders on a small grid + per-stage shape table."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from event_ssm.spatial import BiMambaSpatialStages  # noqa: E402

OUT = REPO / "code" / "event_ssm" / "proofs" / "out"
OUT.mkdir(parents=True, exist_ok=True)

h, w = 6, 8
row_order = np.arange(h * w).reshape(h, w)                # n (h w) c flatten
col_order = np.arange(h * w).reshape(w, h).T              # n (w h) c flatten
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, order, title in ((axes[0], row_order, "row-axis block: scan index"),
                         (axes[1], col_order, "col-axis block: scan index")):
    ax.imshow(order, cmap="viridis")
    for (i, j), v in np.ndenumerate(order):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="w")
    ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "u2_scan_order.png", dpi=160)

m = BiMambaSpatialStages().cuda()
feats = m(torch.randn(1, 20, 256, 320, device="cuda"))
p = sum(t.numel() for t in m.parameters())
lines = ["| stage | shape | axis pattern |", "|---|---|---|"]
for i, stage in enumerate(m.stages):
    lines.append(f"| {i+1} | {tuple(feats[i+1].shape)} | "
                 f"{'/'.join(b.axis for b in stage)} |")
lines.append(f"\nspatial params: **{p/1e6:.2f} M** (gate 8-16 M)")
(OUT / "u2_stage_table.md").write_text("\n".join(lines) + "\n")
print(f"wrote u2_scan_order.png + u2_stage_table.md; params {p/1e6:.2f} M")
```

Run: `$PY code/event_ssm/scripts/stage11_u2_proof.py`
Expected: figure + table written; params printed within gate.

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/spatial/ code/event_ssm/tests/test_bimamba_spatial.py code/event_ssm/scripts/stage11_u2_proof.py code/event_ssm/proofs/out/u2_scan_order.png code/event_ssm/proofs/out/u2_stage_table.md
git commit -m "feat(stage11): BiMambaSpatialStages 4-stage pyramid (depths 2/2/8/2, duck-type of ResNetSpatialStages, param gate test)"
```

---

### Task 5: Spatial injection into `ResNetMambaBackbone`

**Files:**
- Modify: `code/event_ssm/backbone/resnet_mamba.py:40-43`
- Test: `code/event_ssm/tests/test_bimamba_spatial.py` (append)

**Interfaces:**
- Consumes: `BiMambaSpatialStages` (Task 4); existing `ResNetMambaBackbone` contract (`forward(x:(L,B,C,H,W), prev_states) -> (feats dict{1..4} of (L,B,c,h,w), states list len 4)`).
- Produces: `ResNetMambaBackbone(..., spatial: nn.Module | None = None)` — when given, the module is used instead of constructing `ResNetSpatialStages`. Stage 12's register branch will call exactly this.

- [ ] **Step 1: Append the failing integration tests**

Append to `code/event_ssm/tests/test_bimamba_spatial.py`:
```python
def test_backbone_accepts_injected_spatial(device):
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
    from event_ssm.spatial import BiMambaSpatialStages
    torch.manual_seed(0)
    bb = ResNetMambaBackbone(spatial=BiMambaSpatialStages()).to(device)
    x = torch.randn(2, 1, 20, 256, 320, device=device)     # (L=2, B=1)
    feats, states = bb(x, None)
    assert set(feats.keys()) == {1, 2, 3, 4}
    assert feats[2].shape == (2, 1, 128, 32, 40)            # (L,B,c,h,w) layout kept
    assert feats[4].shape == (2, 1, 512, 8, 10)
    # state contract unchanged (spec §3): list len 4; stage-1 placeholder (B,1);
    # temporal stages dim0=B, no Nones anywhere
    assert len(states) == 4
    assert states[0].shape == (1, 1)
    for st in states[1:]:
        for conv_b, ssm_b in st:
            assert conv_b.shape[0] == 1 and ssm_b.shape[0] == 1
            assert not conv_b.requires_grad and not ssm_b.requires_grad


def test_backbone_state_carry_streaming(device):
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
    from event_ssm.spatial import BiMambaSpatialStages
    torch.manual_seed(0)
    bb = ResNetMambaBackbone(spatial=BiMambaSpatialStages()).to(device).eval()
    x = torch.randn(1, 1, 20, 256, 320, device=device)
    with torch.no_grad():
        _, s1 = bb(x, None)
        _, s2 = bb(x, s1)                                   # streaming step with carried state
    assert not torch.allclose(s1[1][0][1], s2[1][0][1]), "temporal state did not evolve"


def test_default_backbone_unchanged(device):
    # regression guard: default construction still builds ResNetSpatialStages
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
    from event_ssm.backbone.resnet_spatial import ResNetSpatialStages
    bb = ResNetMambaBackbone(pretrained=False)
    assert isinstance(bb.spatial, ResNetSpatialStages)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest code/event_ssm/tests/test_bimamba_spatial.py -k backbone -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'spatial'`.

- [ ] **Step 3: Minimal modification to `resnet_mamba.py`**

Change the `__init__` signature and first line (lines 40–43); everything else stays:
```python
    def __init__(self, in_channels: int = 20, pretrained: bool = True, d_state: int = 64,
                 num_layers_per_stage: int = 1, temporal_stages=(2, 3, 4),
                 spatial: nn.Module = None):
        super().__init__()
        # Stage 11: optional spatial-module injection (duck type of ResNetSpatialStages:
        # stage_dims/strides attrs + forward (N,C,H,W)->dict{1..4}). Default unchanged.
        self.spatial = spatial if spatial is not None else ResNetSpatialStages(in_channels, pretrained)
```

- [ ] **Step 4: Run the new tests AND the full existing suite (regression gate)**

Run: `$PY -m pytest code/event_ssm/tests/ -v`
Expected: all pass — the pre-existing 60 plus every Stage-11 test added so far. Any pre-existing test failing = STOP, the injection broke the default path.

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/backbone/resnet_mamba.py code/event_ssm/tests/test_bimamba_spatial.py
git commit -m "feat(stage11): optional spatial-module injection in ResNetMambaBackbone (default path unchanged)"
```

---

### Task 6: U3 — latency/VRAM probe (STAGE GATE)

**Files:**
- Create: `code/event_ssm/scripts/stage11_probe.py`

**Interfaces:**
- Consumes: `ResNetMambaBackbone(spatial=BiMambaSpatialStages())` (Task 5).
- Produces: `proofs/out/u3_probe.json`, `proofs/out/u3_probe_table.md`, `proofs/out/u3_probe_latency.png`. Exit code 0 iff both gates pass (fail-closed, Stage-10 convention).

- [ ] **Step 1: Write the probe script** (load `dataviz` skill before the chart section)

```python
#!/usr/bin/env python
"""Stage-11 U3 probe (STAGE GATE, spec §6 U3). Run on an IDLE GPU (close browsers/trainers;
check `nvidia-smi` first — Stage-10 lesson). Gates:
  G1: projected full pipeline >= 51 Hz  (backbone streaming p50 + 7.2 ms Stage-10 neck+head)
  G2: train-shape (L=21,B=4) forward+backward peak VRAM < 16 GB
Exit code is non-zero on gate failure (fail-closed). Fallback ladder is a USER decision:
(a) stage-1 depth 2->1, (b) stage-1 d_state 16->8, (c) conv stage 1 (revisits purity)."""
import json
import pathlib
import sys
import time

import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone  # noqa: E402
from event_ssm.spatial import BiMambaSpatialStages  # noqa: E402

OUT = REPO / "code" / "event_ssm" / "proofs" / "out"
OUT.mkdir(parents=True, exist_ok=True)
NECK_HEAD_MS = 7.2          # Stage-10 measured (bench_results.json neck_head p50)
EVENTSSM_BACKBONE_MS = 5.83  # Stage-10 reference line
GATE_HZ, GATE_GB = 51.0, 16.0


def p50_ms(fn, warmup=50, iters=300):
    for i in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]


def main():
    torch.manual_seed(0)
    bb = ResNetMambaBackbone(spatial=BiMambaSpatialStages()).cuda().eval()
    spatial = bb.spatial

    # -- streaming latency (B=1, L=1, state carried), bf16 like Stage 10 --
    x1 = torch.zeros(1, 1, 20, 256, 320, device="cuda")
    state = {"s": None}

    def step():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, state["s"] = bb(x1, state["s"])

    backbone_ms = p50_ms(step)

    xf = torch.zeros(1, 20, 256, 320, device="cuda")

    def spatial_only():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            spatial(xf)

    spatial_ms = p50_ms(spatial_only)

    # -- train-shape VRAM --
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    xt = torch.zeros(21, 4, 20, 256, 320, device="cuda")
    bb.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats, _ = bb(xt, None)
        loss = sum(f.float().square().mean() for f in feats.values())
    loss.backward()
    train_gb = torch.cuda.max_memory_allocated() / 2**30

    pipeline_hz = 1000.0 / (backbone_ms + NECK_HEAD_MS)
    g1, g2 = pipeline_hz >= GATE_HZ, train_gb < GATE_GB
    res = {"backbone_p50_ms": round(backbone_ms, 3), "spatial_only_p50_ms": round(spatial_ms, 3),
           "neck_head_ms_ref": NECK_HEAD_MS, "projected_pipeline_hz": round(pipeline_hz, 2),
           "train_peak_vram_gb": round(train_gb, 2), "gate_51hz": g1, "gate_16gb": g2,
           "device": torch.cuda.get_device_name(0)}
    (OUT / "u3_probe.json").write_text(json.dumps(res, indent=2))
    rows = [
        "| metric | value | gate | pass |", "|---|---|---|---|",
        f"| backbone streaming p50 | {backbone_ms:.2f} ms | <= 12.5 ms | {'✅' if g1 else '❌'} |",
        f"| projected pipeline | {pipeline_hz:.1f} Hz | >= 51 Hz | {'✅' if g1 else '❌'} |",
        f"| train step peak VRAM | {train_gb:.2f} GB | < 16 GB | {'✅' if g2 else '❌'} |",
        f"| spatial module alone | {spatial_ms:.2f} ms | (EventSSM ResNet ref 5.83 ms) | — |",
    ]
    (OUT / "u3_probe_table.md").write_text("\n".join(rows) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(["EventSSM backbone (ref)", "PureSSM backbone", "PureSSM spatial only"],
            [EVENTSSM_BACKBONE_MS, backbone_ms, spatial_ms],
            color=["#1baf7a", "#2a78d6", "#2a78d6"])
    ax.axvline(12.5, ls="--", c="gray")
    ax.text(12.5, 2.4, " 51 Hz gate", va="center", fontsize=8, color="gray")
    ax.set_xlabel("streaming p50 (ms)")
    fig.tight_layout()
    fig.savefig(OUT / "u3_probe_latency.png", dpi=160)

    print(json.dumps(res, indent=2))
    sys.exit(0 if (g1 and g2) else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Check GPU is idle, then run the probe**

Run: `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` — expect ~0 % / < 1 GB. Then:
`$PY code/event_ssm/scripts/stage11_probe.py`
Expected: JSON printed; exit 0; three artifacts in `proofs/out/`.

- [ ] **Step 3: GATE decision**

If exit 0: record the numbers in the commit message and proceed.
If exit 1: **STOP. Do not tune silently.** Report the numbers to the user with the fallback ladder from the script docstring — which rung to take is their call (spec U3).

- [ ] **Step 4: Commit**

```bash
git add code/event_ssm/scripts/stage11_probe.py code/event_ssm/proofs/out/u3_probe.json code/event_ssm/proofs/out/u3_probe_table.md code/event_ssm/proofs/out/u3_probe_latency.png
git commit -m "feat(stage11): U3 latency/VRAM probe gate — <numbers from u3_probe.json>"
```

---

### Task 7: U5 — untrained ERF mechanism figure

**Files:**
- Create: `code/event_ssm/scripts/stage11_erf.py`

**Interfaces:**
- Consumes: `BiMambaSpatialStages` (Task 4), `ResNetSpatialStages` (existing).
- Produces: `proofs/out/u5_erf_resnet_vs_bimamba.png` — the thesis mechanism figure (untrained variant; Stage 15 reruns with the trained checkpoint).

- [ ] **Step 1: Write the ERF script** (load `dataviz` skill first; sequential colormap, log scale)

```python
#!/usr/bin/env python
"""U5: gradient-based Effective Receptive Field, ResNet-18 stages vs BiMamba stages
(both UNTRAINED — architecture-intrinsic ERF; Stage 15 repeats with trained weights).
Method: d|f(center)|/dx aggregated over channels (Luo et al. 2016 style)."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages  # noqa: E402
from event_ssm.spatial import BiMambaSpatialStages  # noqa: E402

OUT = REPO / "code" / "event_ssm" / "proofs" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def erf(module, stage: int) -> torch.Tensor:
    module = module.cuda().eval()
    x = torch.zeros(1, 20, 256, 320, device="cuda", requires_grad=True)
    f = module(x)[stage]
    h, w = f.shape[-2:]
    f[0, :, h // 2, w // 2].abs().sum().backward()
    g = x.grad.abs().sum(dim=1)[0]
    return (g / g.max().clamp(min=1e-12)).cpu()


torch.manual_seed(0)
models = {"ResNet-18 (EventSSM)": ResNetSpatialStages(pretrained=False),
          "BiMamba (PureSSM)": BiMambaSpatialStages()}
stages = (3, 4)
fig, axes = plt.subplots(len(models), len(stages), figsize=(8, 6.5))
for r, (name, m) in enumerate(models.items()):
    for c, s in enumerate(stages):
        ax = axes[r][c]
        ax.imshow(torch.log1p(100 * erf(m, s)), cmap="magma")
        ax.set_title(f"{name} — stage {s}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
fig.suptitle("Effective receptive field at the frame centre (untrained, log scale)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "u5_erf_resnet_vs_bimamba.png", dpi=160)
print(f"wrote {OUT / 'u5_erf_resnet_vs_bimamba.png'}")
```

- [ ] **Step 2: Run and eyeball**

Run: `$PY code/event_ssm/scripts/stage11_erf.py`
Expected: figure written. Expected visual: ResNet panels show a compact local blob; BiMamba panels show cross/frame-spanning support (rows+cols reach). If BiMamba's ERF looks as local as ResNet's, the alternating scan is not mixing — STOP and debug (suspect: axis flatten rearranges in `BiMamba2DBlock`).

- [ ] **Step 3: Commit**

```bash
git add code/event_ssm/scripts/stage11_erf.py code/event_ssm/proofs/out/u5_erf_resnet_vs_bimamba.png
git commit -m "feat(stage11): U5 untrained ERF figure — BiMamba global support vs ResNet local blob"
```

---

### Task 8: Stage close-out

**Files:**
- Create: `docs/Stage11_build_notes.md`
- Modify: `CLAUDE.md` (status bullet), `graphify-out/` (via `graphify update .`)

- [ ] **Step 1: Full suite + proofs inventory**

Run: `$PY -m pytest code/event_ssm/tests/ -v`
Expected: everything passes (pre-existing 60 + ~13 new). Then `ls code/event_ssm/proofs/out/ | grep -E "u[1235]"` — expect the 5 Stage-11 artifacts.

- [ ] **Step 2: Write `docs/Stage11_build_notes.md`** — probe numbers, final depths/params, any non-obvious fixes made during the tasks (document-changes convention), links to the four proof artifacts.

- [ ] **Step 3: Run `graphify update .`** (keep the knowledge graph current per repo rules).

- [ ] **Step 4: Code review + merge decision**

Use superpowers:requesting-code-review (fresh-eyes review of the branch diff), then superpowers:verification-before-completion (re-run suite + probe, confirm outputs exist), then superpowers:finishing-a-development-branch (merge `stage11-puressm-backbone` → main or PR, user's call).

- [ ] **Step 5: Commit close-out**

```bash
git add docs/Stage11_build_notes.md CLAUDE.md graphify-out/
git commit -m "docs(stage11): build notes, probe numbers, status update — Stage 11 complete"
```

---

## Self-Review (run before handing off)

1. **Spec coverage:** U1→Tasks 1–2; U2→Tasks 3–4; U3→Task 6; U5→Task 7; spatial-injection seam→Task 5; U4/registration is Stage 12 by design (roadmap). ✅
2. **Placeholders:** none — every step has literal code/commands/expected output. The one intentional degree of freedom (stage-3 depth if the param gate trips) has explicit bounds and instructions. ✅
3. **Type consistency:** `BiMamba1DScan(d_model, d_state, d_conv, expand, headdim, chunk_size)` used identically in Tasks 1/3; `BiMamba2DBlock(d_model, axis, …, drop_path)` in Tasks 3/4; `BiMambaSpatialStages()` defaults in Tasks 4/5/6/7; `spatial=` kwarg in Tasks 5/6. ✅
