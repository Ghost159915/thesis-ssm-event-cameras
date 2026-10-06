# Stage 18 — SpikingSSM Backbone Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Stage-17 `SpikingSSMBlock` trainable/evaluable in the frozen RVT pipeline as `model=rnndet +experiment/gen1=spikingssm`, every ablation arm a CLI override.

**Architecture:** `SpikingSSMBackbone` subclasses `ResNetMambaBackbone` (unmodified), swaps the temporal block on `spiking_stages` for `SpikingSSMBlock`, and overrides `forward` only to route the `(mamba_state, mem)` state through spiking-aware reshape helpers. Spatial = PureSSM's BiMamba (ANN). Dispatch is an additive `"SpikingSSM"` branch in `integration/register.py`; configs are copies of PureSSM's plus a `spiking:` block.

**Tech Stack:** PyTorch 2.11 cu128, mamba-ssm 2.3.2.post1 (CUDA-only kernels), Hydra/OmegaConf, pytest, RVT (`external/ssms_event_cameras/RVT`).

**Spec:** `docs/specs/2026-10-06-stage18-spikingssm-integration-design.md`

## Global Constraints

- Byte-identical, never edit: `code/event_ssm/models/eventssm/`, `code/event_ssm/models/puressm/`, `code/event_ssm/temporal/`, `code/event_ssm/backbone/resnet_mamba.py`.
- The only edits to pre-existing harness files: the additive `"SpikingSSM"` branch + name-tuple entry in `code/event_ssm/integration/register.py`, and a new function appended to `code/event_ssm/integration/monitors.py`. Existing branches/functions unchanged.
- Training recipe of `experiment/gen1/spikingssm.yaml` byte-identical to `puressm.yaml` outside the model group.
- Python: `PY=~/miniforge3/envs/events_signals/bin/python`; run pytest from the repo root (`$PY -m pytest ...`). Default addopts exclude `gpu`; gpu-marked tests run with `-m gpu` (small unit tests only — the GPU must be idle; check `nvidia-smi` first, abort if another job is running).
- **Never** `pip install` anything into `events_signals`.
- Commits: conventional-commit style, **no Claude/Co-Authored-By trailers**, never commit Claude-named files. Run `git log -1 --format=%B` after each commit and amend away any auto-appended trailer.
- Comment density/style: match `code/event_ssm/models/spikingssm/*.py` (module docstrings explaining *why*, inline comments on non-obvious lines).

## File Map

| File | Action | Responsibility |
|---|---|---|
| `code/event_ssm/models/spikingssm/lif.py` | modify | lazy firing-rate tensor (no host sync); validate `beta` range |
| `code/event_ssm/tests/models/spikingssm/test_lif.py` | modify | tests for the two fixes |
| `code/event_ssm/tests/models/spikingssm/test_spiking_temporal.py` | modify | fix crashing `test_ssm_path_matches_the_non_spiking_block` |
| `code/event_ssm/models/spikingssm/backbone.py` | create | state helpers + `SpikingSSMBackbone` |
| `code/event_ssm/models/spikingssm/__init__.py` | modify | lazy-export `SpikingSSMBackbone` |
| `code/event_ssm/tests/models/spikingssm/test_backbone_cpu.py` | create | helpers round-trip, subset validation |
| `code/event_ssm/tests/models/spikingssm/test_backbone_gpu.py` | create | forward/backward, streaming parity, PureSSM equivalence, RVT state ops |
| `code/event_ssm/integration/monitors.py` | modify (append) | `attach_spiking_monitor` |
| `code/event_ssm/tests/models/spikingssm/test_spiking_monitor.py` | create | monitor CPU tests on a stub |
| `code/event_ssm/integration/register.py` | modify (additive) | `"SpikingSSM"` dispatch |
| `code/event_ssm/configs/spikingssm_yolox/default.yaml` | create | model config |
| `code/event_ssm/configs/experiment/gen1/spikingssm.yaml` | create | experiment config |
| `external/.../RVT/config/{model/spikingssm_yolox,experiment/gen1}/` | symlinks | Hydra discovery |
| `docs/patches/README.md` | modify (append) | symlink re-create commands |
| `code/event_ssm/tests/models/spikingssm/test_register_spikingssm.py` | create | dispatch, compose, overrides, recipe parity |
| `docs/notes/Stage18_integration_notes.md` | create | stage notes (incl. the two Stage-17 fixes) |
| `CLAUDE.md` | modify | status line for Stage 18 |

---

### Task 1: LIF fixes — lazy firing rate, beta-range validation, Stage-17 GPU test fix

**Files:**
- Modify: `code/event_ssm/models/spikingssm/lif.py`
- Modify: `code/event_ssm/tests/models/spikingssm/test_lif.py` (append)
- Modify: `code/event_ssm/tests/models/spikingssm/test_spiking_temporal.py` (`test_ssm_path_matches_the_non_spiking_block`)

**Interfaces:**
- Produces: `LIFReadout.last_firing_rate` → `float` (read-only property; `nan` before first forward). `LIFReadout(beta=...)` raises `AssertionError` unless `_BETA_EPS < beta < 1 - _BETA_EPS`.

Background: (a) `forward` ends with `self.last_firing_rate = float(spike_sum / ...)` — a GPU→host sync per stage per step that would contaminate Stage-22 latency. (b) `LIFReadout(4, beta=1e-4)` passes the `0 < beta < 1` assert but `p = (beta-eps)/(1-2eps) = 0` → `math.log(0)` → `ValueError: math domain error`. The pending GPU test `test_ssm_path_matches_the_non_spiking_block` uses exactly `beta=1e-4` and therefore crashes.

- [ ] **Step 1: Write the failing tests** — append to `test_lif.py`:

```python
def test_firing_rate_is_lazy_tensor_not_host_float():
    """forward must not force a host sync: the rate is stored as a detached tensor and only
    converted to float when read (Stage-22 latency would otherwise be contaminated)."""
    lif = LIFReadout(d_model=4)
    assert lif.last_firing_rate != lif.last_firing_rate          # nan before any forward
    lif(torch.full((2, 3, 4), 5.0))
    assert isinstance(lif._last_firing_rate, torch.Tensor)
    assert not lif._last_firing_rate.requires_grad
    assert isinstance(lif.last_firing_rate, float) and lif.last_firing_rate == 1.0


@pytest.mark.parametrize("beta", [1e-4, 1.0 - 1e-4, 1e-5])
def test_beta_at_or_beyond_eps_is_rejected_cleanly(beta):
    """beta == eps used to pass the (0,1) check and then crash in math.log(0)."""
    with pytest.raises(AssertionError, match="beta"):
        LIFReadout(d_model=4, beta=beta)


def test_beta_just_inside_eps_initialises_exactly():
    lif = LIFReadout(d_model=4, beta=2e-4, learn_beta=False)
    assert torch.allclose(lif.beta, torch.full((4,), 2e-4), atol=1e-7)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest code/event_ssm/tests/models/spikingssm/test_lif.py -q`
Expected: the 3 new tests FAIL (`AttributeError: _last_firing_rate`; `ValueError: math domain error` instead of `AssertionError`).

- [ ] **Step 3: Implement in `lif.py`**

Replace the beta assert in `__init__`:

```python
        assert _BETA_EPS < beta < 1.0 - _BETA_EPS, (
            f"beta must lie in ({_BETA_EPS}, {1.0 - _BETA_EPS}) — the epsilon-squeezed sigmoid's "
            f"open range, outside which its inverse is undefined; got {beta}")
```

Replace `self.last_firing_rate = float("nan")` in `__init__` with:

```python
        # Detached 0-d tensor written every forward; converted to float only when READ, so the
        # forward never forces a GPU->host sync (which would contaminate Stage-22 latency).
        # Read by training monitors watching for silence and saturation.
        self._last_firing_rate = None
```

Add the property next to `beta`/`threshold`:

```python
    @property
    def last_firing_rate(self) -> float:
        """Mean spike rate of the most recent forward (nan before the first). Host sync on read."""
        if self._last_firing_rate is None:
            return float("nan")
        return float(self._last_firing_rate)
```

Replace the last-but-one line of `forward`:

```python
        self._last_firing_rate = spike_sum / max(length, 1)          # detached; no host sync
```

- [ ] **Step 4: Fix the crashing Stage-17 GPU test** — in `test_spiking_temporal.py` replace the body of `test_ssm_path_matches_the_non_spiking_block` with:

```python
def test_ssm_path_matches_the_non_spiking_block():
    """The SSM recurrence must be untouched: the block's output equals the LIF readout applied
    to an independent, weight-identical MambaTemporalBlock — so the spiking layer is the ONLY
    change. (Previously used beta=1e-4 == _BETA_EPS, which crashed in the inverse sigmoid.)"""
    from event_ssm.temporal.mamba_temporal import MambaTemporalBlock
    blk = _block(d_model=128, output_mode="analog").eval()
    ref = MambaTemporalBlock(d_model=128).cuda().float().eval()
    ref.load_state_dict(blk.ssm.state_dict())
    x = torch.randn(8, 6, 128, device="cuda")
    with torch.no_grad():
        got, _ = blk(x)
        y, _ = ref(x)
        want, _ = blk.lif(y)
    assert (got - want).abs().max().item() < 1e-5
```

- [ ] **Step 5: Run all spiking CPU tests**

Run: `$PY -m pytest code/event_ssm/tests/models/spikingssm/ -q`
Expected: all pass (31 pre-existing + 5 new parametrised cases), 9 deselected.

- [ ] **Step 6: Commit**

```bash
git add code/event_ssm/models/spikingssm/lif.py code/event_ssm/tests/models/spikingssm/test_lif.py code/event_ssm/tests/models/spikingssm/test_spiking_temporal.py
git commit -m "fix(spikingssm): lazy firing-rate tensor, reject beta at the eps boundary"
```

---

### Task 2: `SpikingSSMBackbone` + spiking-aware state helpers

**Files:**
- Create: `code/event_ssm/models/spikingssm/backbone.py`
- Modify: `code/event_ssm/models/spikingssm/__init__.py`
- Create: `code/event_ssm/tests/models/spikingssm/test_backbone_cpu.py`
- Create: `code/event_ssm/tests/models/spikingssm/test_backbone_gpu.py`

**Interfaces:**
- Consumes: `ResNetMambaBackbone`, `_state_to_bmajor`, `_state_from_bmajor` (`event_ssm.backbone.resnet_mamba`); `SpikingSSMBlock(d_model, d_state, num_layers, residual, **lif_kwargs)`; `LIFReadout.last_firing_rate` (Task 1).
- Produces:
  - `_spk_state_to_bmajor(state, B, hw)`: `(mamba_list[(conv,ssm)], mem(N,C))` → `(mamba_list_bmajor, mem(B,hw,C))`
  - `_spk_state_from_bmajor(state_b, B, hw)`: inverse; `None` → `None`
  - `SpikingSSMBackbone(in_channels=20, d_state=64, num_layers_per_stage=1, temporal_stages=(2,3,4), spatial=None, spiking_stages=(2,3,4), residual=False, lif_kwargs=None, pretrained=True)`; raises `ValueError` if `spiking_stages ⊄ temporal_stages`.
  - attrs `spiking_stages: tuple`; method `spiking_stats() -> dict[int, dict[str, float]]` with keys `rate, beta_mean, beta_min, beta_max, thr_mean`.
  - `forward(x:(L,B,C,H,W), prev_states=None, token_mask=None, train_step=True) -> (feats{1..4}, states list[4])` (parent contract).

- [ ] **Step 1: Write failing CPU tests** — `test_backbone_cpu.py`:

```python
"""Stage 18 — SpikingSSMBackbone parts that need no CUDA kernels: the spiking-aware state
reshape helpers and constructor validation. (Importing backbone.py imports mamba_ssm, which
installs fine on this workstation; only its *kernels* need the GPU.)"""
import pytest
import torch

from event_ssm.models.spikingssm.backbone import (
    SpikingSSMBackbone, _spk_state_from_bmajor, _spk_state_to_bmajor)


def _state(N, C=16):
    mamba = [(torch.randn(N, 3, 40), torch.randn(N, 2, 64, 8)),     # (conv, ssm) per layer
             (torch.randn(N, 3, 40), torch.randn(N, 2, 64, 8))]
    return mamba, torch.randn(N, C)


def test_state_roundtrip_is_exact():
    B, hw = 2, 6
    st = _state(B * hw)
    st_b = _spk_state_to_bmajor(st, B, hw)
    (m_b, mem_b) = st_b
    assert mem_b.shape == (B, hw, 16), "dim0 must be B for RVT's RNNStates storage/reset"
    assert all(c.shape[0] == B and s.shape[0] == B for c, s in m_b)
    back_m, back_mem = _spk_state_from_bmajor(st_b, B, hw)
    assert torch.equal(back_mem, st[1])
    for (c0, s0), (c1, s1) in zip(st[0], back_m):
        assert torch.equal(c0, c1) and torch.equal(s0, s1)


def test_from_bmajor_none_passes_through():
    assert _spk_state_from_bmajor(None, 2, 6) is None


def test_spiking_stages_must_be_subset_of_temporal_stages():
    with pytest.raises(ValueError, match="spiking_stages"):
        SpikingSSMBackbone(temporal_stages=(3, 4), spiking_stages=(2, 3), pretrained=False)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest code/event_ssm/tests/models/spikingssm/test_backbone_cpu.py -q`
Expected: FAIL — `ModuleNotFoundError: event_ssm.models.spikingssm.backbone`.

- [ ] **Step 3: Implement `backbone.py`**

```python
"""SpikingSSM backbone — the Stage-18 deliverable (Thesis-C plan §3, choice C).

PureSSM's recurrent skeleton with the temporal Mamba block on `spiking_stages` replaced by
`SpikingSSMBlock` (Mamba-2 recurrence, numerically exact, + LIF readout). Spatial mixing stays
ANN (BiMamba, injected via `spatial=` exactly as the PureSSM dispatch does).

**Subclass, not edit.** `ResNetMambaBackbone` is reused unmodified; this class only (i) swaps the
modules in `self.temporal` for the spiking stages and (ii) overrides `forward` so each stage's
carried state goes through the right reshape helpers. The spiking state is `(mamba_state, mem)`
where the parent's helpers expect a bare per-layer `[(conv, ssm), ...]` list; `mem` is `(N, C)`
with dim0 = N = B*h*w precisely so the same `(N,..) <-> (B, hw, ..)` reshape applies
(Stage-17 notes, "Next — Stage 18").

`spiking_stages=()` makes this numerically the PureSSM backbone — the integration's own null test.
`spiking_stages ⊂ temporal_stages` is the de-risking ladder: (4,) -> (3,4) -> (2,3,4).

RVT's `RNNStates.recursive_detach/recursive_reset` recurse through lists/tuples of tensors, so the
nested state needs no RVT change; zeroing `mem` on a sequence reset is the correct LIF reset."""
from typing import Optional

import torch.nn as nn

from event_ssm.backbone.resnet_mamba import (ResNetMambaBackbone, _state_from_bmajor,
                                             _state_to_bmajor)
from event_ssm.models.spikingssm.spiking_temporal import SpikingSSMBlock
from event_ssm.temporal.mamba_temporal import MambaTemporalBlock


def _spk_state_to_bmajor(state, B, hw):
    """(mamba [(conv,ssm)..] with dim0=N, mem (N,C)) -> same with dim0=B for RNNStates storage."""
    mamba_state, mem = state
    return _state_to_bmajor(mamba_state, B, hw), mem.reshape(B, hw, mem.shape[-1])


def _spk_state_from_bmajor(state_b, B, hw):
    """RVT dim0=B spiking state -> dim0=N=B*hw for the scan. None passes through."""
    if state_b is None:
        return None
    mamba_b, mem_b = state_b
    return _state_from_bmajor(mamba_b, B, hw), mem_b.reshape(B * hw, mem_b.shape[-1])


class SpikingSSMBackbone(ResNetMambaBackbone):
    """forward(x:(L,B,20,H,W), prev_states) -> (features dict{1..4}, states list[4]) — the
    parent's RVT recurrent-backbone contract, unchanged."""

    def __init__(self, in_channels: int = 20, d_state: int = 64, num_layers_per_stage: int = 1,
                 temporal_stages=(2, 3, 4), spatial: Optional[nn.Module] = None,
                 spiking_stages=(2, 3, 4), residual: bool = False,
                 lif_kwargs: Optional[dict] = None, pretrained: bool = True):
        spiking_stages = tuple(spiking_stages)
        # validate BEFORE building anything: a spiking stage with no temporal block would be
        # silently ignored, mislabelling an ablation arm
        if not set(spiking_stages) <= set(temporal_stages):
            raise ValueError(f"spiking_stages {spiking_stages} must be a subset of "
                             f"temporal_stages {tuple(temporal_stages)}")
        super().__init__(in_channels=in_channels, pretrained=pretrained, d_state=d_state,
                         num_layers_per_stage=num_layers_per_stage,
                         temporal_stages=temporal_stages, spatial=spatial)
        self.spiking_stages = spiking_stages
        for s in spiking_stages:
            # same Mamba-2 hyper-parameters as the block it replaces (d_conv/expand/headdim are
            # the shared defaults) — the LIF readout is the only difference
            self.temporal[str(s)] = SpikingSSMBlock(
                d_model=self.spatial.stage_dims[s - 1], d_state=d_state,
                num_layers=num_layers_per_stage, residual=residual, **(lif_kwargs or {}))

    def spiking_stats(self):
        """Per spiking stage: last firing rate + learned beta / threshold summary (host sync —
        call at monitor cadence, not every step)."""
        out = {}
        for s in self.spiking_stages:
            lif = self.temporal[str(s)].lif
            beta = lif.beta.detach().float()
            out[s] = dict(rate=lif.last_firing_rate, beta_mean=float(beta.mean()),
                          beta_min=float(beta.min()), beta_max=float(beta.max()),
                          thr_mean=float(lif.threshold.detach().float().mean()))
        return out

    def forward(self, x, prev_states=None, token_mask=None, train_step: bool = True):
        # Mirrors ResNetMambaBackbone.forward; the only difference is the per-stage choice of
        # state helpers (spiking stages carry (mamba_state, mem)).
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
                spiking = stage in self.spiking_stages
                to_n = _spk_state_from_bmajor if spiking else _state_from_bmajor
                to_b = _spk_state_to_bmajor if spiking else _state_to_bmajor
                folded, dims = MambaTemporalBlock.fold(seq)     # (B*h*w, L, c)
                folded, st = self.temporal[str(stage)](folded, to_n(prev_states[i], B, h * w))
                feats[stage] = MambaTemporalBlock.unfold(folded, dims)
                new_states.append(to_b(st, B, h * w))
            else:
                feats[stage] = seq                              # no temporal (not FPN-fed)
                new_states.append(seq.new_zeros(B, 1))          # None-free placeholder, dim0=B
        return feats, new_states
```

- [ ] **Step 4: Lazy export** — in `spikingssm/__init__.py`, add `"SpikingSSMBackbone"` to `__all__`, extend the docstring's second paragraph with "`SpikingSSMBackbone` likewise.", and extend `__getattr__`:

```python
    if name == "SpikingSSMBackbone":
        from event_ssm.models.spikingssm.backbone import SpikingSSMBackbone
        return SpikingSSMBackbone
```

(place before the existing `raise AttributeError`).

- [ ] **Step 5: Run CPU tests to verify pass**

Run: `$PY -m pytest code/event_ssm/tests/models/spikingssm/test_backbone_cpu.py -q`
Expected: 3 passed.

- [ ] **Step 6: Write GPU tests** — `test_backbone_gpu.py`:

```python
"""Stage 18 — SpikingSSMBackbone on the real Mamba-2 kernels. Needs an idle 5070 Ti:

    pytest code/event_ssm/tests/models/spikingssm/ -m gpu

Small spatial depths (1,1,1,1) and a 64x96 frame keep these fast; the contract under test
(state threading, the null-spiking equivalence, RVT state ops) does not depend on size."""
import pytest
import torch

CUDA = torch.cuda.is_available()
pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")]

L, B, H, W = 4, 2, 64, 96


def _spatial():
    from event_ssm.models.puressm import BiMambaSpatialStages
    return BiMambaSpatialStages(in_channels=20, depths=(1, 1, 1, 1), d_state=16,
                                drop_path_rate=0.0)


def _bb(spiking_stages=(2, 3, 4), **lif_kwargs):
    from event_ssm.models.spikingssm.backbone import SpikingSSMBackbone
    torch.manual_seed(0)
    return SpikingSSMBackbone(spatial=_spatial(), spiking_stages=spiking_stages,
                              lif_kwargs=lif_kwargs).cuda().float()


def _x(length=L):
    torch.manual_seed(1)
    return torch.randn(length, B, 20, H, W, device="cuda")


def test_blocks_placed_exactly_on_spiking_stages():
    from event_ssm.models.spikingssm import SpikingSSMBlock
    from event_ssm.temporal.mamba_temporal import MambaTemporalBlock
    bb = _bb(spiking_stages=(4,))
    assert isinstance(bb.temporal["4"], SpikingSSMBlock)
    assert type(bb.temporal["2"]) is MambaTemporalBlock and type(bb.temporal["3"]) is MambaTemporalBlock


def test_forward_backward_and_grad_reaches_spatial_stem():
    bb = _bb().train()
    feats, states = bb(_x(), None)
    assert feats[4].shape == (L, B, 512, H // 32, W // 32)
    loss = sum(feats[s].float().mean() for s in (2, 3, 4))
    loss.backward()
    assert torch.isfinite(loss)
    stem_w = next(bb.spatial.stem.parameters())
    assert stem_w.grad is not None and stem_w.grad.abs().sum() > 0, \
        "surrogate gradient must reach the spatial stem through the spikes"
    st = bb.spiking_stats()
    assert set(st) == {2, 3, 4} and all(0.0 <= v["rate"] <= 1.0 for v in st.values())


@pytest.mark.parametrize("mode", ["analog", "graded", "spike"])
def test_streaming_two_clips_equals_one_clip(mode):
    """State carried across clips (TBPTT / streaming eval) must reproduce the long clip."""
    bb = _bb(output_mode=mode).eval()
    x = _x()
    with torch.no_grad():
        full, _ = bb(x, None)
        a, st = bb(x[:2], None)
        b, _ = bb(x[2:], st)
    for s in (2, 3, 4):
        got = torch.cat([a[s], b[s]], dim=0)
        if mode == "spike":
            # binary output: a 1e-6 membrane difference can flip a spike exactly at threshold
            assert (got != full[s]).float().mean().item() < 1e-3
        else:
            assert (got - full[s]).abs().max().item() < 1e-3


def test_no_spiking_stages_equals_puressm():
    """spiking_stages=() must be numerically the PureSSM backbone — the wiring's null test."""
    from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
    spk = _bb(spiking_stages=()).eval()
    ref = ResNetMambaBackbone(spatial=_spatial()).cuda().float().eval()
    ref.load_state_dict(spk.state_dict())                      # strict: same module tree
    x = _x()
    with torch.no_grad():
        got, _ = spk(x, None)
        want, _ = ref(x, None)
    for s in (1, 2, 3, 4):
        assert torch.allclose(got[s], want[s], atol=1e-6, rtol=0)


def test_rvt_detach_and_reset_handle_spiking_state():
    from modules.utils.detection import RNNStates
    bb = _bb().train()
    _, states = bb(_x(2), None)
    det = RNNStates.recursive_detach(states)
    reset = RNNStates.recursive_reset(det, indices_or_bool_tensor=[0])
    mamba_b, mem_b = reset[3]                                  # stage 4 (index 3)
    assert mem_b.shape[0] == B
    assert torch.count_nonzero(mem_b[0]) == 0, "reset must zero the membrane of sequence 0"
    assert all(torch.count_nonzero(c[0]) == 0 and torch.count_nonzero(s_[0]) == 0
               for c, s_ in mamba_b)
    # resumes from the reset state without error
    bb(_x(2), reset)
```

- [ ] **Step 7: Run GPU tests** (idle GPU only — `nvidia-smi` first)

Run: `$PY -m pytest code/event_ssm/tests/models/spikingssm/test_backbone_gpu.py -m gpu -q`
Expected: 7 passed (3 parametrised streaming cases + 4). Investigate any failure with superpowers:systematic-debugging; never loosen a tolerance without recording why in the Stage-18 notes.

- [ ] **Step 8: Commit**

```bash
git add code/event_ssm/models/spikingssm/backbone.py code/event_ssm/models/spikingssm/__init__.py code/event_ssm/tests/models/spikingssm/test_backbone_cpu.py code/event_ssm/tests/models/spikingssm/test_backbone_gpu.py
git commit -m "feat(spikingssm): SpikingSSMBackbone with spiking-aware state carry"
```

---

### Task 3: Spiking training monitor

**Files:**
- Modify (append only): `code/event_ssm/integration/monitors.py`
- Create: `code/event_ssm/tests/models/spikingssm/test_spiking_monitor.py`

**Interfaces:**
- Consumes: any `nn.Module` with `spiking_stats() -> dict[int, dict[str, float]]` (Task 2).
- Produces: `attach_spiking_monitor(backbone, every_n: int = 200, silence: float = 0.01, saturation: float = 0.90) -> Callable[[], None]` (detach handle). Printed prefix `[spk-monitor]`; warning words `SILENT` / `SATURATED`; wandb keys `monitor/spk_{stat}_s{stage}`.

- [ ] **Step 1: Write failing tests** — `test_spiking_monitor.py`:

```python
"""Stage 18 — spiking monitor on a stub backbone (CPU; the monitor only reads spiking_stats)."""
import pytest
import torch
import torch.nn as nn

from event_ssm.integration.monitors import attach_spiking_monitor


class _Stub(nn.Module):
    def __init__(self, rates):
        super().__init__()
        self.rates = rates

    def forward(self, x):
        return x

    def spiking_stats(self):
        return {s: dict(rate=r, beta_mean=0.9, beta_min=0.8, beta_max=0.95, thr_mean=1.0)
                for s, r in self.rates.items()}


def test_every_n_below_one_raises():
    with pytest.raises(ValueError, match="every_n"):
        attach_spiking_monitor(_Stub({4: 0.2}), every_n=0)


def test_reports_at_cadence(capsys):
    bb = _Stub({3: 0.2, 4: 0.3})
    attach_spiking_monitor(bb, every_n=2)
    bb(torch.zeros(1))
    assert "[spk-monitor]" not in capsys.readouterr().out
    bb(torch.zeros(1))
    out = capsys.readouterr().out
    assert "[spk-monitor]" in out and "s3:" in out and "s4:" in out


def test_flags_silence_and_saturation(capsys):
    bb = _Stub({2: 0.001, 3: 0.2, 4: 0.97})
    attach_spiking_monitor(bb, every_n=1)
    bb(torch.zeros(1))
    out = capsys.readouterr().out
    assert "SILENT stage 2" in out and "SATURATED stage 4" in out
    assert "SILENT stage 3" not in out and "SATURATED stage 3" not in out


def test_detach_handle_stops_reporting(capsys):
    bb = _Stub({4: 0.2})
    remove = attach_spiking_monitor(bb, every_n=1)
    remove()
    bb(torch.zeros(1))
    assert "[spk-monitor]" not in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest code/event_ssm/tests/models/spikingssm/test_spiking_monitor.py -q`
Expected: FAIL — `ImportError: cannot import name 'attach_spiking_monitor'`.

- [ ] **Step 3: Implement** — append to `monitors.py`:

```python
def attach_spiking_monitor(backbone, every_n: int = 200, silence: float = 0.01,
                           saturation: float = 0.90):
    """Stage-18 spiking monitor: per spiking stage firing rate + learned beta/threshold.

    An SNN fails *silently* in two ways — every neuron stops firing (no signal, no surrogate
    gradient) or every neuron fires every step (binary output carries nothing). Neither raises;
    both look like a slowly-flat loss. This hook reads `backbone.spiking_stats()` every
    `every_n` forwards, prints one `[spk-monitor]` line, warns on SILENT (< silence) and
    SATURATED (> saturation), and logs to wandb when a run is active. Attach via env
    SPIKING_MONITOR=1 (see register.py) — default OFF, zero overhead."""
    if every_n < 1:
        # same fail-at-attach rationale as attach_spatial_norm_monitor
        raise ValueError(
            f"attach_spiking_monitor: every_n must be >= 1, got {every_n!r} "
            f"(check SPIKING_MONITOR_EVERY if this was set via the environment).")
    state = {"calls": 0}

    def hook(_module, _inputs, _output):
        try:
            state["calls"] += 1
            if state["calls"] % every_n:
                return
            logs, parts = {}, []
            for stage, st in sorted(backbone.spiking_stats().items()):
                rate = st["rate"]
                if rate < silence:
                    print(f"[spk-monitor] SILENT stage {stage}: firing rate {rate:.4f} < {silence} "
                          f"(call {state['calls']}) — no spikes, no surrogate gradient")
                elif rate > saturation:
                    print(f"[spk-monitor] SATURATED stage {stage}: firing rate {rate:.4f} > "
                          f"{saturation} (call {state['calls']}) — binary output carries ~nothing")
                for k, v in st.items():
                    logs[f"monitor/spk_{k}_s{stage}"] = float(v)
                parts.append(f"s{stage}: rate={rate:.3f} beta={st['beta_mean']:.3f} "
                             f"thr={st['thr_mean']:.3f}")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(logs, commit=False)
            except Exception:
                pass  # wandb optional/offline — the printed line is the fallback record
            print(f"[spk-monitor] call {state['calls']} " + " | ".join(parts))
        except Exception as e:
            print(f"[spk-monitor] hook error suppressed: {e!r}")

    handle = backbone.register_forward_hook(hook)
    return handle.remove
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest code/event_ssm/tests/models/spikingssm/test_spiking_monitor.py code/event_ssm/tests/test_monitors.py -q`
Expected: all pass (new 4 + existing spatial-monitor tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/integration/monitors.py code/event_ssm/tests/models/spikingssm/test_spiking_monitor.py
git commit -m "feat(monitors): spiking firing-rate monitor with silence/saturation alerts"
```

---

### Task 4: Dispatch + Hydra configs + symlinks

**Files:**
- Modify (additive): `code/event_ssm/integration/register.py`
- Create: `code/event_ssm/configs/spikingssm_yolox/default.yaml`
- Create: `code/event_ssm/configs/experiment/gen1/spikingssm.yaml`
- Create symlinks under `external/ssms_event_cameras/RVT/config/`
- Modify (append): `docs/patches/README.md`
- Create: `code/event_ssm/tests/models/spikingssm/test_register_spikingssm.py`

**Interfaces:**
- Consumes: `SpikingSSMBackbone` (Task 2), `attach_spiking_monitor` (Task 3), `BiMambaSpatialStages`, `compose_smoke_config(experiment=..., extra_overrides=[...])` (`event_ssm.integration.smoke_harness`).
- Produces: backbone name `"SpikingSSM"`; config keys `model.backbone.spiking.{output_mode, spiking_stages, beta, threshold, alpha, learn_beta, learn_threshold, reset, detach_reset, residual}`; env `SPIKING_MONITOR`, `SPIKING_MONITOR_EVERY`.

- [ ] **Step 1: Write failing tests** — `test_register_spikingssm.py`:

```python
"""Stage 18 — SpikingSSM dispatch through the RVT monkeypatch + Hydra configs.
Builder tests use the `device` fixture (GPU present) like tests/test_register_puressm.py;
compose/recipe tests are pure config and run anywhere."""
import pathlib

import torch
from omegaconf import OmegaConf

CFG = pathlib.Path(__file__).resolve().parents[3] / "configs"


def _register():
    from event_ssm.integration.smoke_harness import setup_paths, register
    setup_paths()
    register()


def _cfg(**spiking):
    base = dict(name="SpikingSSM", input_channels=20, d_state=64, num_layers_per_stage=1,
                in_stages=[2, 3, 4], depths=[1, 1, 1, 1], spatial_d_state=16,
                drop_path_rate=0.0, checkpoint_blocks=False,
                spiking=dict(output_mode="spike", spiking_stages=[2, 3, 4], beta=0.9,
                             threshold=1.0, alpha=2.0, learn_beta=True, learn_threshold=False,
                             reset="subtract", detach_reset=True, residual=False))
    base["spiking"].update(spiking)
    return OmegaConf.create(base)


def test_builder_dispatch_spikingssm(device):
    _register()
    import models.detection.recurrent_backbone as rb
    from event_ssm.models.puressm import BiMambaSpatialStages
    from event_ssm.models.spikingssm import SpikingSSMBackbone, SpikingSSMBlock
    bb = rb.build_recurrent_backbone(_cfg(spiking_stages=[3, 4]))
    assert isinstance(bb, SpikingSSMBackbone) and isinstance(bb.spatial, BiMambaSpatialStages)
    assert bb.spiking_stages == (3, 4)
    assert not isinstance(bb.temporal["2"], SpikingSSMBlock)
    assert isinstance(bb.temporal["3"], SpikingSSMBlock) and isinstance(bb.temporal["4"], SpikingSSMBlock)


def test_lif_kwargs_flow_from_config(device):
    _register()
    import models.detection.recurrent_backbone as rb
    bb = rb.build_recurrent_backbone(_cfg(output_mode="analog", threshold=0.5,
                                          learn_threshold=True, reset="zero", residual=True))
    blk = bb.temporal["4"]
    assert blk.lif.output_mode == "analog" and blk.lif.reset == "zero" and blk.residual is True
    assert torch.allclose(blk.lif.threshold, torch.full_like(blk.lif.threshold, 0.5))
    assert isinstance(blk.lif.threshold_raw, torch.nn.Parameter)


def test_compose_selects_spikingssm_and_sets_hw():
    from event_ssm.integration.smoke_harness import compose_smoke_config
    cfg = compose_smoke_config(experiment="spikingssm")
    bb = cfg.model.backbone
    assert bb.name == "SpikingSSM"
    assert tuple(bb.in_res_hw) == (256, 320)                    # modifier handled the new name
    assert cfg.model.head.num_classes == 2
    assert set(bb.spiking.keys()) == {"output_mode", "spiking_stages", "beta", "threshold",
                                      "alpha", "learn_beta", "learn_threshold", "reset",
                                      "detach_reset", "residual"}
    assert bb.spiking.output_mode == "spike" and list(bb.spiking.spiking_stages) == [2, 3, 4]


def test_cli_override_selects_ablation_arm():
    from event_ssm.integration.smoke_harness import compose_smoke_config
    cfg = compose_smoke_config(experiment="spikingssm", extra_overrides=[
        "model.backbone.spiking.output_mode=analog", "model.backbone.spiking.spiking_stages=[4]"])
    assert cfg.model.backbone.spiking.output_mode == "analog"
    assert list(cfg.model.backbone.spiking.spiking_stages) == [4]


def test_recipe_identical_to_puressm():
    """Controlled experiment: only the model group may differ."""
    spk = OmegaConf.load(CFG / "experiment/gen1/spikingssm.yaml")
    pure = OmegaConf.load(CFG / "experiment/gen1/puressm.yaml")
    assert list(spk.defaults) == [{"/model/spikingssm_yolox": "default"}]
    for c in (spk, pure):
        del c["defaults"]
    assert spk == pure


def test_model_config_identical_to_puressm_except_spiking():
    spk = OmegaConf.load(CFG / "spikingssm_yolox/default.yaml")
    pure = OmegaConf.load(CFG / "puressm_yolox/default.yaml")
    assert spk.model.backbone.name == "SpikingSSM"
    del spk.model.backbone["name"], spk.model.backbone["spiking"], pure.model.backbone["name"]
    assert spk == pure
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest code/event_ssm/tests/models/spikingssm/test_register_spikingssm.py -q`
Expected: FAIL (builder falls through to RVT's `NotImplementedError`; config files missing).

- [ ] **Step 3: Configs**

Create `code/event_ssm/configs/spikingssm_yolox/default.yaml` by copying `puressm_yolox/default.yaml` verbatim, then change (a) the header comment, (b) `name: PureSSM` → `name: SpikingSSM`, (c) append the `spiking:` block at the end of `model.backbone` (after `checkpoint_blocks`):

```bash
cp code/event_ssm/configs/puressm_yolox/default.yaml code/event_ssm/configs/spikingssm_yolox/default.yaml
```

Header (replace the first 4 comment lines after `# @package _global_`):

```yaml
# Canonical (tracked) Hydra model config for the SpikingSSM backbone (Stage 18, choice C).
# Symlinked into the (gitignored) RVT config tree at
# external/ssms_event_cameras/RVT/config/model/spikingssm_yolox/default.yaml.
# Byte-identical to puressm_yolox/default.yaml except `name` and the `spiking` block (controlled
# experiment — enforced by tests/models/spikingssm/test_register_spikingssm.py).
```

Block to insert after `checkpoint_blocks: False ...` (same 4-space indentation as its siblings):

```yaml
    spiking:                   # LIF readout on the temporal Mamba output (Stage 17/18)
      output_mode: spike       # spike | graded | analog (analog = control arm, should ≈ PureSSM 46.4)
      spiking_stages: [2, 3, 4]  # ⊆ in_stages; de-risking ladder [4] -> [3,4] -> [2,3,4]
      beta: 0.9                # initial leak; learnable per channel if learn_beta
      threshold: 1.0           # sweep = energy-accuracy Pareto (Stage 22)
      alpha: 2.0               # arctan surrogate sharpness
      learn_beta: True
      learn_threshold: False
      reset: subtract          # subtract | zero
      detach_reset: True
      residual: False          # analog bypass — escape hatch, MUST be reported if used
```

Create `code/event_ssm/configs/experiment/gen1/spikingssm.yaml`:

```bash
cp code/event_ssm/configs/experiment/gen1/puressm.yaml code/event_ssm/configs/experiment/gen1/spikingssm.yaml
```

then replace its header comment block with:

```yaml
# @package _global_
# Canonical (tracked) Hydra EXPERIMENT config for the SpikingSSM backbone on Gen1 (Stage 18).
# Byte-identical to experiment/gen1/puressm.yaml except the model group (recipe comparability —
# enforced by tests/models/spikingssm/test_register_spikingssm.py).
# Selected at runtime:  model=rnndet +experiment/gen1=spikingssm
# Ablation arms via CLI, e.g.  model.backbone.spiking.output_mode=analog
# For Hydra to find it, this file is symlinked into the (gitignored) RVT tree at
# external/ssms_event_cameras/RVT/config/experiment/gen1/spikingssm.yaml.
```

and change `- /model/puressm_yolox: default` → `- /model/spikingssm_yolox: default`.

- [ ] **Step 4: Symlinks + patches README**

```bash
mkdir -p external/ssms_event_cameras/RVT/config/model/spikingssm_yolox
ln -s ../../../../../../code/event_ssm/configs/spikingssm_yolox/default.yaml \
      external/ssms_event_cameras/RVT/config/model/spikingssm_yolox/default.yaml
ln -s ../../../../../../code/event_ssm/configs/experiment/gen1/spikingssm.yaml \
      external/ssms_event_cameras/RVT/config/experiment/gen1/spikingssm.yaml
ls -L external/ssms_event_cameras/RVT/config/model/spikingssm_yolox/default.yaml external/ssms_event_cameras/RVT/config/experiment/gen1/spikingssm.yaml
```

Expected: both resolve (no "No such file").

Append to `docs/patches/README.md` immediately after the Stage-12 section (before `## Deferred`):

````markdown
## Stage 18 (2026-10-06): SpikingSSM Hydra config symlinks

After any re-clone of external/, re-create the Stage-18 Hydra config symlinks:
```bash
mkdir -p external/ssms_event_cameras/RVT/config/model/spikingssm_yolox
ln -s ../../../../../../code/event_ssm/configs/spikingssm_yolox/default.yaml \
      external/ssms_event_cameras/RVT/config/model/spikingssm_yolox/default.yaml
ln -s ../../../../../../code/event_ssm/configs/experiment/gen1/spikingssm.yaml \
      external/ssms_event_cameras/RVT/config/experiment/gen1/spikingssm.yaml
```
````

- [ ] **Step 5: Dispatch in `register.py`** (additive)

(a) Update the module docstring's item 1 to read: `... for backbone.name == "ResNetMamba", PureSSM backbone for "PureSSM", or SpikingSSMBackbone for "SpikingSSM".`

(b) Add a module-level constant just below the docstring:

```python
# LIF-readout keys forwarded verbatim from `backbone.spiking` to LIFReadout (Stage 18)
_LIF_KEYS = ("output_mode", "beta", "threshold", "alpha", "learn_beta", "learn_threshold",
             "reset", "detach_reset")
```

(c) In `patched(backbone_cfg)`, insert this branch immediately before the final `return orig(backbone_cfg)`:

```python
        if backbone_cfg.name == "SpikingSSM":
            # Stage 18: PureSSM skeleton with the temporal readout spiking (choice C). Spatial and
            # temporal args are IDENTICAL to the PureSSM branch above (controlled experiment);
            # duplicated rather than refactored so the PureSSM branch stays byte-identical.
            from event_ssm.models.puressm import BiMambaSpatialStages
            from event_ssm.models.spikingssm.backbone import SpikingSSMBackbone
            in_stages = backbone_cfg.get("in_stages", None)
            temporal_stages = tuple(in_stages) if in_stages is not None else (2, 3, 4)
            spatial = BiMambaSpatialStages(
                in_channels=backbone_cfg.input_channels,
                depths=tuple(backbone_cfg.get("depths", (2, 2, 8, 2))),
                d_state=backbone_cfg.get("spatial_d_state", 16),
                drop_path_rate=backbone_cfg.get("drop_path_rate", 0.1),
                checkpoint_blocks=backbone_cfg.get("checkpoint_blocks", False),
            )
            spk = backbone_cfg.get("spiking", None) or {}
            bb = SpikingSSMBackbone(
                in_channels=backbone_cfg.input_channels,
                d_state=backbone_cfg.get("d_state", 64),
                num_layers_per_stage=backbone_cfg.get("num_layers_per_stage", 1),
                temporal_stages=temporal_stages,
                spatial=spatial,
                spiking_stages=tuple(spk.get("spiking_stages", temporal_stages)),
                residual=bool(spk.get("residual", False)),
                lif_kwargs={k: spk[k] for k in _LIF_KEYS if k in spk},
            )
            import os
            if os.environ.get("PURESSM_MONITOR") == "1":
                from event_ssm.integration.monitors import attach_spatial_norm_monitor
                attach_spatial_norm_monitor(bb, every_n=int(os.environ.get("PURESSM_MONITOR_EVERY", "200")))
            if os.environ.get("SPIKING_MONITOR") == "1":
                # Stage 18: firing-rate silence/saturation watch (the two silent SNN failure modes)
                from event_ssm.integration.monitors import attach_spiking_monitor
                attach_spiking_monitor(bb, every_n=int(os.environ.get("SPIKING_MONITOR_EVERY", "200")))
            return bb
```

(d) In `register_config_modifier`, change `("ResNetMamba", "PureSSM")` → `("ResNetMamba", "PureSSM", "SpikingSSM")`.

- [ ] **Step 6: Run tests**

Run: `$PY -m pytest code/event_ssm/tests/models/spikingssm/test_register_spikingssm.py code/event_ssm/tests/test_register.py code/event_ssm/tests/test_register_puressm.py -q`
Expected: all pass (6 new + existing register tests unchanged).

- [ ] **Step 7: Verify the frozen packages are untouched**

Run: `git diff --stat main -- code/event_ssm/models/eventssm code/event_ssm/models/puressm code/event_ssm/temporal code/event_ssm/backbone`
Expected: empty output.

- [ ] **Step 8: Commit**

```bash
git add code/event_ssm/integration/register.py code/event_ssm/configs/spikingssm_yolox code/event_ssm/configs/experiment/gen1/spikingssm.yaml docs/patches/README.md code/event_ssm/tests/models/spikingssm/test_register_spikingssm.py
git commit -m "feat(integration): SpikingSSM selectable via +experiment/gen1=spikingssm"
```

---

### Task 5: Full verification, notes, status

**Files:**
- Create: `docs/notes/Stage18_integration_notes.md`
- Modify: `CLAUDE.md` (Thesis C section, after the STAGE 17 bullet)

- [ ] **Step 1: Full default suite**

Run: `$PY -m pytest code/event_ssm/tests/ -q`
Expected: 0 failures; count = 143 + new non-gpu tests.

- [ ] **Step 2: Full spiking GPU session** (idle GPU only)

Run: `$PY -m pytest code/event_ssm/tests/models/spikingssm/ -m gpu -q`
Expected: all gpu-marked tests pass — the 9 Stage-17 ones (incl. the Task-1-fixed parity test) + 7 from Task 2. Record the exact pass count for the notes.

- [ ] **Step 3: Write `docs/notes/Stage18_integration_notes.md`** containing, with real numbers from Steps 1–2:
  - What was built (backbone subclass, helpers, dispatch, configs, monitor) and the one-line selection command `model=rnndet +experiment/gen1=spikingssm`.
  - Ablation-arm override cheat-sheet: `model.backbone.spiking.output_mode={spike,graded,analog}`, `model.backbone.spiking.spiking_stages=[4]` / `[3,4]` / `[2,3,4]`, `model.backbone.spiking.threshold=…`, `model.backbone.spiking.residual=True` (must be reported), `SPIKING_MONITOR=1 SPIKING_MONITOR_EVERY=200`.
  - **Two Stage-17 fixes (non-obvious):** (1) `beta == _BETA_EPS` passed validation then crashed `math.log(0)` — the pending GPU parity test used exactly that value and would never have run; constructor now rejects it and the test now compares against `lif(ref_ssm(x))`. (2) per-forward `float()` host sync on the firing rate removed (lazy tensor) — would have biased Stage-22 latency.
  - Test counts (CPU + GPU) and the frozen-package `git diff --stat` = empty check.
  - Next: Stage 19 smoke — run `analog` first (integration diagnostic), then `spike`.

- [ ] **Step 4: Update `CLAUDE.md`** — after the `* **STAGE 17 COMPLETE ...` bullet add:

```markdown
* **STAGE 18 COMPLETE (integration) 2026-10-06** — `SpikingSSMBackbone` (`models/spikingssm/backbone.py`, subclasses the unmodified `ResNetMambaBackbone`; spiking-aware `(mamba_state, mem)` state helpers) selectable via `model=rnndet +experiment/gen1=spikingssm`; every ablation arm is a CLI override (`model.backbone.spiking.{output_mode,spiking_stages,threshold,residual,...}`); `SPIKING_MONITOR=1` = silence/saturation watch. Recipe/model-config parity with PureSSM enforced by tests; `spiking_stages=[]` ≡ PureSSM (null test). Fixed two latent Stage-17 bugs (beta=eps crash in a pending GPU test; per-forward host sync on firing rate). Notes: `docs/notes/Stage18_integration_notes.md`. **Next = Stage 19 smoke (analog arm first).**
```

- [ ] **Step 5: Refresh the knowledge graph**

Run: `graphify update .`
Expected: completes without error.

- [ ] **Step 6: Commit**

```bash
git add docs/notes/Stage18_integration_notes.md CLAUDE.md
git commit -m "docs(notes): Stage 18 integration notes + status"
git log -5 --format='%h %s%n%b' | grep -i co-authored || echo "clean"
```

Expected: `clean`.
