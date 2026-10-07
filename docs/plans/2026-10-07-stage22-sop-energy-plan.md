# Stage 22 — SOP and Energy Accounting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per SpikingSSM ablation arm, report the synaptic operations and an operation-count energy estimate per frame,
led by the ceiling (the largest saving any firing pattern could give), inside the existing Stage-10/16/22 benchmark.

**Architecture:** A new pure module `benchmark/sop.py` derives every count from the built detector's own modules: the
spike-fed MACs of the neck, the custom-kernel SSM MACs the profiler cannot see, and the energy model. The only
GPU-dependent part is a hook-based nonzero-rate measurement over 16 test sequences. The benchmark calls both for
`spikingssm` and writes a `sop` block to its JSON; the reporter writes a separate `sop_table.md`.

**Tech Stack:** PyTorch 2.11 (cu128), torch.profiler, h5py + hdf5plugin, pytest; conda env `events_signals`.

**Spec:** `docs/specs/2026-10-07-stage22-sop-energy-design.md` (Revision 2 below supersedes the parts it amends).

## Revision 2 of the spec — outcome of the 2026-10-07 logic check

Every number below was checked on the real code or model, not inferred:

| Check | Result |
|---|---|
| Which ops read each stage's raw output | Traced on the real SpikingSSM detector (CPU, neck + head): exactly `lateral_conv0` (stage 4, channel offset 0), `C3_p4.conv1/conv2` (stage 3, offset 256) and `C3_p3.conv1/conv2` (stage 2, offset 128); nothing else does arithmetic on them. **Total spike-fed MACs = 52,428,800**, the spec's value. |
| Spike readout values | `LIFReadout` spike mode emits `atan_spike(...)` ∈ {0, 1} exactly (not scaled by the threshold); graded = `spk · mem_pre` is nonzero only where a spike fired; analog = the membrane (dense). |
| Profiler units | `aten::conv2d`, `mm`, `addmm`, `bmm` carry FLOPs = 2 × MACs (checked against hand counts); out-of-place `add`/`mul` carry 1 FLOP per element; `conv1d`, in-place ops, `sub`, `div`, `exp`, norms and activations carry none. |
| **Stored FLOP totals** | `flops.total_gflops` (Stage 10/16) = profiler + an analytic Mamba-2 add-on that **includes the in/out projections the profiler already counts** (0.832 of PureSSM's 1.061 analytic GFLOPs are double-counted), while **PureSSM's spatial BiMamba conv + scan kernels (≈ 0.72 GFLOPs) are counted nowhere**. |

Amendments:
1. **Denominator.** Dense MACs = (profiler FLOPs of `MAC_OPS`) / 2 + the MACs of the custom SSM kernels the profiler
   cannot see (temporal Mamba-2 causal conv + scan; spatial BiMamba causal conv + scan, both directions). Projections
   are not added again. Not `½ · total_gflops`.
2. **Operation class.** AC only if the readout is `spike` **and** `residual` is off (with `residual`, the neck reads
   spikes + the dense input). Otherwise every nonzero input costs a MAC: driven MACs = ρ·M for graded, analog and
   residual arms alike (for the dense analog membrane the measured ρ ≈ 1, so this equals the spec's M).
3. **Binary check.** The rate measurement records whether each stage's output was exactly {0, 1}; an AC-class arm with
   non-binary output is refused.
4. **Rates.** Measured over 16 test sequences evenly spaced over the sorted test set, 96 frames each (first 32 = warm-up
   for the recurrent state, last 64 counted), not the single 64-frame benchmark clip, and reported with the per-clip
   min/max and the drift between the two halves of the counted frames.
5. **Neuron updates.** Each LIF neuron costs one MAC per frame (β·mem + x), charged to the spiking arm.
6. **Reference.** The same network with the readout's consumers priced dense: the PureSSM-equivalent arithmetic.
7. **Reporting.** A separate `sop_table.md`, not more columns on the 19-column efficiency table. The training-monitor
   rate cross-check is done by hand in the notes (no console-log parsing).
8. **Out of scope, user decision:** correcting the Stage-10/16 `total_gflops` themselves (citable numbers in Ch. 5).
   This plan leaves `flops` unchanged and documents the finding.

## Global Constraints

- Run tests from the **repo root** with `/home/ghost/miniforge3/envs/events_signals/bin/pytest` (running inside `code/event_ssm` shadows RVT's `models`).
- **A GPU training run is active:** run only the listed CPU test files, never the full suite; `-m gpu` tests are run by the user on an idle GPU.
- Do not edit the frozen packages: `code/event_ssm/{models/eventssm,models/puressm,temporal,backbone}` and `external/`.
- Existing benchmark outputs keep their meaning: no change to the `flops`, `latency`, `energy`, `vram` blocks.
- Energy constants: `E_MAC = 4.6 pJ`, `E_AC = 0.9 pJ` (Horowitz, ISSCC 2014, 45 nm, 32-bit float).
- Input resolution for all per-frame counts: `(256, 320)` (Gen1 240×304 zero-padded), the benchmark's network input.
- Commits: conventional prefix, **no Claude attribution or trailers** (user rule).

## Review Focus

1. Residual arm (`residual=True`) with a spike readout: must be priced as MACs (dense input), never as ACs.
2. A spike-readout checkpoint whose measured output is not exactly binary: refused, not silently priced as ACs.
3. Forward hooks surviving an exception in the rate measurement: they would add host syncs to later timed sections; must be removed in `finally`.
4. Warm-up frames leaking into the counted rate (zero initial state biases it): must be excluded exactly.
5. Profiler fallback absent or failing (no per-op FLOPs): the benchmark records an `error` in the `sop` block and finishes; the reporter shows it.

---

### Task 1: Accounting core (`benchmark/sop.py`)

**Files:**
- Create: `code/event_ssm/benchmark/sop.py`
- Test: `code/event_ssm/tests/test_bench_sop.py`

**Interfaces:**
- Produces:
  - `E_MAC_J: float`, `E_AC_J: float`, `ENERGY_SOURCE: str`, `MAC_OPS: tuple[str, ...]`
  - `spike_fed_macs(detector, stages, in_hw=(256, 320)) -> dict[int, int]`
  - `lif_neurons(detector, stages, in_hw=(256, 320)) -> int`
  - `ssm_kernel_macs(detector, in_hw=(256, 320)) -> {"temporal": int, "spatial": int}`
  - `dense_macs(by_op_flops: dict[str, int], kernel_macs: dict[str, int]) -> {"profiled_macs", "kernel_macs", "total_macs", "excluded_flops_by_op"}`
  - `op_class(arm: dict) -> "AC" | "MAC"`
  - `op_energy(total_macs, fed_macs: dict, rates: dict, op: str, neurons: int) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
"""Stage 22 — SOP / energy accounting (CPU). Counts are derived from the built detector; the hand literals are the
spec's first-principles values (docs/specs/2026-10-07-stage22-sop-energy-design.md, Revision 2 in the plan)."""
import collections

import pytest
import torch

from event_ssm.benchmark import sop
from event_ssm.benchmark.bench_models import build_model


@pytest.fixture(scope="module")
def det():
    return build_model("spikingssm", device=torch.device("cpu"), load_ckpt=False).detector.eval()


def test_spike_fed_macs_match_the_hand_count(det):
    assert sop.spike_fed_macs(det, [2, 3, 4]) == {2: 20_971_520, 3: 20_971_520, 4: 10_485_760}
    assert sop.spike_fed_macs(det, [4]) == {4: 10_485_760}


def test_spike_fed_macs_match_the_traced_consumers(det):
    # empirical: taint each stage tensor through data movement, record every other op that reads it
    from torch.overrides import TorchFunctionMode
    stages = tuple(det.fpn.in_features)
    dims, strides = det.backbone.get_stage_dims(stages), det.backbone.get_strides(stages)
    feats = {s: torch.rand(1, c, 256 // st, 320 // st) for s, c, st in zip(stages, dims, strides)}
    taint, macs = {id(t): s for s, t in feats.items()}, collections.Counter()
    move = {"cat", "view", "reshape", "contiguous", "__getitem__", "__get__", "permute", "flatten", "to"}

    class Tracer(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            name = getattr(func, "__name__", str(func))
            flat = list(args[0]) if name == "cat" else list(args)
            hit = [taint[id(a)] for a in flat if isinstance(a, torch.Tensor) and id(a) in taint]
            if hit and name in move:
                if isinstance(out, torch.Tensor):
                    taint[id(out)] = hit[0]
            elif hit:
                assert name == "conv2d", f"unexpected op {name} reads a raw stage output"
                w = args[1]
                s = hit[0]
                i = stages.index(s)
                macs[s] += (256 // strides[i]) * (320 // strides[i]) * dims[i] * w.shape[0] * w.shape[2] * w.shape[3]
            return out

    with torch.no_grad(), Tracer():
        det.forward_detect(feats)
    assert dict(macs) == sop.spike_fed_macs(det, list(stages))


def test_spike_fed_macs_refuse_a_stage_outside_the_neck(det):
    with pytest.raises(ValueError, match="does not feed the neck"):
        sop.spike_fed_macs(det, [1])


def test_lif_neurons(det):
    assert sop.lif_neurons(det, [2, 3, 4]) == 128 * 1280 + 256 * 320 + 512 * 80
    assert sop.lif_neurons(det, [4]) == 512 * 80


def test_ssm_kernel_macs_match_the_hand_count(det):
    # temporal Mamba-2 (d_state 64): conv_dim*d_conv + 3*d_inner*d_state per token
    temporal = 1280 * (384 * 4 + 3 * 256 * 64) + 320 * (640 * 4 + 3 * 512 * 64) + 80 * (1152 * 4 + 3 * 1024 * 64)
    # spatial BiMamba (d_state 16, depths 2/2/8/2), two directions per block
    spatial = (5120 * 2 * 2 * (160 * 4 + 3 * 128 * 16) + 1280 * 2 * 2 * (288 * 4 + 3 * 256 * 16)
               + 320 * 8 * 2 * (544 * 4 + 3 * 512 * 16) + 80 * 2 * 2 * (1056 * 4 + 3 * 1024 * 16))
    assert sop.ssm_kernel_macs(det) == {"temporal": temporal, "spatial": spatial}
    assert (temporal, spatial) == (113_254_400, 361_799_680)


def test_profiler_mac_ops_are_twice_the_macs():
    # the MAC_OPS convention, checked on the profiler itself (CPU)
    from torch.profiler import ProfilerActivity, profile
    conv, lin = torch.nn.Conv2d(8, 4, 1, bias=False), torch.nn.Linear(16, 8)
    x, y, a, b = torch.randn(1, 8, 5, 6), torch.randn(3, 16), torch.randn(2, 3, 4), torch.randn(2, 4, 5)
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        z = conv(x); lin(y); torch.bmm(a, b); z + z; z * z
    by_op = collections.Counter()
    for e in prof.events():
        if getattr(e, "flops", None):
            by_op[e.name] += e.flops
    d = sop.dense_macs(dict(by_op), {"temporal": 0, "spatial": 0})
    assert d["total_macs"] == 30 * 8 * 4 + 3 * 16 * 8 + 2 * 3 * 4 * 5
    assert set(d["excluded_flops_by_op"]) == {"aten::add", "aten::mul"}


def test_dense_macs_adds_the_kernel_macs():
    d = sop.dense_macs({"aten::conv2d": 1000, "aten::mm": 200, "aten::add": 50}, {"temporal": 7, "spatial": 3})
    assert d == {"profiled_macs": 600, "kernel_macs": {"temporal": 7, "spatial": 3}, "total_macs": 610,
                 "excluded_flops_by_op": {"aten::add": 50}}


@pytest.mark.parametrize("mode, residual, expected", [
    ("spike", False, "AC"), ("spike", True, "MAC"), ("graded", False, "MAC"), ("analog", False, "MAC")])
def test_op_class(mode, residual, expected):
    assert sop.op_class({"output_mode": mode, "residual": residual}) == expected


def test_op_energy_spike_arm():
    e = sop.op_energy(1_000_000, {4: 10_000}, {4: 0.25}, "AC", neurons=100)
    assert e["driven_ops"] == 2_500
    assert e["macs"] == 1_000_000 - 10_000 + 100 and e["acs"] == 2_500
    assert e["energy_j"] == pytest.approx(sop.E_MAC_J * 990_100 + sop.E_AC_J * 2_500)
    assert e["energy_dense_j"] == pytest.approx(sop.E_MAC_J * 1_000_000)
    assert e["saving"] == pytest.approx(1 - e["energy_j"] / e["energy_dense_j"])
    assert e["ceiling"] == pytest.approx((10_000 - 100) / 1_000_000)
    assert e["spike_fed_share"] == pytest.approx(0.01)


def test_op_energy_graded_arm_counts_sparse_macs():
    e = sop.op_energy(1_000_000, {4: 10_000}, {4: 0.25}, "MAC", neurons=100)
    assert e["acs"] == 0 and e["macs"] == 1_000_000 - 10_000 + 100 + 2_500
    assert e["saving"] == pytest.approx((10_000 - 100 - 2_500) / 1_000_000)


def test_op_energy_zero_activity_reaches_the_ceiling():
    e = sop.op_energy(1_000_000, {3: 6_000, 4: 4_000}, {3: 0.0, 4: 0.0}, "AC", neurons=0)
    assert e["saving"] == pytest.approx(e["ceiling"]) and e["ceiling"] == pytest.approx(0.01)


@pytest.mark.parametrize("fed, rates, op, match", [
    ({4: 10}, {4: 1.5}, "AC", "rate"),
    ({4: 10}, {4: -0.1}, "AC", "rate"),
    ({4: 10}, {3: 0.1}, "AC", "stages"),
    ({4: 10}, {4: 0.1}, "SOP", "op"),
    ({4: 2_000_000}, {4: 0.1}, "AC", "spike-fed"),
])
def test_op_energy_refuses_bad_inputs(fed, rates, op, match):
    with pytest.raises(ValueError, match=match):
        sop.op_energy(1_000_000, fed, rates, op, neurons=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ghost/miniforge3/envs/events_signals/bin/pytest code/event_ssm/tests/test_bench_sop.py -q`
Expected: collection error, `cannot import name 'sop'`.

- [ ] **Step 3: Write the implementation**

```python
"""Stage 22 — synaptic-operation (SOP) and energy accounting for SpikingSSM.

Spec: docs/specs/2026-10-07-stage22-sop-energy-design.md, amended by Revision 2 in
docs/plans/2026-10-07-stage22-sop-energy-plan.md. Only the temporal readout spikes (choice C); its output feeds five
1x1 convolutions at the entry of the YOLO-PAFPN neck and nothing else (traced in tests/test_bench_sop.py). This
module counts the MACs those convolutions spend on the spiking channels, the dense MACs of the whole network, and
prices both with the 45 nm energy-per-operation constants of the SNN detection literature. Every result is an
OPERATION-COUNT estimate: memory traffic, which dominates energy on real hardware, is not modelled, and pricing
sparse MACs below dense ones assumes hardware that skips zero inputs.
"""
E_MAC_J = 4.6e-12       # 32-bit float multiply (3.7 pJ) + add (0.9 pJ), 45 nm
E_AC_J = 0.9e-12        # 32-bit float add, 45 nm
ENERGY_SOURCE = ("Horowitz, ISSCC 2014, 45 nm, 32-bit float: MAC 4.6 pJ (3.7 mult + 0.9 add), AC 0.9 pJ; "
                 "operation-count estimate, memory traffic not modelled")
# torch.profiler gives these ops FLOPs = 2 x MACs (tests/test_bench_sop.py). Its other FLOP-carrying ops
# (out-of-place add/mul, 1 FLOP per element) are elementwise and are left out of the MAC count.
MAC_OPS = ("aten::conv2d", "aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm")
RATE_CLIPS, RATE_WARMUP, RATE_FRAMES = 16, 32, 64     # Revision 2 §4


def _stage_geometry(detector, in_hw):
    fpn, bb = detector.fpn, detector.backbone
    in_stages = tuple(fpn.in_features)
    dims, strides = bb.get_stage_dims(in_stages), bb.get_strides(in_stages)
    h, w = in_hw
    return in_stages, dims, [(h // st) * (w // st) for st in strides]


def spike_fed_macs(detector, stages, in_hw=(256, 320)) -> dict:
    """{stage: MACs per frame that the neck spends on that stage's output channels}. The YOLO-PAFPN wiring
    (RVT models/detection/yolox_extension/models/yolo_pafpn.py, forward):
        in_stages[2] -> lateral_conv0                 input = x0
        in_stages[1] -> C3_p4.conv1, C3_p4.conv2      input = cat[up(lateral_conv0(x0)), x1]
        in_stages[0] -> C3_p3.conv1, C3_p3.conv2      input = cat[up(reduce_conv1(.)), x2]
    Channel counts, kernel, stride and groups are checked against the modules, so a changed neck fails closed."""
    fpn = detector.fpn
    in_stages, dims, hws = _stage_geometry(detector, in_hw)
    consumers = {2: ((fpn.lateral_conv0.conv,), 0),
                 1: ((fpn.C3_p4.conv1.conv, fpn.C3_p4.conv2.conv), fpn.lateral_conv0.conv.out_channels),
                 0: ((fpn.C3_p3.conv1.conv, fpn.C3_p3.conv2.conv), fpn.reduce_conv1.conv.out_channels)}
    out = {}
    for s in stages:
        if s not in in_stages:
            raise ValueError(f"stage {s} does not feed the neck (in_stages {in_stages})")
        i = in_stages.index(s)
        convs, offset = consumers[i]
        macs = 0
        for conv in convs:
            if (conv.in_channels != offset + dims[i] or conv.kernel_size != (1, 1) or conv.stride != (1, 1)
                    or conv.groups != 1):
                raise ValueError(f"neck wiring changed: stage {s} consumer {conv} (expected 1x1/stride-1/ungrouped "
                                 f"with {offset} + {dims[i]} input channels)")
            macs += hws[i] * dims[i] * conv.out_channels
        out[s] = macs
    return out


def lif_neurons(detector, stages, in_hw=(256, 320)) -> int:
    """LIF neurons per frame: one per channel and position of each spiking stage's output."""
    in_stages, dims, hws = _stage_geometry(detector, in_hw)
    return sum(dims[in_stages.index(s)] * hws[in_stages.index(s)] for s in stages)


def _kernel_macs_per_token(d_ssm, d_state, d_conv, conv_dim) -> int:
    """One scan direction, the part that runs in custom kernels: depthwise causal conv over the xBC channels,
    and the selective scan (state decay, input injection, readout: 3 MACs per state element)."""
    return conv_dim * d_conv + 3 * d_ssm * d_state


def ssm_kernel_macs(detector, in_hw=(256, 320)) -> dict:
    """MACs per frame of the custom SSM kernels, which torch.profiler counts as zero: temporal Mamba-2 layers
    (one direction) and spatial BiMamba scans (two directions). The in/out projections are nn.Linear layers that the
    profiler already counts, so they are NOT added here (Revision 2 §1)."""
    bb = detector.backbone
    h, w = in_hw
    temporal = 0
    for stage_str, block in bb.temporal.items():
        stride = bb.get_strides((int(stage_str),))[0]
        tokens = (h // stride) * (w // stride)
        for layer in getattr(block, "ssm", block).layers:
            conv_dim = layer.d_ssm + 2 * layer.ngroups * layer.d_state
            temporal += tokens * _kernel_macs_per_token(layer.d_ssm, layer.d_state, layer.d_conv, conv_dim)
    spatial = 0
    from event_ssm.models.puressm._scan2d import BiMamba1DScan
    sp = bb.spatial
    for i, stage in enumerate(getattr(sp, "stages", ())):
        tokens = (h // sp.strides[i]) * (w // sp.strides[i])
        for m in stage.modules():
            if isinstance(m, BiMamba1DScan):
                spatial += tokens * 2 * _kernel_macs_per_token(m.d_inner, m.d_state, m.d_conv, m.conv_dim)
    return {"temporal": temporal, "spatial": spatial}


def dense_macs(by_op_flops: dict, kernel_macs: dict) -> dict:
    """Dense MACs per frame: profiler-counted MAC ops (FLOPs / 2) plus the unprofiled SSM kernels."""
    profiled = sum(int(by_op_flops.get(op, 0)) for op in MAC_OPS) // 2
    return {"profiled_macs": profiled, "kernel_macs": dict(kernel_macs),
            "total_macs": profiled + sum(kernel_macs.values()),
            "excluded_flops_by_op": {op: f for op, f in by_op_flops.items() if op not in MAC_OPS}}


def op_class(arm: dict) -> str:
    """'AC' when the neck reads pure binary spikes (spike readout, no residual): a spike only adds weights.
    Otherwise the neck reads real values (graded spikes, the analog membrane, or spikes plus the residual input)
    and every nonzero input costs a multiply-accumulate: 'MAC'."""
    return "AC" if arm["output_mode"] == "spike" and not arm["residual"] else "MAC"


def op_energy(total_macs, fed_macs: dict, rates: dict, op: str, neurons: int) -> dict:
    """Energy per frame of the arm vs the same network priced dense (the PureSSM-equivalent arithmetic). The
    readout's consumers are charged per nonzero input: rate x MACs as accumulates (op 'AC') or as MACs ('MAC');
    each LIF neuron costs one MAC per frame (beta * mem + x). ceiling = the saving at zero activity."""
    if op not in ("AC", "MAC"):
        raise ValueError(f"op must be 'AC' or 'MAC', got {op!r}")
    if set(rates) != set(fed_macs):
        raise ValueError(f"rate stages {sorted(rates)} != spiking stages {sorted(fed_macs)}")
    for s, r in rates.items():
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"rate of stage {s} outside [0, 1]: {r}")
    fed = sum(fed_macs.values())
    if not 0 < fed < total_macs:
        raise ValueError(f"spike-fed MACs {fed} must be positive and below the dense total {total_macs}")
    driven = sum(rates[s] * fed_macs[s] for s in fed_macs)
    macs = total_macs - fed + neurons + (driven if op == "MAC" else 0)
    acs = driven if op == "AC" else 0
    e, e_dense = E_MAC_J * macs + E_AC_J * acs, E_MAC_J * total_macs
    return {"op_class": op, "spike_fed_total": fed, "driven_ops": driven, "neurons": neurons,
            "macs": macs, "acs": acs, "energy_j": e, "energy_dense_j": e_dense,
            "saving": 1.0 - e / e_dense, "ceiling": (fed - neurons) / total_macs,
            "spike_fed_share": fed / total_macs}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ghost/miniforge3/envs/events_signals/bin/pytest code/event_ssm/tests/test_bench_sop.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/benchmark/sop.py code/event_ssm/tests/test_bench_sop.py
git commit -m "feat(stage22): SOP accounting core (spike-fed MACs, kernel MACs, energy model)"
```

---

### Task 2: Rate measurement (`iter_rate_clips` + `measure_nonzero_rates`)

**Files:**
- Modify: `code/event_ssm/benchmark/bench_clip.py` (append `iter_rate_clips`)
- Modify: `code/event_ssm/benchmark/sop.py` (append `measure_nonzero_rates`, `sop_block`)
- Test: `code/event_ssm/tests/test_bench_sop.py` (append)

**Interfaces:**
- Consumes: Task 1's `spike_fed_macs`, `lif_neurons`, `ssm_kernel_macs`, `dense_macs`, `op_class`, `op_energy`.
- Produces:
  - `iter_rate_clips(k=16, n_frames=96, root=None, skip=100)` yields `(sequence_name: str, clip: (n_frames, 20, 256, 320) float32)`
  - `measure_nonzero_rates(model, clips, warmup: int) -> {"stages": {int: {"rate", "rate_min_clip", "rate_max_clip", "drift", "binary"}}, "clips": [str], "warmup": int}`; `model` exposes `.detector` (eval mode), `.device`, `.network_step(frame (1,C,H,W), state) -> (out, state)`
  - `sop_block(detector, arm: dict, by_op_flops: dict, rates: dict, in_hw=(256, 320)) -> dict` (JSON-safe, string stage keys)

- [ ] **Step 1: Write the failing tests** (append to `test_bench_sop.py`)

```python
# ---- rate measurement ----------------------------------------------------------------------------------------
import types

import numpy as np


def _fake_h5_root(tmp_path, lengths):
    import h5py, hdf5plugin  # noqa: F401
    from event_ssm.benchmark.bench_clip import EVR
    for i, n in enumerate(lengths):
        f = tmp_path / f"seq{i:02d}" / EVR
        f.parent.mkdir(parents=True)
        with h5py.File(f, "w") as h:
            h.create_dataset("data", data=np.full((n, 20, 24, 30), i, dtype=np.uint8))   # small frames: fast
    return tmp_path


def test_rate_clips_are_evenly_spaced_and_skip_short_sequences(tmp_path):
    from event_ssm.benchmark.bench_clip import iter_rate_clips
    root = _fake_h5_root(tmp_path, [20, 3, 20, 20, 20, 20, 20, 20])      # seq01 is too short
    got = list(iter_rate_clips(k=4, n_frames=6, root=root, skip=2))
    assert [n for n, _ in got] == ["seq00", "seq02", "seq04", "seq06"]
    clip = got[1][1]
    assert clip.shape == (6, 20, 256, 320) and clip.dtype == torch.float32
    assert float(clip[0, 0, 0, 0]) == 2.0 and float(clip[0, 0, 250, 0]) == 0.0      # zero-padded margin


def test_rate_clips_shift_past_a_short_pick(tmp_path):
    from event_ssm.benchmark.bench_clip import iter_rate_clips
    root = _fake_h5_root(tmp_path, [20, 20, 3, 20])
    assert [n for n, _ in iter_rate_clips(k=2, n_frames=6, root=root, skip=2)] == ["seq00", "seq03"]


def test_rate_clips_refuse_when_too_few_long_sequences(tmp_path):
    from event_ssm.benchmark.bench_clip import iter_rate_clips
    root = _fake_h5_root(tmp_path, [20, 3, 3])
    with pytest.raises(ValueError, match="long enough"):
        list(iter_rate_clips(k=2, n_frames=6, root=root, skip=2))


class _Readout(torch.nn.Module):
    """Stands in for a SpikingSSMBlock: returns (out, state); out depends only on the frame index."""
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def forward(self, t):
        return self.pattern(t), None


class _FakeModel:
    def __init__(self, patterns, fail_at=None):
        self.device = torch.device("cpu")
        self.fail_at = fail_at
        bb = torch.nn.Module()
        bb.spiking_stages = tuple(patterns)
        bb.temporal = torch.nn.ModuleDict({str(s): _Readout(p) for s, p in patterns.items()})
        self.detector = torch.nn.Module()
        self.detector.backbone = bb
        self.detector.eval()

    def network_step(self, frame, state):
        t = 0 if state is None else state + 1
        if self.fail_at is not None and t == self.fail_at:
            raise RuntimeError("boom")
        for s in self.detector.backbone.spiking_stages:
            self.detector.backbone.temporal[str(s)](t)
        return None, t


def _clips(n, k=2):
    return ((f"c{i}", torch.zeros(n, 1, 2, 2)) for i in range(k))


def test_rates_exclude_the_warmup_and_split_halves():
    # stage 4: frame t emits a (10,) vector with t % 10 ones -> rate t/10 per frame (binary)
    pat = lambda t: (torch.arange(10) < (t % 10)).float()
    r = sop.measure_nonzero_rates(_FakeModel({4: pat}), _clips(8), warmup=4)
    st = r["stages"][4]
    assert st["rate"] == pytest.approx((4 + 5 + 6 + 7) / 40)          # frames 4..7 only
    assert st["drift"] == pytest.approx((6 + 7) / 20 - (4 + 5) / 20)
    assert st["binary"] is True and st["rate_min_clip"] == st["rate_max_clip"] == st["rate"]
    assert r["clips"] == ["c0", "c1"] and r["warmup"] == 4


def test_rates_flag_non_binary_output():
    r = sop.measure_nonzero_rates(_FakeModel({3: lambda t: torch.full((4,), 0.5)}), _clips(6), warmup=2)
    assert r["stages"][3]["binary"] is False and r["stages"][3]["rate"] == 1.0


def test_rates_remove_hooks_even_when_a_forward_fails():
    m = _FakeModel({4: lambda t: torch.ones(3)}, fail_at=3)
    with pytest.raises(RuntimeError, match="boom"):
        sop.measure_nonzero_rates(m, _clips(6), warmup=1)
    assert not m.detector.backbone.temporal["4"]._forward_hooks


def test_rates_refuse_training_mode_and_too_short_clips():
    m = _FakeModel({4: lambda t: torch.ones(3)})
    with pytest.raises(ValueError, match="warm-up"):
        sop.measure_nonzero_rates(m, _clips(3), warmup=2)
    m.detector.train()
    with pytest.raises(ValueError, match="eval"):
        sop.measure_nonzero_rates(m, _clips(6), warmup=2)


def _rates(stages, rate=0.2, binary=True):
    return {"stages": {s: {"rate": rate, "rate_min_clip": rate, "rate_max_clip": rate, "drift": 0.0,
                           "binary": binary} for s in stages}, "clips": ["c0"], "warmup": 32}


def test_sop_block_spike_arm(det):
    arm = {"output_mode": "spike", "residual": False, "spiking_stages": [4]}
    b = sop.sop_block(det, arm, {"aten::conv2d": 8_000_000_000}, _rates([4]))
    assert b["op_class"] == "AC" and b["spike_fed_macs"] == {"4": 10_485_760}
    assert b["dense"]["total_macs"] == 4_000_000_000 + 113_254_400 + 361_799_680
    assert b["driven_ops"] == pytest.approx(0.2 * 10_485_760)
    assert b["constants"]["e_mac_j"] == sop.E_MAC_J and "Horowitz" in b["constants"]["source"]
    assert set(b["rates"]["stages"]) == {"4"}
    import json
    json.dumps(b)                                                         # JSON-safe


def test_sop_block_refuses_a_non_binary_spike_readout(det):
    arm = {"output_mode": "spike", "residual": False, "spiking_stages": [4]}
    with pytest.raises(ValueError, match="binary"):
        sop.sop_block(det, arm, {"aten::conv2d": 8_000_000_000}, _rates([4], binary=False))


def test_sop_block_residual_spike_arm_is_priced_as_macs(det):
    arm = {"output_mode": "spike", "residual": True, "spiking_stages": [3, 4]}
    b = sop.sop_block(det, arm, {"aten::conv2d": 8_000_000_000}, _rates([3, 4], rate=1.0, binary=False))
    assert b["op_class"] == "MAC" and b["acs"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ghost/miniforge3/envs/events_signals/bin/pytest code/event_ssm/tests/test_bench_sop.py -q`
Expected: the new tests fail (`iter_rate_clips` / `measure_nonzero_rates` / `sop_block` missing).

- [ ] **Step 3: Write the implementation**

Append to `bench_clip.py`:

```python
def _n_frames(seq: pathlib.Path) -> int:
    import h5py, hdf5plugin  # noqa: F401
    with h5py.File(seq / EVR, "r") as f:
        return f[KEY if KEY in f else list(f.keys())[0]].shape[0]


def iter_rate_clips(k: int = 16, n_frames: int = 96, root=None, skip: int = 100):
    """Stage 22 firing-rate data: k clips of n_frames consecutive dt=50 test frames from sequences evenly spaced
    over the sorted test set (deterministic), zero-padded like the benchmark clip. A pick shorter than
    skip + n_frames moves to the next long-enough sequence. Yields (sequence name, clip) one at a time: sixteen
    float clips at once would need ~10 GB. Selection happens before the first yield, so a shortage raises early."""
    import h5py, hdf5plugin  # noqa: F401
    root = pathlib.Path(root) if root else DEFAULT_ROOT
    seqs = sorted(p for p in root.iterdir() if p.is_dir())
    picks, used = [], set()
    for j in range(k):
        i = (j * len(seqs)) // k
        while i < len(seqs) and (i in used or _n_frames(seqs[i]) < skip + n_frames):
            i += 1
        if i == len(seqs):
            raise ValueError(f"not enough test sequences long enough for {k} clips of {skip}+{n_frames} frames")
        used.add(i)
        picks.append(seqs[i])

    def _gen():
        for seq in picks:
            with h5py.File(seq / EVR, "r") as f:
                arr = f[KEY if KEY in f else list(f.keys())[0]][skip:skip + n_frames]
            t = torch.from_numpy(arr).float()
            yield seq.name, pad_to_network_shape(torch.zeros(n_frames, t.shape[1], PAD_H, PAD_W), t)
    return _gen()
```

Append to `sop.py`:

```python
def measure_nonzero_rates(model, clips, warmup: int) -> dict:
    """Nonzero rate of each spiking stage's readout output (exactly the tensor the neck reads), streaming each
    clip from a zero state with the state carried frame to frame. The first `warmup` frames of a clip settle the
    recurrent state and are not counted. Per stage: the rate over all counted frames, the per-clip min/max, the
    drift (second half minus first half of the counted frames; a check that the warm-up was long enough) and
    whether every output value was exactly 0 or 1. Hooks are removed in `finally`, so they can never leak into a
    later timed section. Needs the CUDA kernels for a real model."""
    if model.detector.training:
        raise ValueError("measure rates in eval mode")
    bb = model.detector.backbone
    stages = list(bb.spiking_stages)
    rec = {"on": False, "half": 0}
    cur = {s: [[0, 0], [0, 0]] for s in stages}                 # per clip: [half][nonzero, total]
    tot = {s: [[0, 0], [0, 0]] for s in stages}
    per_clip = {s: [] for s in stages}
    binary = {s: True for s in stages}

    def make_hook(s):
        def hook(_mod, _inp, out):
            if not rec["on"]:
                return
            y = out[0]
            c = cur[s][rec["half"]]
            c[0] += int((y != 0).sum())
            c[1] += y.numel()
            if binary[s] and not bool(((y == 0) | (y == 1)).all()):
                binary[s] = False
        return hook

    handles = [bb.temporal[str(s)].register_forward_hook(make_hook(s)) for s in stages]
    names = []
    try:
        for name, clip in clips:
            n = clip.shape[0] - warmup
            if n < 2:
                raise ValueError(f"clip {name}: {clip.shape[0]} frames leave fewer than 2 after a {warmup}-frame "
                                 f"warm-up")
            for s in stages:
                cur[s] = [[0, 0], [0, 0]]
            state = None
            for t in range(clip.shape[0]):
                rec["on"] = t >= warmup
                rec["half"] = 0 if t - warmup < n // 2 else 1
                _, state = model.network_step(clip[t:t + 1].to(model.device), state)
            for s in stages:
                (a0, n0), (a1, n1) = cur[s]
                per_clip[s].append((a0 + a1) / (n0 + n1))
                for h in (0, 1):
                    tot[s][h][0] += cur[s][h][0]
                    tot[s][h][1] += cur[s][h][1]
            names.append(name)
    finally:
        for h in handles:
            h.remove()
    if not names:
        raise ValueError("no clips to measure")
    stats = {}
    for s in stages:
        (a0, n0), (a1, n1) = tot[s]
        stats[s] = {"rate": (a0 + a1) / (n0 + n1), "rate_min_clip": min(per_clip[s]),
                    "rate_max_clip": max(per_clip[s]), "drift": a1 / n1 - a0 / n0, "binary": binary[s]}
    return {"stages": stats, "clips": names, "warmup": warmup}


def sop_block(detector, arm: dict, by_op_flops: dict, rates: dict, in_hw=(256, 320)) -> dict:
    """The benchmark JSON's `sop` block for one SpikingSSM arm (string stage keys, JSON-safe)."""
    stages = list(arm["spiking_stages"])
    op = op_class(arm)
    if op == "AC":
        bad = [s for s in stages if not rates["stages"][s]["binary"]]
        if bad:
            raise ValueError(f"spike readout but non-binary output measured at stages {bad}: refusing to price "
                             f"the neck's inputs as accumulates")
    fed = spike_fed_macs(detector, stages, in_hw)
    dense = dense_macs(by_op_flops, ssm_kernel_macs(detector, in_hw))
    energy = op_energy(dense["total_macs"], fed, {s: rates["stages"][s]["rate"] for s in stages}, op,
                       lif_neurons(detector, stages, in_hw))
    return {"constants": {"e_mac_j": E_MAC_J, "e_ac_j": E_AC_J, "source": ENERGY_SOURCE},
            "dense": dense, "spike_fed_macs": {str(s): m for s, m in fed.items()},
            "rates": {**rates, "stages": {str(s): v for s, v in rates["stages"].items()}}, **energy}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ghost/miniforge3/envs/events_signals/bin/pytest code/event_ssm/tests/test_bench_sop.py code/event_ssm/tests/test_bench_clip.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/benchmark/sop.py code/event_ssm/benchmark/bench_clip.py code/event_ssm/tests/test_bench_sop.py
git commit -m "feat(stage22): nonzero-rate measurement over 16 test sequences; sop block"
```

---

### Task 3: Benchmark and report integration

**Files:**
- Modify: `code/event_ssm/benchmark/bench_metrics.py` (`profiler_network_flops` returns `by_op`)
- Modify: `code/event_ssm/scripts/stage10_benchmark.py` (`sop_section`; call it last in `measure_model` for spikingssm)
- Modify: `code/event_ssm/scripts/stage10_report.py` (`_sop_table_md`; written by `generate`)
- Test: `code/event_ssm/tests/test_bench_metrics.py`, `test_bench_schema.py`, `test_bench_report.py`, `test_bench_sop.py` (gpu test)

**Interfaces:**
- Consumes: `sop.measure_nonzero_rates`, `sop.sop_block`, `sop.RATE_*`, `bench_clip.iter_rate_clips`.
- Produces: `profiler_network_flops(...) -> {"counted_gflops", "source", "by_op": {op: flops}}`; `stage10_benchmark.sop_section(model, arm, by_op, smoke) -> dict`; JSON `models.spikingssm.sop`; `<out>/sop_table.md`.

- [ ] **Step 1: Write the failing tests**

`test_bench_metrics.py` (append):
```python
def test_profiler_flops_carry_a_per_op_breakdown():
    conv = torch.nn.Conv2d(3, 4, 3, bias=False)
    x = torch.randn(1, 3, 8, 8)
    r = bm.profiler_network_flops(lambda: conv(x), device=None)
    assert r["by_op"]["aten::conv2d"] == 2 * 6 * 6 * 4 * 3 * 3 * 3
    assert sum(r["by_op"].values()) / 1e9 == pytest.approx(r["counted_gflops"])
```

`test_bench_schema.py` (append):
```python
def test_sop_section_records_an_error_without_per_op_flops():
    out = s10.sop_section(model=None, arm={}, by_op=None, smoke=True)
    assert "error" in out and "per-op" in out["error"]


def test_sop_section_records_an_error_instead_of_aborting(monkeypatch):
    from event_ssm.benchmark import bench_clip, sop as sop_mod
    def boom(*a, **k):
        raise ValueError("non-binary")
    monkeypatch.setattr(bench_clip, "iter_rate_clips", lambda **k: iter(()))   # no dependence on the Gen1 data
    monkeypatch.setattr(sop_mod, "measure_nonzero_rates", boom)
    out = s10.sop_section(model=None, arm={"spiking_stages": [4]}, by_op={"aten::conv2d": 2}, smoke=True)
    assert out == {"error": "non-binary"}
```

`test_bench_report.py` (append; reuses the file's `FIXTURE`, `json`, `rep` imports):
```python
def _with_sop(sop_block):
    d = json.loads(FIXTURE.read_text())
    m = json.loads(json.dumps(next(iter(d["models"].values()))))
    m["arm_tag"], m["sop"] = "spike_s4", sop_block
    d["models"]["spikingssm"] = m
    return d


SOP = {"op_class": "AC", "spike_fed_total": 10_485_760, "driven_ops": 2_621_440.0, "neurons": 40_960,
       "macs": 4_990_000_000, "acs": 2_621_440.0, "energy_j": 0.02296, "energy_dense_j": 0.02300,
       "saving": 0.0017, "ceiling": 0.0021, "spike_fed_share": 0.0021,
       "dense": {"total_macs": 5_000_000_000}, "spike_fed_macs": {"4": 10_485_760},
       "rates": {"stages": {"4": {"rate": 0.25, "rate_min_clip": 0.2, "rate_max_clip": 0.3, "drift": 0.01,
                                  "binary": True}}, "clips": ["a"] * 16, "warmup": 32},
       "constants": {"e_mac_j": 4.6e-12, "e_ac_j": 0.9e-12, "source": "Horowitz, ISSCC 2014"}}


def test_sop_table_is_written_for_spiking_rows(tmp_path):
    jp = tmp_path / "b.json"
    jp.write_text(json.dumps(_with_sop(SOP)))
    rep.generate(jp, tmp_path)
    md = (tmp_path / "sop_table.md").read_text()
    assert "spike_s4" in md and "AC" in md and "0.25" in md and "0.21" in md and "Horowitz" in md


def test_sop_table_shows_an_error_row(tmp_path):
    jp = tmp_path / "b.json"
    jp.write_text(json.dumps(_with_sop({"error": "non-binary"})))
    rep.generate(jp, tmp_path)
    assert "non-binary" in (tmp_path / "sop_table.md").read_text()


def test_no_sop_table_without_spiking_rows(tmp_path):
    rep.generate(FIXTURE, tmp_path)
    assert not (tmp_path / "sop_table.md").exists()
```

`test_bench_sop.py` (append; GPU, the user runs it on an idle GPU):
```python
from event_ssm.benchmark.bench_models import REPO

SPIKE_S4 = REPO / "external/ssms_event_cameras/RVT/RVT/ax1lj36q/checkpoints/epoch=000-step=25000-val_AP=0.34.ckpt"


@pytest.mark.gpu
@pytest.mark.skipif(not SPIKE_S4.exists(), reason="Stage-19 spike [4] checkpoint not present")
def test_sop_on_a_real_spike_checkpoint():
    from event_ssm.benchmark import bench_metrics as bm
    from event_ssm.benchmark.bench_clip import iter_rate_clips
    from event_ssm.integration.spiking_ckpt import arm_from_checkpoint
    dev = torch.device("cuda")
    model = build_model("spikingssm", device=dev, load_ckpt=True, ckpt_path=SPIKE_S4)
    arm = arm_from_checkpoint(str(SPIKE_S4))
    rates = sop.measure_nonzero_rates(model, iter_rate_clips(k=2, n_frames=12), warmup=4)
    assert 0.0 < rates["stages"][4]["rate"] < 1.0 and rates["stages"][4]["binary"] is True
    by_op = bm.profiler_network_flops(lambda: model.network_step(torch.zeros(1, 20, 256, 320, device=dev), None),
                                      device=dev)["by_op"]
    b = sop.sop_block(model.detector, arm, by_op, rates)
    assert b["op_class"] == "AC" and 0.0 < b["saving"] < b["ceiling"] < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ghost/miniforge3/envs/events_signals/bin/pytest code/event_ssm/tests/test_bench_metrics.py code/event_ssm/tests/test_bench_schema.py code/event_ssm/tests/test_bench_report.py -q`
Expected: the new tests fail (`by_op` key, `sop_section`, `sop_table.md` missing).

- [ ] **Step 3: Write the implementation**

`bench_metrics.py`, in `profiler_network_flops` replace the final two lines:
```python
    by_op: dict = {}
    for e in prof.events():
        if getattr(e, "flops", None):
            by_op[e.name] = by_op.get(e.name, 0) + int(e.flops)
    # by_op feeds the Stage-22 dense-MAC count (benchmark/sop.py separates MAC ops from elementwise ones)
    return {"counted_gflops": sum(by_op.values()) / 1e9, "source": "torch.profiler", "by_op": by_op}
```

`stage10_benchmark.py`:
- In `measure_model`, before the fvcore attempt: `by_op = None`. After `prof_counted = bm.profiler_network_flops(...)`: `by_op = prof_counted.get("by_op")`.
- Immediately before `return out` of `measure_model`:
```python
    # ---- Stage 22: SOP / energy accounting, LAST so its forward hooks can never touch a timed section ----
    if kind == "spikingssm":
        out["sop"] = sop_section(model, arm, by_op, smoke)
```
- New module-level function next to `check_env`:
```python
def sop_section(model, arm, by_op, smoke: bool) -> dict:
    """Stage-22 `sop` block for a SpikingSSM arm. Like the FLOP fallback, a failure is recorded, printed and
    reported, never allowed to abort the rest of the run."""
    from event_ssm.benchmark import sop
    if not by_op:
        print("[stage10] WARNING: SOP accounting skipped: no per-op profiler FLOPs")
        return {"error": "no per-op profiler FLOPs (the torch.profiler fallback did not run)"}
    k, warm, n = (2, 4, 8) if smoke else (sop.RATE_CLIPS, sop.RATE_WARMUP, sop.RATE_FRAMES)
    try:
        from event_ssm.benchmark.bench_clip import iter_rate_clips
        rates = sop.measure_nonzero_rates(model, iter_rate_clips(k=k, n_frames=warm + n), warmup=warm)
        return sop.sop_block(model.detector, arm, by_op, rates)
    except Exception as e:
        print(f"[stage10] WARNING: SOP accounting failed: {e}")
        return {"error": str(e)[:300]}
```

`stage10_report.py` — add before `generate` and call from it:
```python
def _sop_table_md(d: dict):
    """Stage-22 SOP / energy table for SpikingSSM rows (None when there are none). Operation-count estimate."""
    rows = [(k, m) for k, m in d["models"].items() if "sop" in m]
    if not rows:
        return None
    head = ("| Model | Op class | Nonzero rate per stage (min–max over clips) | Spike-fed MACs (M) | Driven ops (M) "
            "| Dense MACs (G) | Est. energy (mJ/frame) | Dense twin (mJ/frame) | Saving (%) | Ceiling (%) |")
    lines = [head, "|" + "---|" * 10]
    source = ""
    for kind, m in rows:
        s = m["sop"]
        if "error" in s:
            lines.append(f"| {_label(kind, m)} | error: {s['error']} |" + " — |" * 8)
            continue
        source = s["constants"]["source"]
        rates = ", ".join(f"s{st} {v['rate']:.3f} ({v['rate_min_clip']:.3f}–{v['rate_max_clip']:.3f})"
                          + ("" if v["binary"] else " non-binary") for st, v in s["rates"]["stages"].items())
        lines.append(f"| {_label(kind, m)} | {s['op_class']} | {rates} | {s['spike_fed_total'] / 1e6:.2f} "
                     f"| {s['driven_ops'] / 1e6:.2f} | {s['dense']['total_macs'] / 1e9:.3f} "
                     f"| {s['energy_j'] * 1e3:.3f} | {s['energy_dense_j'] * 1e3:.3f} | {s['saving'] * 100:.2f} "
                     f"| {s['ceiling'] * 100:.2f} |")
    lines += ["", "*Operation-count estimate. Driven ops = nonzero rate × spike-fed MACs, priced as accumulates "
              "(AC: binary spikes, no residual) or as MACs (graded, analog, residual). Each LIF neuron costs one MAC "
              "per frame. Dense twin = the same network priced dense (PureSSM-equivalent). Ceiling = the saving at "
              "zero activity. Memory traffic is not modelled; sparse-MAC savings assume zero-skipping hardware. "
              f"Constants: {source}.*"]
    return "\n".join(lines)
```
In `generate`, after the CSV is written:
```python
    sop_md = _sop_table_md(d)
    if sop_md:
        (out_dir / "sop_table.md").write_text(sop_md + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ghost/miniforge3/envs/events_signals/bin/pytest code/event_ssm/tests/test_bench_sop.py code/event_ssm/tests/test_bench_metrics.py code/event_ssm/tests/test_bench_schema.py code/event_ssm/tests/test_bench_report.py code/event_ssm/tests/test_bench_models.py code/event_ssm/tests/test_bench_clip.py -q`
Expected: all pass (gpu tests deselected).

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/benchmark/bench_metrics.py code/event_ssm/scripts/stage10_benchmark.py code/event_ssm/scripts/stage10_report.py code/event_ssm/tests/test_bench_metrics.py code/event_ssm/tests/test_bench_schema.py code/event_ssm/tests/test_bench_report.py code/event_ssm/tests/test_bench_sop.py
git commit -m "feat(stage22): SOP block in the benchmark JSON and a sop_table.md report"
```

---

### Task 4: Records

**Files:**
- Modify: `docs/specs/2026-10-07-stage22-sop-energy-design.md` (status line + a "Revision 2" section pointing to this plan's table)
- Modify: `docs/notes/Stage21_22_tooling_notes.md` (D5 rewritten; new D6 = the FLOP-total finding; usage line; test count)
- Modify: `CLAUDE.md` (Stage 20–22 bullet: SOP tooling + the FLOP-total finding, pending the user's decision)

- [ ] **Step 1:** Spec: set status to "approved 2026-10-07, amended by Revision 2 (logic check)" and add a short §8 listing the eight amendments with a pointer to the plan's table.
- [ ] **Step 2:** Notes: D5 = what the SOP tooling measures and how (denominator, op class, rates over 16 sequences, neuron MACs, the ceiling ≈ 1 %); D6 = the Stage-10/16 FLOP finding with the measured magnitudes (0.832 GFLOPs double-counted for EventSSM and PureSSM; ≈ 0.72 GFLOPs of PureSSM spatial kernels uncounted; baseline S5 feed-forward probably double-counted, unverified), recorded as a user decision, numbers in Ch. 5 unchanged until then.
- [ ] **Step 3:** CLAUDE.md: one line each for the SOP tooling and the FLOP finding.
- [ ] **Step 4:** Commit: `git commit -m "docs(stage22): SOP accounting records; FLOP-total double count found"`.
