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
