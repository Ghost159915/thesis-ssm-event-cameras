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
    alive = []          # keep every tainted tensor referenced: a freed tensor's id() is reused by later tensors
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
                    alive.append(out)
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
