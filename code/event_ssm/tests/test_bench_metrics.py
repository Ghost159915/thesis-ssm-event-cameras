"""Stage-10 bench_metrics unit tests — CPU-only by default (GPU busy with Stage-9 sweeps)."""
import time
import torch
import pytest
from event_ssm.benchmark.bench_metrics import time_fn, PowerSampler, state_bytes, measure_peak_vram


def test_time_fn_measures_sleep():
    r = time_fn(lambda: time.sleep(0.001), warmup=2, iters=20, device=torch.device("cpu"))
    assert 0.8 <= r["p50_ms"] <= 10.0          # sleep(1ms) plus scheduler noise
    assert r["p50_ms"] <= r["p95_ms"]
    assert r["iters"] == 20 and r["hz"] > 0
    assert set(r) == {"mean_ms", "std_ms", "p50_ms", "p95_ms", "hz", "iters"}


def test_power_sampler_with_injected_reader():
    with PowerSampler(interval_s=0.05, _read=lambda: 42.0) as ps:
        time.sleep(0.3)
    assert len(ps.samples) >= 3
    assert abs(ps.mean_w - 42.0) < 1e-6


def test_power_sampler_drops_sample_on_reader_timeout():
    # Important-3: read_gpu_power_w propagates subprocess.TimeoutExpired on a hung nvidia-smi
    # call (timeout=2); the sampler loop must swallow it per-sample rather than dying/hanging.
    import subprocess as sp

    def _flaky():
        raise sp.TimeoutExpired(cmd="nvidia-smi", timeout=2)

    with PowerSampler(interval_s=0.05, _read=_flaky) as ps:
        time.sleep(0.3)
    assert ps.samples == []            # every sample dropped
    assert ps.mean_w != ps.mean_w       # nan (no samples) -- thread never died mid-loop


def test_state_bytes_recursive():
    t1 = torch.zeros(2, 3, dtype=torch.float32)      # 24 B
    t2 = torch.zeros(4, dtype=torch.float16)         # 8 B
    nested = [(t1, {"a": t2}), None, [t2]]
    assert state_bytes(nested) == 24 + 8 + 8
    assert state_bytes(None) == 0


def test_measure_peak_vram_cpu_returns_sentinel():
    assert measure_peak_vram(lambda: None, device=torch.device("cpu")) == -1.0


from event_ssm.benchmark.bench_metrics import (
    mamba2_layer_macs_per_token, s5_block_macs_per_token, fvcore_network_flops,
)


def test_mamba2_macs_hand_case():
    # d_model=128, expand=2 -> d_inner=256; headdim=64 -> nheads=4; S=64; k=4; ngroups=1
    # in_proj: 128*(2*256+2*64+4)=128*644=82432 ; conv: (256+128)*4=1536
    # scan: 3*256*64=49152 ; norm/gate: 2*256=512 ; out_proj: 256*128=32768  => 166400
    assert mamba2_layer_macs_per_token(128, d_state=64, d_conv=4, expand=2, headdim=64) == 166400


def test_s5_macs_hand_case():
    # dim=c=128, P=128, ff_mult=1, glu=True (complex MAC = 4 real MACs, spec §5.2):
    # Bu: 4*128*128=65536 ; Lambda*x: 4*128=512 ; Cx: 65536 ; D: 128 ; discretize: 10*128=1280
    # ff: enc 128*128*2=32768 + dec 128*128=16384 ; norms: 4*128=512  => 182656
    assert s5_block_macs_per_token(128, 128) == 182656


def test_fvcore_counts_toy_conv():
    import torch.nn as nn
    m = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    r = fvcore_network_flops(m, (torch.randn(1, 3, 16, 16),))
    # 3*8*3*3*16*16 = 55296 MACs -> 110592 FLOPs = 1.10592e-4 GFLOPs
    assert abs(r["counted_gflops"] - 55296 * 2 / 1e9) / (55296 * 2 / 1e9) < 0.05
    assert isinstance(r["unsupported_ops"], dict)


from event_ssm.benchmark.bench_metrics import profiler_network_flops


def test_profiler_counts_toy_conv():
    import torch.nn as nn
    m = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    x = torch.randn(1, 3, 16, 16)
    r = profiler_network_flops(lambda: m(x), device=torch.device("cpu"))
    # Same 55,296-MAC/110,592-FLOP toy case as test_fvcore_counts_toy_conv above -- pins the
    # unit convention empirically: torch.profiler's `flops` field for aten::conv2d is already
    # true FLOPs (2x MACs), matching fvcore's *2 convention exactly, so no extra scaling is
    # applied inside profiler_network_flops.
    assert abs(r["counted_gflops"] - 110592 / 1e9) / (110592 / 1e9) < 0.10
    assert r["source"] == "torch.profiler"


def test_profiler_flops_carry_a_per_op_breakdown():
    from event_ssm.benchmark import bench_metrics as bm
    conv = torch.nn.Conv2d(3, 4, 3, bias=False)
    x = torch.randn(1, 3, 8, 8)
    r = bm.profiler_network_flops(lambda: conv(x), device=None)
    assert r["by_op"]["aten::conv2d"] == 2 * 6 * 6 * 4 * 3 * 3 * 3
    assert sum(r["by_op"].values()) / 1e9 == pytest.approx(r["counted_gflops"])


# ---- FLOP add-on v2 (Stage-22 finding D6): count only the work the profiler cannot see -----------------------
def test_mamba2_kernel_macs_exclude_the_profiled_projections():
    from event_ssm.benchmark.bench_metrics import mamba2_kernel_macs_per_token
    # the full-layer hand case above minus in_proj (82432) and out_proj (32768), which are nn.Linear layers the
    # profiler already counts as aten::mm: conv 1536 + scan 49152 + gated norm 512
    assert mamba2_kernel_macs_per_token(128, d_state=64, d_conv=4, expand=2) == 51200
    assert mamba2_kernel_macs_per_token(128, d_state=64) == mamba2_layer_macs_per_token(128, d_state=64) - 82432 - 32768


def test_bimamba_kernel_macs_two_directions_one_norm():
    from event_ssm.benchmark.bench_metrics import bimamba_kernel_macs_per_token
    # d_model 64, d_state 16: d_inner 128, conv_dim 160; per direction conv 640 + scan 6144; one shared norm 256
    assert bimamba_kernel_macs_per_token(64, d_state=16, d_conv=4, expand=2) == 2 * (640 + 6144) + 256


def test_s5_unprofiled_macs_top_up_the_profiled_complex_products():
    from event_ssm.benchmark.bench_metrics import s5_unprofiled_macs_per_token
    # the full hand case minus the FF layers (32768 + 16384, profiled as aten::mm) minus 1 real MAC per complex MAC
    # of B~u and C~x (128*128 each), which the profiler counts as aten::bmm at 2 FLOPs per complex MAC
    assert s5_unprofiled_macs_per_token(128, 128) == 182656 - 32768 - 16384 - 2 * 128 * 128


def test_analytic_unprofiled_gflops_sums_temporal_and_spatial():
    from event_ssm.benchmark.bench_metrics import analytic_unprofiled_gflops
    temporal = [{"kind": "mamba2", "tokens": 10, "d_model": 128, "d_state": 64, "d_conv": 4, "expand": 2,
                 "headdim": 64},
                {"kind": "s5", "tokens": 3, "dim": 128, "state_dim": 128}]
    spatial = [{"kind": "bimamba", "tokens": 5, "d_model": 64, "d_state": 16, "d_conv": 4, "expand": 2}]
    macs = 10 * 51200 + 3 * 100736 + 5 * (2 * 6784 + 256)
    assert analytic_unprofiled_gflops(temporal, spatial) == pytest.approx(2 * macs / 1e9)


def test_s5_complex_products_are_profiled_at_one_real_mac_each():
    # the premise of s5_unprofiled_macs_per_token, checked on the benchmark's own streaming path (CPU): every S5
    # stage profiles B~u and C~x as two aten::bmm of shape [N,P,P]@[N,P,1] at 2*N*P*P FLOPs each
    import collections
    from torch.profiler import ProfilerActivity, profile
    from event_ssm.benchmark.bench_models import build_model
    m = build_model("baseline", device=torch.device("cpu"), load_ckpt=False)
    m.autocast_bf16 = False
    x = torch.zeros(1, 20, 256, 320)
    m.network_step(x, None)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True, record_shapes=True) as prof:
        m.network_step(x, None)
    matvec = collections.Counter()
    for e in prof.events():
        if e.name == "aten::bmm" and getattr(e, "flops", None) and e.input_shapes[1][-1] == 1:
            matvec[tuple(e.input_shapes[0])] += e.flops
    assert dict(matvec) == {(n, p, p): 2 * (2 * n * p * p) for n, p in ((5120, 64), (1280, 128), (320, 256), (80, 512))}
