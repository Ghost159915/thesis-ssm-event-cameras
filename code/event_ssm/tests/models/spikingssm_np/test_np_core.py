"""NumPy spiking block — framework-free correctness (no torch import; runs on a NumPy-only machine).

What is checked here and what is not:
  * the per-timestep SSD recurrence against the INDEPENDENT dual (masked-attention) form of Mamba-2,
    y = (L o C B^T) (dt x) with L[t, s] = exp(sum_{r=s+1..t} dt_r A), including a nonzero initial state;
  * the whole layer, stepped, against a sequence-level re-derivation (full-clip causal conv + dual form);
  * streaming invariance (any clip split, incl. one frame at a time, equals the full clip, bit for bit);
  * the LIF recurrence on hand-computed values, the beta/threshold maps, the loader and its refusals;
  * the parity criterion itself (it must pass on noise and fail on a planted bug).
That the NumPy layer reproduces the REAL Mamba-2 kernels is the gpu-marked `test_np_block_parity_gpu.py`; the LIF
against the real `LIFReadout` is `test_np_lif_parity.py` (torch, CPU).
"""
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from event_ssm.eval.np_parity import spiking_parity
from event_ssm.models.spikingssm_np import (LIFReadoutNP, Mamba2LayerNP, MambaTemporalBlockNP, SpikingSSMBlockNP,
                                            ssd_step)
from event_ssm.models.spikingssm_np.mamba2 import silu, softplus
from event_ssm.tests.models.spikingssm_np._synthetic import block_sd, mamba2_layer_sd

CODE = pathlib.Path(__file__).resolve().parents[4]


# ---- independent references -----------------------------------------------------------------------------------------

def ssd_dual(x, dt, A, B, C, D, h0):
    """Mamba-2's dual form over a whole clip. x (N, L, H, P), dt (N, L, H), B/C (N, L, S), h0 (N, H, P, S)."""
    la = np.cumsum(dt * A, axis=1)                                         # (N, L, H): log of the running decay
    diff = la[:, :, None, :] - la[:, None, :, :]                           # (N, t, s, H) = sum_{r=s+1..t} dt_r A
    causal = np.tril(np.ones((x.shape[1], x.shape[1]), bool))[None, :, :, None]
    Lmat = np.exp(np.where(causal, diff, -np.inf))
    G = np.einsum("ntk,nsk->nts", C, B)                                    # C_t . B_s (heads share B, C)
    y = np.einsum("nts,ntsh,nsh,nshp->nthp", G, Lmat, dt, x)
    y += np.exp(la)[..., None] * np.einsum("nhpk,ntk->nthp", h0, C)        # the initial state, decayed to t
    y += D[None, None, :, None] * x
    h_end = (np.exp(la[:, -1])[:, :, None, None] * h0
             + np.einsum("nsh,nsh,nshp,nsk->nhpk", np.exp(la[:, -1:] - la), dt, x, B))
    return y, h_end


def layer_sequence_reference(layer, x, conv0, h0, step_scale=1.0):
    """The whole Mamba-2 layer over a clip, written at sequence level (no stepping): full-clip causal conv, the dual
    form above, gated RMSNorm (mamba_ssm's rms_norm_ref, norm_before_gate=False), out_proj."""
    n, length, _ = x.shape
    zxbcdt = x @ layer.in_proj_w.T
    z, xBC, dt_raw = np.split(zxbcdt, [layer.d_inner, layer.d_inner + layer.conv_dim], axis=-1)
    ext = np.concatenate([conv0, xBC], axis=1)                             # (N, K-1+L, conv_dim)
    k = layer.d_conv
    conv = sum(ext[:, j:j + length] * layer.conv_w[:, j] for j in range(k)) + layer.conv_b
    xBC = silu(conv)
    xs, B, C = np.split(xBC, [layer.d_inner, layer.d_inner + layer.d_state], axis=-1)
    dt = step_scale * softplus(dt_raw + layer.dt_bias)
    y, h_end = ssd_dual(xs.reshape(n, length, layer.nheads, layer.headdim), dt, layer.A, B, C, layer.D, h0)
    y = y.reshape(n, length, layer.d_inner) * silu(z)
    y = y / np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + layer.norm_eps) * layer.norm_w
    return y @ layer.out_proj_w.T, (ext[:, -(k - 1):], h_end)


def _layer(d_model=32, seed=0, dtype=np.float64, **kw):
    return Mamba2LayerNP.from_state_dict(mamba2_layer_sd(np.random.default_rng(seed), d_model, **kw), dtype=dtype)


def _random_state(layer, n, rng):
    return (rng.normal(size=(n, layer.d_conv - 1, layer.conv_dim)),
            rng.normal(size=(n, layer.nheads, layer.headdim, layer.d_state)))


# ---- the recurrence -------------------------------------------------------------------------------------------------

def test_ssd_step_matches_dual_form():
    rng = np.random.default_rng(1)
    n, length, h, p, s = 3, 23, 4, 8, 16
    x = rng.normal(size=(n, length, h, p))
    dt = softplus(rng.normal(-2, 1, (n, length, h)))
    A = -rng.uniform(1, 16, h)
    B, C = rng.normal(size=(n, length, s)), rng.normal(size=(n, length, s))
    D, h0 = rng.normal(size=h), rng.normal(size=(n, h, p, s))
    y_ref, h_ref = ssd_dual(x, dt, A, B, C, D, h0)
    hs, ys = h0, []
    for t in range(length):
        y_t, hs = ssd_step(hs, x[:, t], dt[:, t], A, B[:, t], C[:, t], D)
        ys.append(y_t)
    np.testing.assert_allclose(np.stack(ys, 1), y_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(hs, h_ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("step_scale", [1.0, 2.5])
def test_layer_matches_sequence_reference(step_scale):
    rng = np.random.default_rng(2)
    layer = _layer(d_model=32)
    x = rng.normal(size=(5, 17, 32))
    conv0, h0 = _random_state(layer, 5, rng)
    y_ref, (conv_ref, h_ref) = layer_sequence_reference(layer, x, conv0, h0, step_scale)
    y, [(conv, h)] = MambaTemporalBlockNP([layer], step_scale=step_scale).forward(x, [(conv0, h0)])
    np.testing.assert_allclose(y, y_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(conv, conv_ref, rtol=0, atol=0)               # the carry is the raw frames, exactly
    np.testing.assert_allclose(h, h_ref, rtol=1e-9, atol=1e-9)


def test_step_scale_changes_the_output():
    x = np.random.default_rng(3).normal(size=(4, 9, 32))
    layer = _layer()
    y1, _ = MambaTemporalBlockNP([layer], 1.0).forward(x)
    y2, _ = MambaTemporalBlockNP([layer], 2.0).forward(x)
    assert np.abs(y1 - y2).max() > 1e-3


def test_sequence_start_is_zero_state():
    layer = _layer()
    x = np.random.default_rng(4).normal(size=(4, 6, 32))
    blk = MambaTemporalBlockNP([layer])
    a, _ = blk.forward(x, None)
    b, _ = blk.forward(x, [layer.zero_state(4)])
    np.testing.assert_array_equal(a, b)


# ---- the composed block ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["spike", "graded", "analog"])
@pytest.mark.parametrize("splits", [[7], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [3, 11]])
def test_streaming_split_equals_full(mode, splits):
    blk = SpikingSSMBlockNP.from_state_dict(block_sd(d_model=32, num_layers=2, output_mode=mode))
    x = np.random.default_rng(5).normal(size=(6, 16, 32)).astype(np.float32)
    full, (st_full, mem_full) = blk.forward(x)
    parts, state = [], None
    for a, b in zip([0] + splits, splits + [16]):
        out, state = blk.forward(x[:, a:b], state)
        parts.append(out)
    np.testing.assert_array_equal(np.concatenate(parts, 1), full)          # same op order -> bit-identical
    np.testing.assert_array_equal(state[1], mem_full)
    for (c1, s1), (c2, s2) in zip(state[0], st_full):
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(s1, s2)


def test_state_layout_matches_torch_contract():
    blk = SpikingSSMBlockNP.from_state_dict(block_sd(d_model=32, d_state=16, headdim=16, num_layers=2))
    out, (mamba_state, mem) = blk.forward(np.zeros((5, 3, 32), np.float32))
    assert out.shape == (5, 3, 32) and mem.shape == (5, 32)
    assert len(mamba_state) == 2
    for conv, ssm in mamba_state:
        assert conv.shape == (5, 3, 64 + 2 * 16)                           # (N, d_conv-1, d_inner + 2*d_state)
        assert ssm.shape == (5, 64 // 16, 16, 16)                          # (N, nheads, headdim, d_state)


def test_spike_output_is_binary_and_not_degenerate():
    blk = SpikingSSMBlockNP.from_state_dict(block_sd(d_model=32, output_mode="spike"))
    out, _ = blk.forward(np.random.default_rng(6).normal(size=(16, 20, 32)))
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}
    assert 0.0 < out.mean() < 1.0


def test_graded_is_spike_times_membrane_and_analog_is_the_membrane():
    x = np.random.default_rng(7).normal(size=(8, 12, 32))
    outs = {m: SpikingSSMBlockNP.from_state_dict(block_sd(output_mode=m), dtype=np.float64).forward(x)[0]
            for m in ("spike", "graded", "analog")}
    np.testing.assert_array_equal(outs["graded"], outs["spike"] * outs["analog"])
    np.testing.assert_array_equal(outs["spike"], (outs["analog"] >= 1.0).astype(float))


def test_residual_adds_the_input():
    x = np.random.default_rng(8).normal(size=(4, 5, 32)).astype(np.float32)
    plain, _ = SpikingSSMBlockNP.from_state_dict(block_sd(residual=False)).forward(x)
    res, _ = SpikingSSMBlockNP.from_state_dict(block_sd(residual=True)).forward(x)
    np.testing.assert_array_equal(res, plain + x)


def test_float32_tracks_float64():
    sd = block_sd(output_mode="graded")
    x = np.random.default_rng(9).normal(size=(16, 24, 32))
    b32, b64 = SpikingSSMBlockNP.from_state_dict(sd), SpikingSSMBlockNP.from_state_dict(sd, dtype=np.float64)
    o32, _ = b32.forward(x)
    o64, _ = b64.forward(x)
    assert o32.dtype == np.float32 and o64.dtype == np.float64
    r = spiking_parity(o32, o64, b32.last_ssm_out, b64.last_ssm_out, b64.lif.beta, b64.lif.threshold, "graded")
    assert r["parity"], r
    assert r["y_rel_l2_err"] < 1e-5, r


# ---- LIF -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reset,expect_spk,expect_pre", [
    # beta = 0.5 exactly (logit 0 through the eps-squeezed sigmoid), theta = 1, constant input 0.6
    ("subtract", [0, 0, 1, 0, 0, 1], [0.6, 0.9, 1.05, 0.625, 0.9125, 1.05625]),
    ("zero", [0, 0, 1, 0, 0, 1], [0.6, 0.9, 1.05, 0.6, 0.9, 1.05]),
])
def test_lif_hand_computed(reset, expect_spk, expect_pre):
    y = np.full((1, 6, 1), 0.6)
    outs = {m: LIFReadoutNP(np.zeros(1), np.ones(1), output_mode=m, reset=reset, dtype=np.float64).forward(y)[0]
            for m in ("spike", "graded", "analog")}
    np.testing.assert_allclose(outs["spike"][0, :, 0], expect_spk)
    np.testing.assert_allclose(outs["analog"][0, :, 0], expect_pre, rtol=1e-12)
    np.testing.assert_allclose(outs["graded"][0, :, 0], np.array(expect_spk) * expect_pre, rtol=1e-12)


def test_lif_fires_at_equality_and_carries_membrane():
    lif = LIFReadoutNP(np.zeros(1), np.ones(1), output_mode="spike", dtype=np.float64)
    out, mem = lif.forward(np.full((1, 1, 1), 1.0))
    assert out[0, 0, 0] == 1.0 and mem[0, 0] == 0.0                        # u = 0 fires (torch: u >= 0)
    out, mem = lif.forward(np.full((1, 1, 1), 0.5), mem=np.full((1, 1), 1.2))
    assert out[0, 0, 0] == 1.0 and mem[0, 0] == pytest.approx(0.1)          # 0.5*1.2 + 0.5 = 1.1 -> 0.1


def test_beta_and_threshold_maps():
    lif = LIFReadoutNP(np.array([-1e4, 0.0, 1e4]), np.array([-5.0, 1e-4, 2.0]), dtype=np.float64)
    assert np.all(lif.beta > 0) and np.all(lif.beta < 1)
    np.testing.assert_allclose(lif.beta, [1e-4, 0.5, 1 - 1e-4], rtol=1e-12)
    np.testing.assert_allclose(lif.threshold, [1e-3, 1e-3, 2.0])


# ---- loader --------------------------------------------------------------------------------------------------------

def test_loader_infers_config_and_arm():
    sd = block_sd(d_model=64, d_state=16, headdim=32, num_layers=2, output_mode="graded", reset="zero",
                  residual=True, prefix="backbone.temporal.3.")
    blk = SpikingSSMBlockNP.from_state_dict(sd, prefix="backbone.temporal.3.")
    assert len(blk.ssm.layers) == 2 and blk.residual and blk.output_mode == "graded" and blk.lif.reset == "zero"
    l0 = blk.ssm.layers[0]
    assert (l0.d_model, l0.d_inner, l0.nheads, l0.headdim, l0.d_state, l0.d_conv) == (64, 128, 4, 32, 16, 4)


def test_loader_reads_torch_like_tensors_by_duck_typing():
    class FakeTensor:                                                      # .detach().cpu().float().numpy()
        def __init__(self, a): self.a = a
        def detach(self): return self
        def cpu(self): return self
        def float(self): return FakeTensor(self.a.astype(np.float32))
        def numpy(self): return self.a
    sd = {k: (FakeTensor(v) if isinstance(v, np.ndarray) else v) for k, v in block_sd().items()}
    ref = SpikingSSMBlockNP.from_state_dict(block_sd())
    blk = SpikingSSMBlockNP.from_state_dict(sd)
    x = np.random.default_rng(10).normal(size=(2, 3, 32))
    np.testing.assert_array_equal(blk.forward(x)[0], ref.forward(x)[0])


def test_loader_refuses_unsupported_configurations():
    sd = block_sd()
    bad = dict(sd)
    bad["ssm.layers.0.D"] = np.ones(64)                                    # D_has_hdim=True
    with pytest.raises(ValueError, match="D_has_hdim"):
        SpikingSSMBlockNP.from_state_dict(bad)
    bad = dict(sd)
    w = sd["ssm.layers.0.in_proj.weight"]
    bad["ssm.layers.0.in_proj.weight"] = np.concatenate([w, w[:32]])       # ngroups / gated-MLP shaped
    with pytest.raises(ValueError, match="in_proj"):
        SpikingSSMBlockNP.from_state_dict(bad)
    bad = {k: v for k, v in sd.items() if k != "_extra_state"}
    with pytest.raises(ValueError, match="residual"):
        SpikingSSMBlockNP.from_state_dict(bad)
    with pytest.raises(ValueError, match="contradicts"):
        SpikingSSMBlockNP.from_state_dict(sd, output_mode="analog")
    bad = dict(sd)
    bad["lif._extra_state"] = {**sd["lif._extra_state"], "version": 2}
    with pytest.raises(ValueError, match="extra state"):
        SpikingSSMBlockNP.from_state_dict(bad)
    with pytest.raises(KeyError, match="no keys"):
        SpikingSSMBlockNP.from_state_dict(sd, prefix="backbone.temporal.2.")


# ---- the parity criterion has teeth ----------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["spike", "graded", "analog"])
def test_parity_criterion_accepts_noise_and_rejects_a_bug(mode):
    sd = block_sd(output_mode=mode)
    blk = SpikingSSMBlockNP.from_state_dict(sd, dtype=np.float64)
    lif = blk.lif
    x = np.random.default_rng(11).normal(size=(32, 30, 32))
    out_b, _ = blk.forward(x)
    y_b = blk.last_ssm_out

    # (1) noise: perturb y by 1e-3 relative -> some flips, all near threshold, so all explained
    y_a = y_b * (1 + 1e-3 * np.random.default_rng(12).standard_normal(y_b.shape))
    out_a, _ = lif.forward(y_a)
    r = spiking_parity(out_a, out_b, y_a, y_b, lif.beta, lif.threshold, mode)
    assert r["parity"] and r["rows_with_flip"] > 0, r                     # flips happened, and all were explained

    # (2) a bug that leaves y intact: the readout uses a 5 % lower threshold -> flips far from threshold
    buggy = LIFReadoutNP(sd["lif.beta_logit"], 0.95 * sd["lif.threshold_raw"], output_mode=mode,
                         dtype=np.float64)
    out_bug, _ = buggy.forward(y_b)
    r = spiking_parity(out_bug, out_b, y_b, y_b, lif.beta, lif.threshold, mode)
    assert not r["parity"], r


# ---- the package stays torch-free ------------------------------------------------------------------------------------

def test_package_imports_without_torch():
    code = ("import sys; import event_ssm.models.spikingssm_np, event_ssm.eval.np_parity; "
            "assert 'torch' not in sys.modules, 'torch was imported'")
    env = dict(os.environ, PYTHONPATH=str(CODE))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
