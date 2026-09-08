"""Stage 17 — SpikingSSMBlock composition logic, on CPU.

`spiking_temporal` imports `mamba_ssm`, which is CUDA-only, so the real block cannot be
exercised without the 5070 Ti. The *composition* logic can be: this module substitutes a
stand-in Mamba-2 (a real, stateful leaky recurrence — not an identity, so state carry is
genuinely tested) and checks the parts that are ours rather than mamba-ssm's — state
threading, the residual switch, the firing-rate passthrough, the fold/unfold re-export.

The numerics of the real kernel are covered in `test_spiking_temporal.py` (gpu-marked).
Stubs are installed per-fixture and torn down, so they never leak into the gpu tests.

d_model=32 throughout: MambaTemporalBlock asserts (d_model*expand) % headdim == 0, so with
the defaults expand=2/headdim=64 the smallest legal width is 32 (real stage dims are
64/128/256/512)."""
import importlib
import sys
import types

import pytest
import torch
import torch.nn as nn


class _FakeMamba2(nn.Module):
    def __init__(self, d_model, **_kw):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model)


def _fake_scan(layer, x, state, step_scale=1.0):
    """Leaky linear recurrence over time with a carried state — same contract as
    `mamba2_scan_time`: (layer, x(N,L,C), state) -> (y(N,L,C), new_state)."""
    h = state if state is not None else x.new_zeros(x.shape[0], x.shape[2])
    outs = []
    for t in range(x.shape[1]):
        h = 0.5 * h + layer.proj(x[:, t])
        outs.append(h)
    return torch.stack(outs, dim=1), h


@pytest.fixture
def mods():
    """Import spiking_temporal against stubbed mamba-ssm; restore sys.modules afterwards."""
    targets = ("event_ssm.temporal.mamba_temporal",
               "event_ssm.models.spikingssm.spiking_temporal")
    saved = {n: sys.modules.get(n) for n in targets}

    fake_mamba = types.ModuleType("mamba_ssm")
    fake_mamba.Mamba2 = _FakeMamba2
    fake_scan_mod = types.ModuleType("event_ssm.temporal._scan")
    fake_scan_mod.mamba2_scan_time = _fake_scan

    saved_stub = {n: sys.modules.get(n) for n in ("mamba_ssm", "event_ssm.temporal._scan")}
    sys.modules["mamba_ssm"] = fake_mamba
    sys.modules["event_ssm.temporal._scan"] = fake_scan_mod
    for n in targets:
        sys.modules.pop(n, None)
    try:
        st = importlib.import_module("event_ssm.models.spikingssm.spiking_temporal")
        mt = importlib.import_module("event_ssm.temporal.mamba_temporal")
        yield st.SpikingSSMBlock, mt.MambaTemporalBlock
    finally:
        for n in targets:
            sys.modules.pop(n, None)
        for n, v in {**saved, **saved_stub}.items():
            if v is not None:
                sys.modules[n] = v
            else:
                sys.modules.pop(n, None)


def test_forward_shape_and_state_structure(mods):
    Block, _ = mods
    blk = Block(d_model=32, num_layers=2).eval()
    out, state = blk(torch.randn(4, 6, 32))
    assert out.shape == (4, 6, 32)
    ssm_state, mem = state
    assert len(ssm_state) == 2, "one SSM state per layer"
    assert mem.shape == (4, 32), "membrane is (N, C), dim0=N for the b-major reshape"


def test_output_is_binary_by_default(mods):
    Block, _ = mods
    out, _ = Block(d_model=32).eval()(torch.randn(4, 6, 32))
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_state_carry_split_equals_full(mods):
    """Both halves of the state — SSM and membrane — must thread correctly across clips."""
    torch.manual_seed(0)
    Block, _ = mods
    blk = Block(d_model=32).eval()
    x = torch.randn(4, 10, 32)
    full, _ = blk(x)
    a, st = blk(x[:, :4])
    b, _ = blk(x[:, 4:], st)
    assert torch.allclose(full, torch.cat([a, b], dim=1), atol=1e-5)


def test_carrying_no_state_differs_from_carrying_state(mods):
    """Guards against a silently dropped state — the failure mode that would make the
    split/full test above pass for the wrong reason."""
    torch.manual_seed(0)
    Block, _ = mods
    blk = Block(d_model=32, output_mode="analog").eval()
    x = torch.randn(4, 10, 32)
    _, st = blk(x[:, :4])
    with_state, _ = blk(x[:, 4:], st)
    without_state, _ = blk(x[:, 4:])
    assert not torch.allclose(with_state, without_state, atol=1e-4)


def test_residual_switch(mods):
    torch.manual_seed(0)
    Block, _ = mods
    x = torch.randn(4, 6, 32)
    plain = Block(d_model=32, residual=False).eval()
    res = Block(d_model=32, residual=True).eval()
    res.load_state_dict(plain.state_dict())
    a, _ = plain(x)
    b, _ = res(x)
    assert torch.allclose(b, a + x, atol=1e-6), "residual must add the analog input back"


def test_gradients_flow_through_the_composition(mods):
    Block, _ = mods
    blk = Block(d_model=32)
    x = torch.randn(4, 6, 32, requires_grad=True)
    blk(x)[0].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    grads = [p.grad for p in blk.parameters() if p.grad is not None]
    assert grads, "no parameter received gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert blk.lif.beta_logit.grad is not None, "spiking parameters must train too"


def test_firing_rate_exposed_for_monitors(mods):
    Block, _ = mods
    blk = Block(d_model=32)
    blk(torch.randn(4, 6, 32))
    assert 0.0 <= blk.last_firing_rate <= 1.0


def test_fold_unfold_reexported(mods):
    """Drop-in compatibility with MambaTemporalBlock: same helpers, same behaviour."""
    Block, Mamba = mods
    assert Block.fold is Mamba.fold and Block.unfold is Mamba.unfold
    x = torch.randn(5, 3, 8, 4, 6)                             # (L, B, C, H, W)
    folded, dims = Block.fold(x)
    assert folded.shape == (3 * 4 * 6, 5, 8)
    assert torch.allclose(Block.unfold(folded, dims), x)
