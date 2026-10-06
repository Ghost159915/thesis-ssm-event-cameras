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


def _small_bb(**kw):
    from event_ssm.models.puressm import BiMambaSpatialStages
    torch.manual_seed(0)
    spatial = BiMambaSpatialStages(in_channels=20, depths=(1, 1, 1, 1), d_state=16,
                                   drop_path_rate=0.0)
    return SpikingSSMBackbone(spatial=spatial, spiking_stages=(4,), **kw)   # construction only


def test_backbone_checkpoint_carries_arm_at_nested_prefixes():
    """D14 end-to-end at backbone level: the arm metadata sits under the spiking stage's prefix
    only (none on the backbone itself, nor on plain stages), a same-arm load is strict, and a
    cross-arm load raises from inside the nested module."""
    sd = _small_bb(lif_kwargs=dict(output_mode="analog")).state_dict()
    extra = sorted(k for k in sd if k.endswith("_extra_state"))
    assert extra == ["temporal.4._extra_state", "temporal.4.lif._extra_state"]
    _small_bb(lif_kwargs=dict(output_mode="analog")).load_state_dict(sd, strict=True)
    with pytest.raises(ValueError, match="output_mode"):
        _small_bb(lif_kwargs=dict(output_mode="spike")).load_state_dict(sd)
    with pytest.raises(ValueError, match="residual"):
        _small_bb(residual=True, lif_kwargs=dict(output_mode="analog")).load_state_dict(sd)
