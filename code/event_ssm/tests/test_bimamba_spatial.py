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
    # far more than a row-axis block does at init.
    # NOTE: the perturbation must NOT be uniform across channels — the block is
    # pre-norm, and a uniform all-channel shift is in LayerNorm's null space
    # (the scan would see identical input and off-site effects would be exactly 0).
    from event_ssm.spatial import BiMamba2DBlock
    torch.manual_seed(0)
    blk = BiMamba2DBlock(64, axis="col").to(device).eval()
    x = torch.randn(1, 64, 16, 20, device=device)
    x2 = x.clone()
    x2[0, 3, 2, 7] += 5.0                       # single-channel bump at (h=2, w=7)
    d = (blk(x2) - blk(x)).abs().sum(dim=1)[0]  # (H, W)
    col_effect = d[:, 7].sum() - d[2, 7]
    row_effect = d[2, :].sum() - d[2, 7]
    assert col_effect > 0, "no off-site propagation at all — scan branch dead"
    assert col_effect > row_effect, "col-axis block did not mix along its column"


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
