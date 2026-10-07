"""Fact-check decision D1 (2026-10-07): RVT's original ConvLSTM backbone (Gehrig & Scaramuzza, CVPR 2023) in the
S5 fork's clip interface, so the public RVT-B Gen1 checkpoint can be evaluated on the Stage-9 true-rate test set
by the same evaluator as every other model. CPU."""
import ast
import pathlib

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
ORIG = REPO / "external/rvt_original/models/detection/recurrent_backbone/maxvit_rnn.py"
VENDORED = REPO / "code/event_ssm/baselines/rvt_maxvit_lstm.py"
RVTB_CKPT = REPO / "checkpoints/rvt-b-gen1.ckpt"


def _ast(path):
    tree = ast.parse(path.read_text())
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        tree.body = body[1:]                                   # module docstring (the vendoring note)
    for node in ast.walk(tree):                                # the one permitted change: `.base` made absolute
        if isinstance(node, ast.ImportFrom) and node.module in ("base", "models.detection.recurrent_backbone.base"):
            node.module, node.level = "base", 1
    return ast.dump(tree)


@pytest.mark.skipif(not ORIG.exists(), reason="original RVT clone (external/rvt_original) not present")
def test_vendored_backbone_is_the_original_code():
    assert _ast(VENDORED) == _ast(ORIG)


@pytest.fixture(scope="module")
def cfg():
    from event_ssm.benchmark.bench_models import compose_cfg
    return compose_cfg("baseline")


@pytest.fixture(scope="module")
def backbone(cfg):
    from event_ssm.baselines.rvt_lstm import RVTLSTMBackbone
    torch.manual_seed(0)
    return RVTLSTMBackbone(cfg.model.backbone).eval()


def test_clip_forward_equals_per_frame_rvt(backbone):
    from event_ssm.baselines.rvt_maxvit_lstm import RNNDetector
    x = torch.rand(3, 1, 20, 256, 320)
    with torch.no_grad():
        feats, states = backbone(x, None, None, train_step=False)
        ref, ref_states = [], None
        for t in range(3):
            f, ref_states = RNNDetector.forward(backbone, x[t], ref_states)
            ref.append(f)
    assert set(feats) == {1, 2, 3, 4}
    for k in feats:
        assert torch.equal(feats[k], torch.stack([r[k] for r in ref]))
    for (h, c), (rh, rc) in zip(states, ref_states):
        assert torch.equal(h, rh) and torch.equal(c, rc)


def test_state_carries_across_clips(backbone):
    # the pipeline streams a recording as consecutive clips with the state carried: 2 + 2 frames == 4 frames
    x = torch.rand(4, 1, 20, 256, 320)
    with torch.no_grad():
        whole, _ = backbone(x, None, None, train_step=False)
        a, st = backbone(x[:2], None, None, train_step=False)
        b, _ = backbone(x[2:], st, None, train_step=False)
    for k in whole:
        assert torch.allclose(whole[k], torch.cat([a[k], b[k]]), atol=1e-6)


def test_register_routes_maxvit_to_the_lstm_backbone(cfg):
    import models.detection.recurrent_backbone as rb
    import models.detection.yolox_extension.models.detector as det
    from event_ssm.baselines.rvt_lstm import RVTLSTMBackbone, register_rvt_lstm
    saved = rb.build_recurrent_backbone, det.build_recurrent_backbone
    try:
        register_rvt_lstm()
        assert isinstance(det.build_recurrent_backbone(cfg.model.backbone), RVTLSTMBackbone)
    finally:
        rb.build_recurrent_backbone, det.build_recurrent_backbone = saved


@pytest.mark.skipif(not RVTB_CKPT.exists(), reason="RVT-B Gen1 checkpoint not downloaded")
def test_rvt_checkpoint_loads_strictly(backbone):
    sd = torch.load(RVTB_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    bb = {k[len("mdl.backbone."):]: v for k, v in sd.items() if k.startswith("mdl.backbone.")}
    backbone.load_state_dict(bb, strict=True)
