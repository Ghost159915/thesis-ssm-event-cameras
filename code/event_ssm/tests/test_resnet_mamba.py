import torch
import pytest
from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="mamba-ssm kernels are CUDA-only")


# ---------------------------------------------------------------------------
# Helper: construct the backbone used by most tests
# ---------------------------------------------------------------------------

def _bb():
    return ResNetMambaBackbone(in_channels=20, pretrained=False, d_state=64,
                               temporal_stages=(2, 3, 4)).cuda().float()


# ---------------------------------------------------------------------------
# Finding §8: temporal blocks only on FPN-consumed stages
# ---------------------------------------------------------------------------

def test_temporal_only_on_fpn_stages():
    bb = _bb()
    assert set(bb.temporal.keys()) == {"2", "3", "4"}        # no stage-1 temporal (Finding §8)


# ---------------------------------------------------------------------------
# Forward: shapes + unified state threading in both train and eval
# ---------------------------------------------------------------------------

def test_forward_shapes_and_states_train_and_eval():
    bb = _bb()
    x = torch.randn(3, 2, 20, 256, 320, device="cuda")        # (L,B,C,H,W)
    for mode in ("train", "eval"):
        getattr(bb, mode)()
        feats, states = bb(x, None)
        assert set(feats.keys()) == {1, 2, 3, 4}
        assert feats[2].shape[0] == 3 and feats[2].shape[1] == 2   # (L,B,c,h,w)
        assert len(states) == 4
        for s in (2, 3, 4):                                   # temporal stages carry a real state in BOTH modes
            assert isinstance(states[s - 1], list) and len(states[s - 1]) == 1


# ---------------------------------------------------------------------------
# State carry: cross-clip memory must genuinely affect output
# ---------------------------------------------------------------------------

def test_state_carry_changes_output():
    bb = _bb().eval()
    x = torch.randn(3, 2, 20, 256, 320, device="cuda")
    with torch.no_grad():
        f0, st = bb(x, None)
        f1, _ = bb(x, st)                                     # carried memory must change stage-4 output
    assert (f0[4] - f1[4]).abs().max().item() > 1e-5


# ---------------------------------------------------------------------------
# Structural: dims/strides API unchanged
# ---------------------------------------------------------------------------

def test_forward_shapes_and_state(device):
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).eval()
    x = torch.randn(5, 2, 20, 256, 320, device=device)   # (L=time, B, C=20, H, W)
    feats, states = m(x, prev_states=None)
    assert feats[2].shape == (5, 2, 128, 32, 40)
    assert feats[3].shape == (5, 2, 256, 16, 20)
    assert feats[4].shape == (5, 2, 512, 8, 10)
    assert len(states) == 4
    assert m.get_stage_dims((2, 3, 4)) == (128, 256, 512)
    assert m.get_strides((2, 3, 4)) == (8, 16, 32)


def test_integration_with_pafpn_head(device):
    """Reused RVT PAFPN + YOLOX head accept our backbone's stage outputs."""
    from models.detection.yolox_extension.models.yolo_pafpn import YOLOPAFPN
    from models.detection.yolox.models.yolo_head import YOLOXHead
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).eval()
    x = torch.randn(1, 1, 20, 256, 320, device=device)
    feats, _ = m(x, prev_states=None)
    pafpn = YOLOPAFPN(depth=0.33, in_stages=(2, 3, 4), in_channels=(128, 256, 512)).to(device).eval()
    head = YOLOXHead(num_classes=2, strides=(8, 16, 32), in_channels=(128, 256, 512)).to(device).eval()
    outs, _ = head(pafpn({2: feats[2][0], 3: feats[3][0], 4: feats[4][0]}))  # [0]=timestep L0 -> (B,c,h,w)
    assert outs.shape[-1] == 7          # x,y,w,h,obj,cls0,cls1
    assert outs.shape[1] == 32 * 40 + 16 * 20 + 8 * 10   # 1680


def test_eval_state_rvt_compatible(device):
    """Eval states: None-free, batch is dim0, survive RVT recursive detach + per-seq reset,
    and can be carried back into the step path."""
    from modules.utils.detection import RNNStates
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).eval()
    B = 2
    x = torch.randn(3, B, 20, 256, 320, device=device)        # (L,B,C,H,W)
    feats, states = m(x, prev_states=None)
    assert feats[2].shape == (3, B, 128, 32, 40)
    conv_b, ssm_b = states[1][0]                              # stage2 (index1), layer0 -> (conv,ssm)
    assert conv_b.shape[0] == B and ssm_b.shape[0] == B       # batch is dim0
    detached = RNNStates.recursive_detach(states)             # must not raise on None
    RNNStates.recursive_reset(detached, indices_or_bool_tensor=[0])
    feats2, _ = m(x, prev_states=detached)                    # carry back into step path
    assert feats2[2].shape == (3, B, 128, 32, 40)


def test_train_state_placeholder_rvt_compatible(device):
    """Train: stage-1 (non-temporal) gets a None-free (B,1) placeholder with batch dim0.
    Temporal stages return real state lists in BOTH train and eval (unified threading)."""
    from modules.utils.detection import RNNStates
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).train()
    B = 2
    x = torch.randn(3, B, 20, 256, 320, device=device)
    feats, states = m(x, prev_states=None)
    assert feats[2].shape == (3, B, 128, 32, 40)
    # stage-1 (index 0) is non-temporal -> placeholder tensor (B,1), dim0=B
    assert states[0].shape[0] == B
    # stage-2/3/4 (indices 1/2/3) are temporal -> real state lists in train too (unified)
    assert isinstance(states[1], list) and len(states[1]) == 1
    detached = RNNStates.recursive_detach(states)            # no None -> ok
    RNNStates.recursive_reset(detached, indices_or_bool_tensor=[0])
    # the real Lightning loop resets with a length-B BOOL tensor (is_first_sample), not a list:
    RNNStates.recursive_reset(detached, indices_or_bool_tensor=torch.tensor([True, False], device=device))


def test_state_helpers_roundtrip_numeric(device):
    """_state_from_bmajor(_state_to_bmajor(.)) is an EXACT identity and preserves per-sample
    grouping. A transposed reshape (e.g. reshape(hw, B, ...)) would pass every shape assert yet
    silently corrupt cross-clip state -- this catches that."""
    from event_ssm.backbone.resnet_mamba import _state_to_bmajor, _state_from_bmajor
    B, hw, di, dc, ds = 2, 4, 3, 5, 7
    N = B * hw
    conv = torch.arange(N * di * dc, dtype=torch.float32, device=device).reshape(N, di, dc)
    ssm = torch.arange(N * di * ds, dtype=torch.float32, device=device).reshape(N, di, ds) + 1000.
    st_N = [(conv, ssm)]                                      # one temporal layer, kernels' (N,...) layout
    b = _state_to_bmajor(st_N, B, hw)
    back = _state_from_bmajor(b, B, hw)
    assert torch.equal(back[0][0], conv) and torch.equal(back[0][1], ssm)   # exact round-trip
    # b-major row [s] must be exactly sample s's contiguous block of hw rows (B is slowest in fold's (B H W)):
    assert torch.equal(b[0][0][0], conv[0:hw]) and torch.equal(b[0][0][1], conv[hw:2 * hw])


def test_eval_carried_state_changes_output(device):
    """Carrying eval state from the previous clip must actually change this clip's features --
    proves the cross-clip carry is wired, not silently dropped."""
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).eval()
    B = 2
    x = torch.randn(3, B, 20, 256, 320, device=device)
    feats0, st = m(x, prev_states=None)                       # fresh (zero) state
    feats1, _ = m(x, prev_states=st)                          # carry prev-clip state
    assert not torch.allclose(feats0[2], feats1[2])           # carry genuinely affects output


def test_recursive_reset_bool_mask_per_sequence(device):
    """Lightning's per-sequence reset uses a length-B BOOL tensor; it must zero exactly the
    flagged sequence's dim0 slice and leave the others intact, then still carry into the step path."""
    from modules.utils.detection import RNNStates
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).eval()
    B = 2
    x = torch.randn(3, B, 20, 256, 320, device=device)
    _, states = m(x, prev_states=None)
    detached = RNNStates.recursive_detach(states)
    mask = torch.tensor([True, False], device=device)        # reset only sequence 0
    RNNStates.recursive_reset(detached, indices_or_bool_tensor=mask)
    conv_b, _ = detached[1][0]                                # stage2 (index1), layer0 b-major conv (B, hw, ...)
    assert torch.count_nonzero(conv_b[0]) == 0               # seq0 zeroed
    assert torch.count_nonzero(conv_b[1]) > 0                # seq1 preserved
    feats2, _ = m(x, prev_states=detached)                   # still valid for the step path
    assert feats2[2].shape == (3, B, 128, 32, 40)
