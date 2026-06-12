import torch
from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone

def test_forward_shapes_and_state(device):
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).eval()
    x = torch.randn(5, 2, 20, 256, 320, device=device)   # (L=time, B, C=20, H, W)
    feats, states = m(x, prev_states=None)
    assert feats[2].shape == (5, 2, 128, 32, 40)
    assert feats[3].shape == (5, 2, 256, 16, 20)
    assert feats[4].shape == (5, 2, 512, 8, 10)
    assert len(states) == 4
    assert m.get_stage_dims((2,3,4)) == (128,256,512)
    assert m.get_strides((2,3,4)) == (8,16,32)

def test_integration_with_pafpn_head(device):
    """Reused RVT PAFPN + YOLOX head accept our backbone's stage outputs."""
    from models.detection.yolox_extension.models.yolo_pafpn import YOLOPAFPN
    from models.detection.yolox.models.yolo_head import YOLOXHead
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).eval()
    x = torch.randn(1, 1, 20, 256, 320, device=device)
    feats, _ = m(x, prev_states=None)
    pafpn = YOLOPAFPN(depth=0.33, in_stages=(2,3,4), in_channels=(128,256,512)).to(device).eval()
    head = YOLOXHead(num_classes=2, strides=(8,16,32), in_channels=(128,256,512)).to(device).eval()
    outs, _ = head(pafpn({2:feats[2][0], 3:feats[3][0], 4:feats[4][0]}))  # [0]=timestep L0 -> (B,c,h,w)
    assert outs.shape[-1] == 7          # x,y,w,h,obj,cls0,cls1
    assert outs.shape[1] == 32*40 + 16*20 + 8*10   # 1680


def test_eval_state_rvt_compatible(device):
    """Eval states: None-free, batch is dim0, survive RVT recursive detach + per-seq reset,
    and can be carried back into the step path."""
    from modules.utils.detection import RNNStates
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).eval()
    B = 2
    x = torch.randn(3, B, 20, 256, 320, device=device)        # (L,B,C,H,W)
    feats, states = m(x, prev_states=None)
    assert feats[2].shape == (3, B, 128, 32, 40)
    conv_b, ssm_b = states[1][0]                              # stage1, layer0 -> (conv,ssm)
    assert conv_b.shape[0] == B and ssm_b.shape[0] == B       # batch is dim0
    detached = RNNStates.recursive_detach(states)             # must not raise on None
    RNNStates.recursive_reset(detached, indices_or_bool_tensor=[0])
    feats2, _ = m(x, prev_states=detached)                    # carry back into step path
    assert feats2[2].shape == (3, B, 128, 32, 40)


def test_train_state_placeholder_rvt_compatible(device):
    """Train states: None-free placeholder with batch dim0 (beta deferred -> zero-init per clip)."""
    from modules.utils.detection import RNNStates
    m = ResNetMambaBackbone(in_channels=20, pretrained=False).to(device).train()
    B = 2
    x = torch.randn(3, B, 20, 256, 320, device=device)
    feats, states = m(x, prev_states=None)
    assert feats[2].shape == (3, B, 128, 32, 40)
    assert states[1].shape[0] == B                           # (B,1) placeholder, dim0=B
    detached = RNNStates.recursive_detach(states)            # no None -> ok
    RNNStates.recursive_reset(detached, indices_or_bool_tensor=[0])
