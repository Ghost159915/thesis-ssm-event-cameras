"""Visual proof for Unit 3: end-to-end shape trace + per-stage state presence.
Per-stage feature is (L, B, c, h, w) (RVT indexes v[tidx]); states are dim0=B, None-free."""
import torch
from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
m = ResNetMambaBackbone(in_channels=20, pretrained=False).cuda().eval()
feats, states = m(torch.randn(5, 2, 20, 256, 320, device="cuda"), prev_states=None)
print("=== forward shape trace (L=5,B=2) ===")
for k in (1,2,3,4): print(f" stage{k}: {tuple(feats[k].shape)}")
print(" stage_dims(2,3,4):", m.get_stage_dims((2,3,4)), "strides:", m.get_strides((2,3,4)))
print(" per-stage state present:", [s is not None for s in states])
