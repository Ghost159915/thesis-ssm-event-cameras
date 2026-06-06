"""Visual proof for Unit 3: end-to-end shape trace + per-stage state presence."""
import torch
from event_ssm.backbone.resnet_mamba import ResNetMambaBackbone
m = ResNetMambaBackbone(in_channels=10, pretrained=False).cuda().eval()
feats, states = m(torch.randn(5, 2, 10, 256, 320, device="cuda"), prev_states=None)
print("=== forward shape trace (L=5,B=2) ===")
for k in (1,2,3,4): print(f" stage{k}: {tuple(feats[k].shape)}")
print(" stage_dims(2,3,4):", m.get_stage_dims((2,3,4)), "strides:", m.get_strides((2,3,4)))
print(" per-stage state present:", [s is not None for s in states])
