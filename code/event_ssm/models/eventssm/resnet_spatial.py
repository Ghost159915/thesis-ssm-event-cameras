import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def _avg_projection_conv1(pretrained_conv1: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    """(64,3,7,7) -> (64,in_ch,7,7): mean over RGB, tile to in_ch, scale 3/in_ch."""
    new = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        w = pretrained_conv1.weight.data.mean(dim=1, keepdim=True)      # (64,1,7,7)
        new.weight.copy_(w.repeat(1, in_ch, 1, 1) * (3.0 / in_ch))      # (64,in_ch,7,7)
    return new


class ResNetSpatialStages(nn.Module):
    """ResNet-18 stem + 4 stages, 20-channel input (stacked histogram: 2 pol x 10 bins,
    matches baseline gen1 pipeline). Returns per-stage feature maps.
    Temporal Mamba is interleaved by ResNetMambaBackbone (Unit 3), not here."""
    stage_dims = (64, 128, 256, 512)
    strides = (4, 8, 16, 32)

    def __init__(self, in_channels: int = 20, pretrained: bool = True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.conv1 = (_avg_projection_conv1(net.conv1, in_channels) if pretrained
                     else nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False))
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return {1: c1, 2: c2, 3: c3, 4: c4}
