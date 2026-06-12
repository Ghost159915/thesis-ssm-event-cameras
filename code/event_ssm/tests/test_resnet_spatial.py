import torch, pytest
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages

def test_shapes(device):
    m = ResNetSpatialStages(in_channels=20, pretrained=False).to(device).eval()
    x = torch.randn(2, 20, 256, 320, device=device)  # padded Gen1 resolution, 20ch stacked hist
    f = m(x)
    assert f[1].shape == (2, 64, 64, 80)
    assert f[2].shape == (2, 128, 32, 40)
    assert f[3].shape == (2, 256, 16, 20)
    assert f[4].shape == (2, 512, 8, 10)

def test_gradients(device):
    m = ResNetSpatialStages(in_channels=20, pretrained=False).to(device).train()
    x = torch.randn(2, 20, 256, 320, device=device)
    sum(v.sum() for v in m(x).values()).backward()
    assert m.stem[0].weight.grad is not None       # conv1
    assert m.layer4[0].conv1.weight.grad is not None

def test_avg_projection_init(device):
    m = ResNetSpatialStages(in_channels=20, pretrained=True).to(device)
    w = m.stem[0].weight.data
    assert w.shape == (64, 20, 7, 7)
    assert 0.005 < w.std().item() < 0.2   # not random (~1.0) nor ~zero

def test_param_count():
    m = ResNetSpatialStages(in_channels=20, pretrained=False)
    mparams = sum(p.numel() for p in m.parameters()) / 1e6
    assert 11.0 < mparams < 11.5
