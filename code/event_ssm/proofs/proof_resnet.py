"""Visual proof for Unit 1: conv1 filters before/after avg-proj + feature heatmaps + shape table."""
import pathlib, torch, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from event_ssm.backbone.resnet_spatial import ResNetSpatialStages, _avg_projection_conv1
from torchvision.models import resnet18, ResNet18_Weights

OUT = pathlib.Path(__file__).parent / "out"; OUT.mkdir(parents=True, exist_ok=True)
rgb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).conv1
new = _avg_projection_conv1(rgb, 20)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(make_grid(rgb.weight.mean(1, keepdim=True), nrow=8, normalize=True)[0].cpu()); ax[0].set_title("ImageNet conv1 (RGB mean)")
ax[1].imshow(make_grid(new.weight[:, :1], nrow=8, normalize=True)[0].detach().cpu()); ax[1].set_title("avg-proj conv1 (20ch, ch0)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "u1_conv1_filters.png", dpi=150); plt.close()

m = ResNetSpatialStages(20, pretrained=True).eval()
f = m(torch.randn(1, 20, 256, 320))
fig, ax = plt.subplots(1, 4, figsize=(14, 3))
for i, k in enumerate((1, 2, 3, 4)):
    ax[i].imshow(f[k][0].mean(0).detach().cpu()); ax[i].set_title(f"stage{k} {tuple(f[k].shape[1:])}"); ax[i].axis("off")
plt.tight_layout(); plt.savefig(OUT / "u1_feature_heatmaps.png", dpi=150); plt.close()
print("shapes:", {k: tuple(v.shape) for k, v in f.items()})
print("params(M):", round(sum(p.numel() for p in m.parameters())/1e6, 3))
print("wrote u1_conv1_filters.png, u1_feature_heatmaps.png")
