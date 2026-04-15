"""
EventSSMDetector: Full detection model.

Architecture overview:

  Input: voxel grid (B_batch, num_bins, H, W)
            │
            ▼
  ┌─────────────────────┐
  │  Spatial Backbone   │  ResNet-18 (modified first layer for num_bins channels)
  │  per temporal bin   │  Shared weights across bins → feature map per bin
  └─────────────────────┘
            │  (B_batch, num_bins, C, h, w)
            ▼
  ┌─────────────────────┐
  │  Spatial Pooling    │  Global average pool → (B_batch, num_bins, C)
  └─────────────────────┘
            │
            ▼
  ┌─────────────────────┐
  │  Mamba Temporal     │  Processes sequence of bin features
  │  Module             │  Output: (B_batch, num_bins, d_model)
  └─────────────────────┘
            │  Take last token (most temporally recent)
            ▼
  ┌─────────────────────┐
  │  Detection Head     │  FC layers → objectness + class + box regression
  └─────────────────────┘

Note on the detection head: we use a simple grid-based anchor-free approach.
The backbone produces feature maps at stride 8; each cell predicts one box.
This is similar in spirit to FCOS but simplified for the MVP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from einops import rearrange

from .mamba import MambaStack


class LightweightBackbone(nn.Module):
    """Stripped-down ResNet-18 backbone adapted for event voxel grids.

    Key changes from standard ResNet-18:
      1. First conv takes num_bins input channels instead of 3 (RGB).
      2. We extract features after layer2 only (stride-8) for speed.
         This gives a 38×48 feature map for 240×304 input.
      3. No pretrained weights (input channels differ from ImageNet).
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        base = resnet18(weights=None)

        # Replace first conv: 3-channel RGB → num_bins event bins
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2  # output: 128 channels, stride-8

        # Project to desired output channels
        self.proj = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            Feature map of shape (B, out_channels, H/8, W/8)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.proj(x)
        return x


class DetectionHead(nn.Module):
    """Anchor-free detection head.

    Operates on a (B, d_model) feature vector (the aggregated temporal feature)
    but also takes the spatial feature map for spatially-aware predictions.

    For each cell in the feature grid, predicts:
      - 1 objectness score (is there an object?)
      - num_classes class scores
      - 4 box regression values (dx, dy, dw, dh relative to cell)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feat_h: int,
        feat_w: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_h = feat_h
        self.feat_w = feat_w

        # Shared convolutional head
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Output branches
        self.cls_head = nn.Conv2d(128, num_classes, 1)    # class logits
        self.obj_head = nn.Conv2d(128, 1, 1)              # objectness logit
        self.box_head = nn.Conv2d(128, 4, 1)              # box regression

    def forward(self, feat_map: torch.Tensor):
        """
        Args:
            feat_map: Spatial feature map (B, C, H_feat, W_feat)

        Returns:
            cls_pred: (B, num_classes, H_feat, W_feat)
            obj_pred: (B, 1, H_feat, W_feat)
            box_pred: (B, 4, H_feat, W_feat) — (dx, dy, log_dw, log_dh)
        """
        feat = self.shared(feat_map)
        cls_pred = self.cls_head(feat)
        obj_pred = self.obj_head(feat)
        box_pred = self.box_head(feat)
        return cls_pred, obj_pred, box_pred


class EventSSMDetector(nn.Module):
    """Full event camera object detector using Mamba temporal processing.

    This is the thesis's core model. It:
      1. Extracts per-bin spatial features from the voxel grid using a shared CNN.
      2. Processes the sequence of per-bin features with Mamba blocks.
      3. Uses the temporally-processed features to produce detection outputs.
    """

    def __init__(
        self,
        num_bins: int = 10,
        height: int = 240,
        width: int = 304,
        backbone_out: int = 256,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_layers: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.height = height
        self.width = width
        self.d_model = d_model

        # Feature map spatial dimensions after stride-8 backbone
        self.feat_h = height // 8
        self.feat_w = width // 8

        # --- Spatial backbone (shared across bins) ---
        # We process all bins at once by treating them as a batch dim
        self.backbone = LightweightBackbone(
            in_channels=1,        # process one bin at a time
            out_channels=backbone_out,
        )

        # --- Mamba temporal module ---
        # Input is a sequence of spatially-pooled bin features
        self.mamba = MambaStack(
            d_model=d_model,
            num_layers=mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
        )

        # Project pooled spatial features → d_model for Mamba input
        self.feat_to_seq = nn.Linear(backbone_out, d_model)

        # Upsample Mamba output back to spatial map for detection
        # We fuse the last-bin backbone feature with the Mamba output
        self.fuse = nn.Sequential(
            nn.Conv2d(backbone_out + d_model, backbone_out, 1, bias=False),
            nn.BatchNorm2d(backbone_out),
            nn.ReLU(inplace=True),
        )

        # --- Detection head ---
        self.head = DetectionHead(
            in_channels=backbone_out,
            num_classes=num_classes,
            feat_h=self.feat_h,
            feat_w=self.feat_w,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, voxel: torch.Tensor):
        """
        Args:
            voxel: (B, num_bins, H, W) — voxel grid from event camera

        Returns:
            cls_pred: (B, num_classes, H_feat, W_feat)
            obj_pred: (B, 1, H_feat, W_feat)
            box_pred: (B, 4, H_feat, W_feat)
        """
        B, T, H, W = voxel.shape

        # --- Step 1: Extract spatial features for every bin independently ---
        # Reshape to (B*T, 1, H, W) so backbone processes all bins in one forward pass
        voxel_flat = rearrange(voxel, "b t h w -> (b t) 1 h w")
        feat_flat = self.backbone(voxel_flat)                   # (B*T, C, h, w)
        C, fh, fw = feat_flat.shape[1], feat_flat.shape[2], feat_flat.shape[3]

        # --- Step 2: Pool spatial features → sequence tokens for Mamba ---
        # Global average pool: (B*T, C, h, w) → (B*T, C)
        feat_pooled = feat_flat.mean(dim=[-2, -1])
        feat_seq = rearrange(feat_pooled, "(b t) c -> b t c", b=B, t=T)
        feat_seq = self.feat_to_seq(feat_seq)                   # (B, T, d_model)

        # --- Step 3: Temporal processing with Mamba ---
        mamba_out = self.mamba(feat_seq)                        # (B, T, d_model)
        # Take the last token — represents the full temporal context
        temporal_feat = mamba_out[:, -1, :]                     # (B, d_model)

        # --- Step 4: Fuse temporal context back to spatial feature map ---
        # Use the last bin's spatial features as the spatial anchor
        last_bin_feat = rearrange(
            feat_flat, "(b t) c h w -> b t c h w", b=B, t=T
        )[:, -1]                                                 # (B, C, h, w)

        # Broadcast temporal feature to spatial dims and concatenate
        temporal_spatial = temporal_feat[:, :, None, None].expand(
            B, self.d_model, fh, fw
        )
        fused = torch.cat([last_bin_feat, temporal_spatial], dim=1)
        fused = self.fuse(fused)                                # (B, C, h, w)

        # --- Step 5: Detection predictions ---
        cls_pred, obj_pred, box_pred = self.head(fused)

        return cls_pred, obj_pred, box_pred

    def decode_predictions(
        self,
        cls_pred: torch.Tensor,
        obj_pred: torch.Tensor,
        box_pred: torch.Tensor,
        conf_threshold: float = 0.3,
        stride: int = 8,
    ):
        """Decode raw predictions into bounding boxes for inference.

        Args:
            cls_pred: (B, num_classes, H_feat, W_feat) — class logits
            obj_pred: (B, 1, H_feat, W_feat) — objectness logit
            box_pred: (B, 4, H_feat, W_feat) — (dx, dy, log_dw, log_dh)
            conf_threshold: Minimum objectness * class score to keep.
            stride: Backbone downsampling stride (8 for our backbone).

        Returns:
            List (one per image) of dicts with:
                'boxes': (N, 4) tensor in [x1, y1, x2, y2] pixel coords
                'scores': (N,) confidence scores
                'labels': (N,) predicted class indices
        """
        B, num_cls, fh, fw = cls_pred.shape
        device = cls_pred.device

        obj_score = torch.sigmoid(obj_pred)        # (B, 1, fh, fw)
        cls_score = torch.sigmoid(cls_pred)        # (B, num_cls, fh, fw)

        # Build cell centre grid
        gy, gx = torch.meshgrid(
            torch.arange(fh, device=device, dtype=torch.float32),
            torch.arange(fw, device=device, dtype=torch.float32),
            indexing="ij",
        )  # both (fh, fw)

        results = []
        for b in range(B):
            obj = obj_score[b, 0]           # (fh, fw)
            cls = cls_score[b]              # (num_cls, fh, fw)
            box = box_pred[b]               # (4, fh, fw)

            # Best class per cell
            best_cls_score, best_cls_idx = cls.max(dim=0)  # (fh, fw)
            conf = obj * best_cls_score                     # (fh, fw)

            # Filter by threshold
            mask = conf > conf_threshold
            if not mask.any():
                results.append({
                    "boxes": torch.zeros(0, 4, device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                })
                continue

            # Decode boxes: cell-relative offsets → absolute pixel coords
            cx = (gx[mask] + box[0][mask].sigmoid()) * stride
            cy = (gy[mask] + box[1][mask].sigmoid()) * stride
            bw = box[2][mask].exp() * stride
            bh = box[3][mask].exp() * stride

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            scores = conf[mask]
            labels = best_cls_idx[mask]

            results.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })

        return results
