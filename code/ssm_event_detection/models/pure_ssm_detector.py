"""
PureSSMDetector: Event camera object detector with no CNN.

Architecture overview:

  Input: voxel grid (B, T, H, W)
            │
            ▼
  ┌─────────────────────────┐
  │  Patch Embedding        │  Divide each bin into P×P patches → linear projection
  │  (no convolution)       │  Output: (B, T, num_patches, d_model)
  └─────────────────────────┘
            │
            ▼
  ┌─────────────────────────┐
  │  Spatial BiMamba        │  Bidirectional Mamba across patch tokens within each bin
  │  (per bin, shared)      │  Learns spatial relationships without convolution
  └─────────────────────────┘
            │  Pool patch tokens → one vector per bin
            ▼
  ┌─────────────────────────┐
  │  Temporal Mamba         │  Causal Mamba across T bin summaries
  └─────────────────────────┘
            │
            ▼
  ┌─────────────────────────┐
  │  Detection Head         │  Reshape patch tokens back to 2D grid → predict boxes
  └─────────────────────────┘

Key differences from CNN-SSM hybrid:
  - No convolutional layers anywhere
  - Spatial structure learned by Mamba, not hard-coded by convolution kernels
  - Bidirectional spatial scan (space is non-causal; time is causal)
  - Patch size controls the speed/accuracy tradeoff directly

This is the thesis's primary novel contribution — demonstrating that SSMs
can handle both spatial and temporal processing of event camera data without
any CNN components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .mamba import MambaBlock, MambaStack


class PatchEmbedding(nn.Module):
    """Convert each voxel bin into a sequence of patch tokens.

    Divides a (H, W) bin into non-overlapping patches of size patch_size×patch_size,
    then linearly projects each flattened patch to d_model dimensions.

    This is the only learned transformation before Mamba — one matrix multiply,
    no convolution, no local neighbourhood assumptions.

    Args:
        patch_size: Size of each square patch (8 or 16 recommended).
            Smaller = more spatial detail preserved, longer sequence, slower.
            Larger = coarser spatial detail, shorter sequence, faster.
        d_model: Output embedding dimension.
    """

    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        patch_dim = patch_size * patch_size   # values per patch (single bin channel)
        self.proj = nn.Linear(patch_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Single bin tensor of shape (B, H, W)

        Returns:
            tokens: (B, num_patches, d_model)
            grid_h: number of patches along height dimension
            grid_w: number of patches along width dimension
        """
        B, H, W = x.shape
        P = self.patch_size

        assert H % P == 0 and W % P == 0, (
            f"Image dimensions ({H}×{W}) must be divisible by patch_size ({P}). "
            f"Consider padding or adjusting patch_size."
        )

        grid_h = H // P
        grid_w = W // P

        # Rearrange into patches: (B, grid_h, grid_w, P, P)
        # then flatten each patch: (B, num_patches, P*P)
        patches = rearrange(x, "b (gh ph) (gw pw) -> b (gh gw) (ph pw)",
                            gh=grid_h, gw=grid_w, ph=P, pw=P)

        tokens = self.proj(patches)    # (B, num_patches, d_model)
        tokens = self.norm(tokens)
        return tokens, grid_h, grid_w


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block for spatial processing.

    Runs one Mamba forward (left→right) and one backward (right→left),
    then combines the two outputs. This is important for spatial tokens
    because spatial context flows in both directions — unlike time,
    which only flows forward.

    The combination strategy is learned addition via a gating parameter,
    similar to Vision Mamba (Zhu et al., ICML 2024).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.forward_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.backward_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        # Learned gate to blend forward and backward outputs
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) — sequence of spatial patch tokens

        Returns:
            (B, L, d_model)
        """
        # Forward pass: left to right
        out_fwd = self.forward_mamba(x)

        # Backward pass: flip sequence, run Mamba, flip back
        x_flipped = x.flip(dims=[1])
        out_bwd = self.backward_mamba(x_flipped).flip(dims=[1])

        # Blend forward and backward with learned gate
        gate = torch.sigmoid(self.gate)
        return gate * out_fwd + (1 - gate) * out_bwd


class SpatialMambaStack(nn.Module):
    """Stack of bidirectional Mamba blocks for spatial token processing."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_patches, d_model)

        Returns:
            (B, num_patches, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class PureSSMDetectionHead(nn.Module):
    """Detection head that operates on a 2D grid of patch tokens.

    Since we have no CNN, the spatial feature map comes directly from
    reshaping the processed patch tokens back to a 2D grid.
    Each grid cell (patch) independently predicts objectness, class, and box.
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        # Small MLP head per patch position (implemented as 1×1 convs)
        self.shared = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model // 2, 1),
            nn.GELU(),
        )
        self.cls_head = nn.Conv2d(d_model // 2, num_classes, 1)
        self.obj_head = nn.Conv2d(d_model // 2, 1, 1)
        self.box_head = nn.Conv2d(d_model // 2, 4, 1)

    def forward(self, feat_map: torch.Tensor):
        """
        Args:
            feat_map: (B, d_model, grid_h, grid_w) — patch tokens reshaped to 2D

        Returns:
            cls_pred: (B, num_classes, grid_h, grid_w)
            obj_pred: (B, 1, grid_h, grid_w)
            box_pred: (B, 4, grid_h, grid_w)
        """
        feat = self.shared(feat_map)
        return self.cls_head(feat), self.obj_head(feat), self.box_head(feat)


class PureSSMDetector(nn.Module):
    """Pure SSM event camera detector — no CNN layers.

    This is the thesis's primary novel model. It demonstrates that
    Mamba can replace CNNs for spatial feature extraction from event
    voxel grids, while also handling temporal processing — removing
    the CNN entirely from the event camera detection pipeline.

    The detection grid resolution is determined by patch_size:
        grid_h = height // patch_size
        grid_w = width  // patch_size

    For Gen1 (240×304) with patch_size=16: 15×19 = 285 patches per bin.
    For Gen1 (240×304) with patch_size=8:  30×38 = 1140 patches per bin.
    """

    def __init__(
        self,
        num_bins: int = 10,
        height: int = 240,
        width: int = 304,
        patch_size: int = 16,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        spatial_layers: int = 2,
        temporal_layers: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.d_model = d_model

        # Derived grid dimensions
        assert height % patch_size == 0 and width % patch_size == 0, (
            f"Height ({height}) and width ({width}) must be divisible by "
            f"patch_size ({patch_size})."
        )
        self.grid_h = height // patch_size
        self.grid_w = width // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # --- Patch embedding (shared across bins) ---
        self.patch_embed = PatchEmbedding(patch_size, d_model)

        # --- Spatial BiMamba (shared across bins) ---
        # Shared weights mean the spatial model is the same for every bin,
        # which drastically reduces parameters and encourages generalisation.
        self.spatial_mamba = SpatialMambaStack(
            d_model=d_model,
            num_layers=spatial_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # --- Temporal Mamba (causal — time flows forward only) ---
        self.temporal_mamba = MambaStack(
            d_model=d_model,
            num_layers=temporal_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Project pooled spatial features → temporal token
        # (average-pooled patch tokens are already d_model, no projection needed)

        # Fuse temporal context (last temporal token) back into the
        # spatial patch tokens of the final bin
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # --- Detection head ---
        self.head = PureSSMDetectionHead(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, voxel: torch.Tensor):
        """
        Args:
            voxel: (B, T, H, W) — voxel grid from event camera

        Returns:
            cls_pred: (B, num_classes, grid_h, grid_w)
            obj_pred: (B, 1, grid_h, grid_w)
            box_pred: (B, 4, grid_h, grid_w)
        """
        B, T, H, W = voxel.shape

        # === Stage 1: Patch Embedding ===
        # Process all bins at once by merging B and T into batch dim
        voxel_flat = rearrange(voxel, "b t h w -> (b t) h w")
        tokens_flat, gh, gw = self.patch_embed(voxel_flat)
        # tokens_flat: (B*T, num_patches, d_model)

        # === Stage 2: Spatial BiMamba (per bin, shared weights) ===
        # Each bin's patch tokens attend to each other spatially
        spatial_out = self.spatial_mamba(tokens_flat)
        # spatial_out: (B*T, num_patches, d_model)

        # Pool patch tokens → one summary vector per bin
        bin_summary = spatial_out.mean(dim=1)              # (B*T, d_model)
        bin_seq = rearrange(bin_summary, "(b t) d -> b t d", b=B, t=T)
        # bin_seq: (B, T, d_model)

        # === Stage 3: Temporal Mamba (causal, across bins) ===
        temporal_out = self.temporal_mamba(bin_seq)        # (B, T, d_model)
        temporal_ctx = temporal_out[:, -1, :]              # (B, d_model) — last bin

        # === Stage 4: Fuse temporal context back to spatial tokens ===
        # Take the last bin's spatial tokens (most recent events)
        last_bin_tokens = rearrange(
            spatial_out, "(b t) n d -> b t n d", b=B, t=T
        )[:, -1]                                           # (B, num_patches, d_model)

        # Broadcast temporal context to all patch positions and fuse
        temporal_expanded = temporal_ctx.unsqueeze(1).expand(
            B, self.num_patches, self.d_model
        )
        fused = self.fuse(
            torch.cat([last_bin_tokens, temporal_expanded], dim=-1)
        )                                                  # (B, num_patches, d_model)

        # === Stage 5: Reshape patches back to 2D grid for detection ===
        feat_map = rearrange(
            fused, "b (gh gw) d -> b d gh gw", gh=gh, gw=gw
        )                                                  # (B, d_model, grid_h, grid_w)

        # === Stage 6: Detection head ===
        cls_pred, obj_pred, box_pred = self.head(feat_map)
        return cls_pred, obj_pred, box_pred

    def decode_predictions(
        self,
        cls_pred: torch.Tensor,
        obj_pred: torch.Tensor,
        box_pred: torch.Tensor,
        conf_threshold: float = 0.3,
    ):
        """Decode raw predictions into bounding boxes.

        Same interface as EventSSMDetector.decode_predictions() so the
        training loop works identically for both models.

        The stride here is patch_size (not 8 like the CNN backbone)
        because each grid cell corresponds to one patch.
        """
        stride = self.patch_size
        B, num_cls, fh, fw = cls_pred.shape
        device = cls_pred.device

        obj_score = torch.sigmoid(obj_pred)
        cls_score = torch.sigmoid(cls_pred)

        gy, gx = torch.meshgrid(
            torch.arange(fh, device=device, dtype=torch.float32),
            torch.arange(fw, device=device, dtype=torch.float32),
            indexing="ij",
        )

        results = []
        for b in range(B):
            obj = obj_score[b, 0]
            cls = cls_score[b]
            box = box_pred[b]

            best_cls_score, best_cls_idx = cls.max(dim=0)
            conf = obj * best_cls_score
            mask = conf > conf_threshold

            if not mask.any():
                results.append({
                    "boxes": torch.zeros(0, 4, device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                })
                continue

            cx = (gx[mask] + box[0][mask].sigmoid()) * stride
            cy = (gy[mask] + box[1][mask].sigmoid()) * stride
            bw = box[2][mask].exp() * stride
            bh = box[3][mask].exp() * stride

            boxes = torch.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], dim=-1)
            results.append({
                "boxes": boxes,
                "scores": conf[mask],
                "labels": best_cls_idx[mask],
            })

        return results
