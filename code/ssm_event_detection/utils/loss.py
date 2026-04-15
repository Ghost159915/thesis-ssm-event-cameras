"""
Detection loss functions.

We use a combination of:
  - Focal loss for classification (handles class imbalance)
  - Binary cross-entropy for objectness
  - IoU-based regression loss for boxes

Target assignment: for each ground-truth box, find the feature map cells
whose centre falls inside the box and assign them as positive samples.
This is the FCOS-style assignment strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss (Lin et al. 2017) for handling class imbalance.

    Args:
        pred: Logits of shape (N,) or (N, C)
        target: Binary targets of same shape as pred
        alpha: Weighting factor for positive samples
        gamma: Focusing exponent
        reduction: 'mean', 'sum', or 'none'
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p = torch.sigmoid(pred)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    loss = focal_weight * bce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    mode: str = "giou",
) -> torch.Tensor:
    """IoU-based bounding box regression loss.

    Args:
        pred_boxes: (N, 4) in [x1, y1, x2, y2]
        target_boxes: (N, 4) in [x1, y1, x2, y2]
        mode: 'iou' or 'giou'

    Returns:
        Scalar loss.
    """
    # Intersection
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (
        (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0)
        * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    )
    target_area = (
        (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0)
        * (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)
    )

    union_area = pred_area + target_area - inter_area + 1e-6
    iou = inter_area / union_area

    if mode == "iou":
        return (1 - iou).mean()

    # GIoU: add penalty for enclosing box area
    enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enc_area = (
        (enc_x2 - enc_x1).clamp(min=0)
        * (enc_y2 - enc_y1).clamp(min=0)
    ) + 1e-6

    giou = iou - (enc_area - union_area) / enc_area
    return (1 - giou).mean()


class DetectionLoss(nn.Module):
    """Combined detection loss with FCOS-style target assignment."""

    def __init__(
        self,
        num_classes: int = 2,
        stride: int = 8,
        cls_weight: float = 1.0,
        box_weight: float = 5.0,
        obj_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.obj_weight = obj_weight

    def forward(
        self,
        cls_pred: torch.Tensor,   # (B, num_classes, H_feat, W_feat)
        obj_pred: torch.Tensor,   # (B, 1, H_feat, W_feat)
        box_pred: torch.Tensor,   # (B, 4, H_feat, W_feat)
        targets: List[Dict],      # list of {'boxes': (M,4), 'labels': (M,)}
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with 'loss', 'loss_cls', 'loss_obj', 'loss_box'.
        """
        B, _, fh, fw = cls_pred.shape
        device = cls_pred.device

        # Build cell centre grid in pixel coords
        gy, gx = torch.meshgrid(
            torch.arange(fh, device=device, dtype=torch.float32),
            torch.arange(fw, device=device, dtype=torch.float32),
            indexing="ij",
        )
        cell_cx = (gx + 0.5) * self.stride   # (fh, fw)
        cell_cy = (gy + 0.5) * self.stride

        # Flatten spatial dims for easier indexing
        cls_flat = cls_pred.permute(0, 2, 3, 1).reshape(B, fh * fw, self.num_classes)
        obj_flat = obj_pred.permute(0, 2, 3, 1).reshape(B, fh * fw)
        box_flat = box_pred.permute(0, 2, 3, 1).reshape(B, fh * fw, 4)
        cx_flat = cell_cx.reshape(-1)
        cy_flat = cell_cy.reshape(-1)

        all_cls_loss = []
        all_obj_loss = []
        all_box_loss = []

        for b in range(B):
            gt_boxes = targets[b]["boxes"].to(device)   # (M, 4) x1y1x2y2
            gt_labels = targets[b]["labels"].to(device)  # (M,)
            M = len(gt_boxes)

            # Objectness targets: start all negative
            obj_target = torch.zeros(fh * fw, device=device)

            if M == 0:
                # No objects in this sample — pure background loss
                all_obj_loss.append(
                    F.binary_cross_entropy_with_logits(
                        obj_flat[b], obj_target, reduction="mean"
                    )
                )
                all_cls_loss.append(torch.tensor(0.0, device=device))
                all_box_loss.append(torch.tensor(0.0, device=device))
                continue

            # FCOS-style assignment: a cell is positive if its centre is
            # inside any ground-truth box
            pos_indices = []
            pos_gt_idx = []

            for m in range(M):
                x1, y1, x2, y2 = gt_boxes[m]
                inside = (
                    (cx_flat >= x1) & (cx_flat <= x2)
                    & (cy_flat >= y1) & (cy_flat <= y2)
                )
                inside_idx = inside.nonzero(as_tuple=True)[0]
                if len(inside_idx) > 0:
                    pos_indices.append(inside_idx)
                    pos_gt_idx.extend([m] * len(inside_idx))

            if len(pos_indices) == 0:
                # GT boxes too small — fall back to nearest cell
                for m in range(M):
                    cx_gt = (gt_boxes[m, 0] + gt_boxes[m, 2]) / 2
                    cy_gt = (gt_boxes[m, 1] + gt_boxes[m, 3]) / 2
                    dist = (cx_flat - cx_gt) ** 2 + (cy_flat - cy_gt) ** 2
                    nearest = dist.argmin().unsqueeze(0)
                    pos_indices.append(nearest)
                    pos_gt_idx.append(m)

            pos_idx = torch.cat(pos_indices)                # (P,)
            pos_gt = torch.tensor(pos_gt_idx, device=device)

            obj_target[pos_idx] = 1.0
            all_obj_loss.append(
                focal_loss(obj_flat[b], obj_target, reduction="mean")
            )

            # --- Classification loss (positive cells only) ---
            cls_target = torch.zeros(len(pos_idx), self.num_classes, device=device)
            cls_target[torch.arange(len(pos_idx)), gt_labels[pos_gt]] = 1.0
            all_cls_loss.append(
                focal_loss(cls_flat[b][pos_idx], cls_target, reduction="mean")
            )

            # --- Box regression loss (positive cells only) ---
            # Decode predicted boxes at positive cells
            bx = box_flat[b][pos_idx]       # (P, 4): (dx, dy, log_dw, log_dh)
            pred_cx = (cx_flat[pos_idx] / self.stride + bx[:, 0].sigmoid()) * self.stride
            pred_cy = (cy_flat[pos_idx] / self.stride + bx[:, 1].sigmoid()) * self.stride
            pred_w = bx[:, 2].exp() * self.stride
            pred_h = bx[:, 3].exp() * self.stride
            pred_x1 = pred_cx - pred_w / 2
            pred_y1 = pred_cy - pred_h / 2
            pred_x2 = pred_cx + pred_w / 2
            pred_y2 = pred_cy + pred_h / 2
            pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

            target_boxes = gt_boxes[pos_gt]
            all_box_loss.append(iou_loss(pred_boxes, target_boxes, mode="giou"))

        loss_cls = torch.stack(all_cls_loss).mean()
        loss_obj = torch.stack(all_obj_loss).mean()
        loss_box = torch.stack(all_box_loss).mean()

        total = (
            self.cls_weight * loss_cls
            + self.obj_weight * loss_obj
            + self.box_weight * loss_box
        )

        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_obj": loss_obj,
            "loss_box": loss_box,
        }
