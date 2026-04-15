"""
Evaluation metrics: mAP@0.5 for event camera object detection.

We implement a straightforward per-class AP calculation using the
11-point interpolation method (Pascal VOC style) — sufficient for
comparing against baselines in the thesis.
"""

import torch
import numpy as np
from typing import List, Dict


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes.

    Args:
        boxes_a: (N, 4) in [x1, y1, x2, y2]
        boxes_b: (M, 4) in [x1, y1, x2, y2]

    Returns:
        IoU matrix of shape (N, M)
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = area_a[:, None] + area_b[None, :] - inter_area + 1e-6
    return inter_area / union_area


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """Non-maximum suppression.

    Args:
        boxes: (N, 4) in [x1, y1, x2, y2]
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of kept boxes.
    """
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[i].unsqueeze(0), boxes[rest])[0]
        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


class MeanAveragePrecision:
    """Accumulates predictions and computes mAP@0.5.

    Usage:
        metric = MeanAveragePrecision(num_classes=2)
        for batch in dataloader:
            ...
            preds = model.decode_predictions(...)
            metric.update(preds, targets)
        results = metric.compute()
        print(results['mAP'])
    """

    def __init__(self, num_classes: int = 2, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Clear accumulated predictions."""
        # Per class: list of (score, is_tp) tuples
        self.predictions = [[] for _ in range(self.num_classes)]
        self.num_gt = [0] * self.num_classes

    def update(
        self,
        preds: List[Dict],    # list of {'boxes', 'scores', 'labels'} per image
        targets: List[Dict],  # list of {'boxes', 'labels'} per image
    ):
        """Accumulate predictions for a batch."""
        for pred, target in zip(preds, targets):
            # Move everything to CPU — mAP computation doesn't need GPU
            pred = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in pred.items()}
            target = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            # Count GT boxes per class
            for c in range(self.num_classes):
                self.num_gt[c] += int((gt_labels == c).sum())

            if len(pred["boxes"]) == 0:
                continue

            # Apply NMS per class
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]

            for c in range(self.num_classes):
                # Predicted boxes for this class
                pc_mask = pred_labels == c
                if not pc_mask.any():
                    continue

                pc_boxes = pred_boxes[pc_mask]
                pc_scores = pred_scores[pc_mask]

                # NMS
                keep = nms(pc_boxes, pc_scores, self.iou_threshold)
                pc_boxes = pc_boxes[keep]
                pc_scores = pc_scores[keep]

                # GT boxes for this class
                gc_mask = gt_labels == c
                gc_boxes = gt_boxes[gc_mask]

                matched_gt = set()
                order = pc_scores.argsort(descending=True)

                for idx in order:
                    pb = pc_boxes[idx]
                    sc = pc_scores[idx].item()

                    if len(gc_boxes) == 0:
                        self.predictions[c].append((sc, 0))
                        continue

                    ious = box_iou(pb.unsqueeze(0), gc_boxes)[0]
                    best_iou, best_gt = ious.max(0)

                    if best_iou >= self.iou_threshold and best_gt.item() not in matched_gt:
                        self.predictions[c].append((sc, 1))
                        matched_gt.add(best_gt.item())
                    else:
                        self.predictions[c].append((sc, 0))

    def compute(self) -> Dict:
        """Compute mAP and per-class AP.

        Returns:
            dict with 'mAP', 'AP_car', 'AP_pedestrian' (or generic AP_0, AP_1)
        """
        aps = []
        result = {}

        class_names = ["car", "pedestrian"] if self.num_classes == 2 else [
            f"class_{i}" for i in range(self.num_classes)
        ]

        for c in range(self.num_classes):
            if self.num_gt[c] == 0:
                result[f"AP_{class_names[c]}"] = 0.0
                continue

            preds = sorted(self.predictions[c], key=lambda x: x[0], reverse=True)
            if len(preds) == 0:
                result[f"AP_{class_names[c]}"] = 0.0
                aps.append(0.0)
                continue

            scores, tps = zip(*preds)
            tps = np.cumsum(tps)
            fps = np.cumsum([1 - t for _, t in preds])
            precision = tps / (tps + fps + 1e-8)
            recall = tps / (self.num_gt[c] + 1e-8)

            # 11-point interpolation (VOC 2007 style)
            ap = 0.0
            for thr in np.linspace(0, 1, 11):
                prec_at_rec = precision[recall >= thr]
                if len(prec_at_rec) > 0:
                    ap += prec_at_rec.max() / 11.0

            result[f"AP_{class_names[c]}"] = round(ap, 4)
            aps.append(ap)

        result["mAP"] = round(np.mean(aps) if aps else 0.0, 4)
        return result
