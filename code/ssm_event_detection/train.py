"""
Training entry point — supports both model architectures.

Usage:
    # Train CNN-SSM hybrid (default)
    python train.py --config configs/default.yaml

    # Train pure SSM (no CNN)
    python train.py --config configs/pure_ssm.yaml

    # Smoke test either model with synthetic data
    python train.py --smoke-test
    python train.py --config configs/pure_ssm.yaml --smoke-test

    # Override specific fields
    python train.py --config configs/pure_ssm.yaml --batch-size 4 --epochs 50

The model is selected via the 'model.name' field in the config:
    model.name: "hybrid"    → EventSSMDetector (CNN backbone + Mamba)
    model.name: "pure_ssm"  → PureSSMDetector  (patch embed + BiMamba + Mamba)
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

from models import EventSSMDetector, PureSSMDetector
from data import Gen1Detection, gen1_collate_fn
from utils.device import get_device
from utils.loss import DetectionLoss
from utils.metrics import MeanAveragePrecision


# ---------------------------------------------------------------------------
# Synthetic dataset for smoke-testing (no real data required)
# ---------------------------------------------------------------------------

class SyntheticEventDataset(Dataset):
    """Generates random event camera data for pipeline testing.

    Creates voxel grids with synthetic objects at random positions,
    letting you verify the full training loop works before downloading
    the Gen1 dataset.
    """

    def __init__(
        self,
        num_samples: int = 64,
        num_bins: int = 10,
        height: int = 240,
        width: int = 304,
        num_classes: int = 2,
        max_objects: int = 3,
    ):
        self.num_samples = num_samples
        self.num_bins = num_bins
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.max_objects = max_objects

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random voxel grid (simulates sparse event activity)
        voxel = torch.zeros(self.num_bins, self.height, self.width)
        # Sparse random events: ~2% of cells active per bin
        for b in range(self.num_bins):
            n_active = int(0.02 * self.height * self.width)
            ys = torch.randint(0, self.height, (n_active,))
            xs = torch.randint(0, self.width, (n_active,))
            vals = torch.randn(n_active) * 0.5
            voxel[b, ys, xs] = vals

        # Random bounding boxes
        n_obj = random.randint(0, self.max_objects)
        if n_obj > 0:
            boxes = []
            labels = []
            for _ in range(n_obj):
                x1 = random.randint(0, self.width - 40)
                y1 = random.randint(0, self.height - 40)
                x2 = x1 + random.randint(20, min(80, self.width - x1))
                y2 = y1 + random.randint(20, min(60, self.height - y1))
                boxes.append([x1, y1, x2, y2])
                labels.append(random.randint(0, self.num_classes - 1))
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
            }

        return voxel, target


def synthetic_collate_fn(batch):
    """Collate a batch of (voxel, target) tuples from SyntheticEventDataset."""
    voxels = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return voxels, targets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: DetectionLoss,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
) -> dict:
    model.train()
    total_loss = 0.0
    loss_cls_total = 0.0
    loss_obj_total = 0.0
    loss_box_total = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for i, (voxels, targets) in enumerate(pbar):
        voxels = voxels.to(device)

        optimizer.zero_grad()
        cls_pred, obj_pred, box_pred = model(voxels)
        losses = criterion(cls_pred, obj_pred, box_pred, targets)

        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += losses["loss"].item()
        loss_cls_total += losses["loss_cls"].item()
        loss_obj_total += losses["loss_obj"].item()
        loss_box_total += losses["loss_box"].item()
        n_batches += 1

        if (i + 1) % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{total_loss/n_batches:.3f}",
                "cls": f"{loss_cls_total/n_batches:.3f}",
                "obj": f"{loss_obj_total/n_batches:.3f}",
                "box": f"{loss_box_total/n_batches:.3f}",
            })

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_cls": loss_cls_total / max(n_batches, 1),
        "loss_obj": loss_obj_total / max(n_batches, 1),
        "loss_box": loss_box_total / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DetectionLoss,
    device: torch.device,
    conf_threshold: float = 0.01,
    num_classes: int = 2,
) -> dict:
    model.eval()
    metric = MeanAveragePrecision(num_classes=num_classes)
    total_loss = 0.0
    n_batches = 0

    for voxels, targets in tqdm(loader, desc="Evaluating", leave=False):
        voxels = voxels.to(device)

        cls_pred, obj_pred, box_pred = model(voxels)
        losses = criterion(cls_pred, obj_pred, box_pred, targets)
        total_loss += losses["loss"].item()
        n_batches += 1

        # Decode predictions and accumulate metrics
        preds = model.decode_predictions(
            cls_pred, obj_pred, box_pred,
            conf_threshold=conf_threshold,
        )
        metric.update(preds, targets)

    results = metric.compute()
    results["val_loss"] = total_loss / max(n_batches, 1)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EventSSMDetector")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick test with synthetic data (no dataset needed)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Command-line overrides
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    print(f"\n{'='*60}")
    print(f"  EventSSMDetector Training")
    print(f"  Mode: {'SMOKE TEST (synthetic data)' if args.smoke_test else 'Gen1 dataset'}")
    print(f"  Device: {device}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # --- Datasets ---
    num_bins = cfg["dataset"]["num_bins"]
    height = cfg["dataset"]["height"]
    width = cfg["dataset"]["width"]
    num_classes = cfg["dataset"]["num_classes"]
    bs = cfg["training"]["batch_size"]

    if args.smoke_test:
        train_ds = SyntheticEventDataset(
            num_samples=128, num_bins=num_bins, height=height, width=width
        )
        val_ds = SyntheticEventDataset(
            num_samples=32, num_bins=num_bins, height=height, width=width
        )
        collate = synthetic_collate_fn
        cfg["training"]["epochs"] = min(cfg["training"]["epochs"], 3)
    else:
        train_ds = Gen1Detection(
            root=cfg["dataset"]["root"],
            split="train",
            num_bins=num_bins,
            height=height,
            width=width,
            time_window_us=cfg["dataset"]["time_window_us"],
        )
        val_ds = Gen1Detection(
            root=cfg["dataset"]["root"],
            split="test",
            num_bins=num_bins,
            height=height,
            width=width,
            time_window_us=cfg["dataset"]["time_window_us"],
        )
        collate = gen1_collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=cfg["training"]["num_workers"] if not args.smoke_test else 0,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"] if not args.smoke_test else 0,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

    # --- Model ---
    model_name = cfg["model"].get("name", "hybrid")

    if model_name == "hybrid":
        model = EventSSMDetector(
            num_bins=num_bins,
            height=height,
            width=width,
            backbone_out=cfg["model"]["backbone"]["out_features"],
            d_model=cfg["model"]["mamba"]["d_model"],
            d_state=cfg["model"]["mamba"]["d_state"],
            d_conv=cfg["model"]["mamba"]["d_conv"],
            mamba_expand=cfg["model"]["mamba"]["expand"],
            mamba_layers=cfg["model"]["mamba"]["num_layers"],
            num_classes=num_classes,
        ).to(device)

    elif model_name == "pure_ssm":
        pss = cfg["model"]["pure_ssm"]
        model = PureSSMDetector(
            num_bins=num_bins,
            height=height,
            width=width,
            patch_size=pss["patch_size"],
            d_model=pss["d_model"],
            d_state=pss["d_state"],
            d_conv=pss["d_conv"],
            expand=pss["expand"],
            spatial_layers=pss["spatial_layers"],
            temporal_layers=pss["temporal_layers"],
            num_classes=num_classes,
        ).to(device)

    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'hybrid' or 'pure_ssm'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}")
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)\n")

    # --- Optimizer & scheduler ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    epochs = cfg["training"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Loss ---
    criterion = DetectionLoss(
        num_classes=num_classes,
        stride=8,
        cls_weight=cfg["training"]["loss"]["cls_weight"],
        box_weight=cfg["training"]["loss"]["box_weight"],
        obj_weight=cfg["training"]["loss"]["obj_weight"],
    )

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_map = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} (best mAP: {best_map:.4f})\n")

    # --- Training loop ---
    history = []
    log_interval = cfg.get("log_interval", 50)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, log_interval
        )
        scheduler.step()

        val_metrics = evaluate(
            model, val_loader, criterion, device,
            conf_threshold=cfg["evaluation"]["conf_threshold"],
            num_classes=num_classes,
        )

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss: {train_metrics['loss']:.4f} | "
            f"val_loss: {val_metrics['val_loss']:.4f} | "
            f"mAP: {val_metrics['mAP']:.4f} | "
            f"AP_car: {val_metrics.get('AP_car', 0):.4f} | "
            f"AP_ped: {val_metrics.get('AP_pedestrian', 0):.4f} | "
            f"lr: {lr_now:.2e} | "
            f"time: {elapsed:.1f}s"
        )

        history.append({**train_metrics, **val_metrics, "epoch": epoch})

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_map": best_map,
            "config": cfg,
        }
        torch.save(ckpt, out_dir / "last.pth")

        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            ckpt["best_map"] = best_map
            torch.save(ckpt, out_dir / "best.pth")
            print(f"  ✓ New best mAP: {best_map:.4f} — saved to {out_dir / 'best.pth'}")

    print(f"\nTraining complete. Best mAP: {best_map:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
