# Stage 5 — Smoke Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax. Runs use the `events_signals` conda env on the RTX
> 5070 Ti (mamba is CUDA-only). Commits must be plain (no AI co-author trailer); never stage `CLAUDE.md` /
> `PLAN_FIXES_FOR_CLAUDE.md`. Small smokes only — **no full training**.

**Goal:** Prove the assembled real system (drop-in `ResNetMamba` + RVT PAFPN/YOLOX/SimOTA + real Gen1 data +
real Lightning `Module`) can overfit a tiny real-data batch and is numerically healthy, before Stage 6.

**Architecture:** Build a tiny `data/gen1_smoke/{train,val,test}` tree by symlinking 2 real `test/`
sequences; a harness sets `sys.path` + `register_resnet_mamba()` + Hydra-composes the real `train` config;
an overfit proof drives the real `Module` with a real `pl.Trainer(limit_train_batches=1)`; a health proof
checks grad-flow / VRAM / latency. No baseline files edited.

**Tech Stack:** PyTorch, PyTorch-Lightning, Hydra/OmegaConf, mamba-ssm (CUDA), the RVT codebase under
`external/ssms_event_cameras/RVT`, matplotlib.

**Spec:** `docs/superpowers/specs/2026-06-12-stage5-smoke-design.md`.

---

## File Structure

- **Create** `code/event_ssm/integration/make_smoke_dataset.py` — build the symlinked smoke dataset tree.
- **Create** `code/event_ssm/integration/smoke_harness.py` — `setup_paths()`, `register`, `compose_smoke_config()`.
- **Create** `code/event_ssm/proofs/smoke_overfit.py` — load-one-batch check + overfit + loss curve.
- **Create** `code/event_ssm/proofs/smoke_health.py` — grad-flow + VRAM sweep + eval-step latency.
- **Modify** `stages/Stage_05_Smoke_Testing.md` — reconcile to the drop-in design.
- **Output** `results/smoke_test/{overfit_loss_curve.png, smoke_results.md}`.

---

## Task 1: Smoke dataset builder

**Files:** Create `code/event_ssm/integration/make_smoke_dataset.py`

- [ ] **Step 1: Write the builder**

```python
"""Build a tiny Gen1 smoke dataset by symlinking K real sequences into train/val/test.
Only the Gen1 `test/` split is present locally; overfitting test data is fine for a smoke
(goal = prove the machinery learns, not generalisation). Data dir is gitignored.

Usage: python -m event_ssm.integration.make_smoke_dataset   (run from code/, or via proofs)
"""
import argparse, pathlib, shutil

REPO = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO / "data/gen1_raw/gen1/test"
DEFAULT_DEST = REPO / "data/gen1_smoke"
LEAF_REPR = "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5"
LEAF_LABELS = "labels_v2/labels.npz"


def build_smoke_dataset(src: pathlib.Path = DEFAULT_SRC,
                        dest: pathlib.Path = DEFAULT_DEST, k: int = 2) -> pathlib.Path:
    src, dest = pathlib.Path(src), pathlib.Path(dest)
    assert src.is_dir(), f"source split not found: {src}"
    seqs = sorted(p for p in src.iterdir() if p.is_dir()
                  and (p / LEAF_REPR).exists() and (p / LEAF_LABELS).exists())
    assert len(seqs) >= k, f"need >= {k} valid sequences in {src}, found {len(seqs)}"
    chosen = seqs[:k]
    if dest.exists():
        shutil.rmtree(dest)
    for split in ("train", "val", "test"):
        sp = dest / split
        sp.mkdir(parents=True, exist_ok=True)
        for seq in chosen:
            link = sp / seq.name
            link.symlink_to(seq.resolve(), target_is_directory=True)
            assert (link / LEAF_REPR).exists(), f"broken symlink leaf: {link}"
    print(f"smoke dataset @ {dest}  (k={k} seqs x 3 splits)")
    for split in ("train", "val", "test"):
        names = [p.name for p in sorted((dest / split).iterdir())]
        print(f"  {split}: {names}")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()
    build_smoke_dataset(pathlib.Path(a.src), pathlib.Path(a.dest), a.k)
```

- [ ] **Step 2: Run it**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code
python -m event_ssm.integration.make_smoke_dataset
```
Expected: prints `smoke dataset @ .../data/gen1_smoke (k=2 seqs x 3 splits)` and the 3 split listings; no
AssertionError.

- [ ] **Step 3: Confirm data/ is gitignored, then commit the builder only**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git check-ignore data/gen1_smoke   # expect: prints the path (ignored)
git add code/event_ssm/integration/make_smoke_dataset.py
git commit -m "feat(stage5): smoke dataset builder (symlink real test seqs into train/val/test)"
```

---

## Task 2: Register + Hydra-compose harness

**Files:** Create `code/event_ssm/integration/smoke_harness.py`

- [ ] **Step 1: Write the harness**

```python
"""Stage-5 harness: put code+RVT on sys.path, register the drop-in backbone, and Hydra-compose the
REAL RVT train config with smoke overrides. Importing this module is side-effect-light; call the
functions explicitly."""
import os, sys, pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
RVT = REPO / "external/ssms_event_cameras/RVT"


def setup_paths():
    for p in (REPO / "code", RVT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def register():
    from event_ssm.integration.register import register_resnet_mamba
    register_resnet_mamba()


def compose_smoke_config(dataset_path=None, max_epochs=50, batch_size=2, extra_overrides=None):
    setup_paths()
    register()
    os.environ.setdefault("WANDB_MODE", "disabled")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from config.modifier import dynamically_modify_train_config
    if dataset_path is None:
        dataset_path = REPO / "data/gen1_smoke"
    overrides = [
        "dataset=gen1",
        "model=resnet_mamba_yolox/default",
        f"dataset.path={dataset_path}",
        "dataset.train.sampling=random",
        f"batch_size.train={batch_size}",
        f"batch_size.eval={batch_size}",
        "hardware.gpus=0",
        "hardware.num_workers.train=0",
        "hardware.num_workers.eval=0",
        f"training.max_epochs={max_epochs}",
    ] + (extra_overrides or [])
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(RVT / "config"), version_base="1.2"):
        cfg = compose(config_name="train", overrides=overrides)
    dynamically_modify_train_config(cfg)
    return cfg


if __name__ == "__main__":
    cfg = compose_smoke_config()
    print("composed OK:",
          "backbone=", cfg.model.backbone.name,
          "num_classes=", cfg.model.head.num_classes,
          "in_ch=", cfg.model.backbone.input_channels,
          "dataset.path=", cfg.dataset.path)
```

- [ ] **Step 2: Run it (validates compose + register + num_classes injection)**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code
python -m event_ssm.integration.smoke_harness
```
Expected: `composed OK: backbone= ResNetMamba num_classes= 2 in_ch= 20 dataset.path= .../data/gen1_smoke`.
If `model=resnet_mamba_yolox/default` fails to compose, ensure the symlink
`external/.../RVT/config/model/resnet_mamba_yolox/default.yaml` resolves (Stage 4 Task 4).

- [ ] **Step 3: Commit**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add code/event_ssm/integration/smoke_harness.py
git commit -m "feat(stage5): register+Hydra-compose harness for the real train config"
```

---

## Task 3: Overfit smoke + loss curve

**Files:** Create `code/event_ssm/proofs/smoke_overfit.py`; Output `results/smoke_test/`

- [ ] **Step 1: Write the overfit proof**

```python
"""Stage-5 overfit smoke: the REAL Lightning Module (drop-in backbone + PAFPN + YOLOX + SimOTA) overfits
a tiny real Gen1 batch via a real pl.Trainer. Asserts loss reduction >= 3x and no NaN. Writes a loss curve.
Run (events_signals, CUDA): python proofs/smoke_overfit.py"""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "code"))
from event_ssm.integration.smoke_harness import compose_smoke_config, REPO
from event_ssm.integration.make_smoke_dataset import build_smoke_dataset

import torch, lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = REPO / "results/smoke_test"; OUT.mkdir(parents=True, exist_ok=True)
build_smoke_dataset()                                  # ensure the tiny dataset exists
cfg = compose_smoke_config(max_epochs=50, batch_size=2)

from modules.utils.fetch import fetch_data_module, fetch_model_module
dm = fetch_data_module(cfg)

# ---- load-one-batch check (de-risk the data path before the overfit) ----
dm.setup("fit")
loader = dm.train_dataloader()
batch0 = next(iter(loader))
print("one batch OK; top keys:", list(batch0.keys()) if isinstance(batch0, dict) else type(batch0))

module = fetch_model_module(cfg)


class LossCurve(Callback):
    def __init__(self): self.losses = []
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and "loss" in outputs:
            self.losses.append(float(outputs["loss"].detach()))


curve = LossCurve()
trainer = pl.Trainer(
    accelerator="gpu", devices=1, precision="bf16-mixed",
    max_epochs=cfg.training.max_epochs, limit_train_batches=1, limit_val_batches=0,
    num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, gradient_clip_val=1.0, callbacks=[curve],
)
trainer.fit(module, datamodule=dm)

losses = curve.losses
assert len(losses) >= 5, f"too few loss points: {len(losses)}"
assert all(torch.isfinite(torch.tensor(l)) for l in losses), "NaN/Inf loss encountered"
initial = sum(losses[:3]) / 3
final = sum(losses[-3:]) / 3
reduction = initial / max(final, 1e-9)

plt.figure(figsize=(6, 4))
plt.plot(losses, lw=1.5)
plt.xlabel("train step (1/epoch)"); plt.ylabel("total loss"); plt.yscale("log")
plt.title(f"Stage 5 overfit: {initial:.2f} -> {final:.2f}  ({reduction:.1f}x)")
plt.tight_layout(); plt.savefig(OUT / "overfit_loss_curve.png", dpi=120)
print(f"initial={initial:.3f} final={final:.3f} reduction={reduction:.1f}x")
print("PASS" if reduction >= 3.0 else "MARGINAL (<3x) -- see report")
assert reduction >= 3.0, f"overfit reduction {reduction:.1f}x < 3x"
print("wrote", OUT / "overfit_loss_curve.png")
```

- [ ] **Step 2: Run it**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python proofs/smoke_overfit.py
```
Expected: prints `one batch OK ...`, trains 50 epochs (a few minutes), prints `reduction=…x` ≥ 3 and `PASS`,
writes `results/smoke_test/overfit_loss_curve.png`. If MARGINAL: re-run with `build_smoke_dataset(k=1)` and
`max_epochs=80` (documented fallback) — a smoke proves learning-trend + finiteness.

- [ ] **Step 3: Commit (script + curve)**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add code/event_ssm/proofs/smoke_overfit.py results/smoke_test/overfit_loss_curve.png
git commit -m "feat(stage5): overfit smoke on real Gen1 batch (loss reduction + curve)"
```

---

## Task 4: Health probes (grad-flow, VRAM, latency)

**Files:** Create `code/event_ssm/proofs/smoke_health.py`; Output `results/smoke_test/smoke_results.md`

- [ ] **Step 1: Write the health proof**

```python
"""Stage-5 health probes on the assembled YoloXDetector (drop-in backbone): gradient flow, VRAM at
batch sizes, and eval single-window step latency. Synthetic real-shaped clips (no data module needed).
Run (events_signals, CUDA): python proofs/smoke_health.py"""
import sys, pathlib, time, torch
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "code"))
from event_ssm.integration.smoke_harness import setup_paths, register, REPO
setup_paths(); register()
from omegaconf import OmegaConf
from models.detection.yolox_extension.models.detector import YoloXDetector

OUT = REPO / "results/smoke_test"; OUT.mkdir(parents=True, exist_ok=True)
mcfg = OmegaConf.create({
    "backbone": {"name": "ResNetMamba", "input_channels": 20, "pretrained": False,
                 "d_state": 16, "num_layers_per_stage": 1, "compile": {"enable": False}},
    "fpn": {"name": "PAFPN", "depth": 0.67, "in_stages": [2, 3, 4],
            "depthwise": False, "act": "silu", "compile": {"enable": False}},
    "head": {"name": "YoloX", "num_classes": 2, "depthwise": False, "act": "silu",
             "compile": {"enable": False}},
})
model = YoloXDetector(mcfg).cuda()

# ---------- grad-flow ----------
model.train()
x = torch.randn(5, 2, 20, 256, 320, device="cuda")
feats, _ = model.forward_backbone(x, previous_states=None, train_step=True)
sel = {k: v[-1] for k, v in feats.items() if k in (2, 3, 4)}
tgt = torch.zeros(2, 3, 5, device="cuda"); tgt[:, 0] = torch.tensor([0., 160., 128., 40., 30.], device="cuda")
_, losses = model.forward_detect(backbone_features=sel, targets=tgt)
loss = losses["loss"] if isinstance(losses, dict) else losses
loss.backward()
EXCLUDE = "backbone.temporal.0."        # stage-1 temporal Mamba: unused by FPN (Finding S8, documented)
missing, nan = [], []
for n, p in model.named_parameters():
    if not p.requires_grad or n.startswith(EXCLUDE):
        continue
    if p.grad is None or p.grad.abs().max() == 0:
        missing.append(n)
    elif not torch.isfinite(p.grad).all():
        nan.append(n)
grad_ok = not missing and not nan

# ---------- VRAM sweep ----------
vram = {}
for B in (1, 2, 4):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    xb = torch.randn(5, B, 20, 256, 320, device="cuda")
    fb, _ = model.forward_backbone(xb, previous_states=None, train_step=True)
    s = {k: v[-1] for k, v in fb.items() if k in (2, 3, 4)}
    _, lo = model.forward_detect(backbone_features=s, targets=tgt[:1].expand(B, 3, 5).contiguous())
    (lo["loss"] if isinstance(lo, dict) else lo).backward()
    vram[B] = torch.cuda.max_memory_allocated() / 1024 ** 3
    model.zero_grad(set_to_none=True)

# ---------- eval single-window step latency ----------
model.eval()
xw = torch.randn(1, 1, 20, 256, 320, device="cuda")
st = None
with torch.no_grad():
    for _ in range(20):                         # warmup
        f, st = model.forward_backbone(xw, previous_states=st, train_step=False)
    torch.cuda.synchronize()
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        f, st = model.forward_backbone(xw, previous_states=st, train_step=False)
        sd = {k: v[-1] for k, v in f.items() if k in (2, 3, 4)}
        _ = model.forward_detect(backbone_features=sd)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
mean = sum(ts) / len(ts)
std = (sum((t - mean) ** 2 for t in ts) / len(ts)) ** 0.5

lines = [
    "# Stage 5 - smoke health probes", "",
    "| test | result | notes |", "|---|---|---|",
    f"| gradient flow | {'PASS' if grad_ok else 'FAIL'} | "
    f"missing={len(missing)} nan={len(nan)} (excl. stage-1 temporal[0], unused by FPN) |",
    f"| VRAM @ bs1/bs2/bs4 | {vram[1]:.2f}/{vram[2]:.2f}/{vram[4]:.2f} GB | target <10GB @ bs4 |",
    f"| eval step latency | {mean:.2f} +/- {std:.2f} ms | S5-RVT ~12 ms/window |",
    f"| max throughput | {1000/mean:.0f} Hz | window dt=50ms -> need <50ms |",
]
if missing: lines.append(f"\nMISSING GRAD ({len(missing)}): " + ", ".join(missing[:10]))
if nan: lines.append(f"\nNAN GRAD ({len(nan)}): " + ", ".join(nan[:10]))
(OUT / "smoke_results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print("wrote", OUT / "smoke_results.md")
assert grad_ok, f"grad-flow FAIL: missing={missing[:5]} nan={nan[:5]}"
assert vram[4] < 10.0, f"VRAM @ bs4 = {vram[4]:.2f}GB exceeds 10GB"
```

- [ ] **Step 2: Run it**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python proofs/smoke_health.py
```
Expected: prints the table; `gradient flow | PASS`; VRAM @ bs4 < 10 GB; latency mean ± std; writes
`results/smoke_test/smoke_results.md`.

- [ ] **Step 3: Commit**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add code/event_ssm/proofs/smoke_health.py results/smoke_test/smoke_results.md
git commit -m "feat(stage5): health probes (grad-flow + VRAM sweep + eval-step latency)"
```

---

## Task 5: Reconcile the Stage 5 doc

**Files:** Modify `stages/Stage_05_Smoke_Testing.md`

- [ ] **Step 1: Replace the stale sections** — change the standalone-`EventSSMDetector` framing to the
  drop-in design: input is **20-channel** stacked histogram (not 10); state via the **`LstmStates`** contract
  (not `model.reset_state`); loss is **SimOTA / IoU `1−iou²`** (not Focal/GIoU); the "model" is the real RVT
  `Module`. Add a STATUS banner pointing at `proofs/smoke_overfit.py`, `proofs/smoke_health.py`,
  `results/smoke_test/`, and noting the grad-flow `temporal[0]` exclusion (Finding §8). Keep the 4-test
  structure and success thresholds (overfit ≥3×; VRAM <10GB @ bs4; latency vs ~12 ms).

- [ ] **Step 2: Commit**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add stages/Stage_05_Smoke_Testing.md
git commit -m "docs(stage5): reconcile smoke-testing doc to the drop-in design (20-ch, LstmStates, SimOTA)"
```

---

## Task 6: Regression + code review + Stage 5 report

- [ ] **Step 1: Confirm the pytest suite still green**

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm
python -m pytest tests/ -q
```
Expected: 20 passed.

- [ ] **Step 2: Code review** the Stage-5 changes (requesting-code-review skill) over the Stage-5 commit
  range; fix Critical/Important inline.

- [ ] **Step 3: Write `reports/Stage_05_Smoke_Report.md`** — added/changed/removed, the overfit reduction +
  curve, the health table, what went well/wrong, the Finding-§8 re-evaluation, and the Stage-6 prerequisites
  (β cross-clip state; user downloads full Gen1 train split). Commit.

```bash
cd /home/ghost/Desktop/thesis-ssm-event-cameras
git add reports/Stage_05_Smoke_Report.md
git commit -m "docs(stage5): stage report (overfit + health results, findings, Stage-6 prereqs)"
```

---

## Notes / Out of Scope
- β cross-clip **training**-state parity → Stage 6 (training still zero-inits per clip).
- Finding §8 (dead stage-1 temporal) architecture change → recommended, user-approved, after the smoke passes.
- Full Gen1 train split download + real short training → Stage 6 (user-run).
