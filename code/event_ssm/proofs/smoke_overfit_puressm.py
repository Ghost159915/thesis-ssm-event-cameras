"""Stage-12 overfit smoke: PureSSM (BiMamba spatial) through the REAL Lightning stack..."""
import sys, pathlib
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))      # parents[2] == repo/code (so `event_ssm` is importable)
from event_ssm.integration.smoke_harness import compose_smoke_config, REPO
from event_ssm.integration.make_smoke_dataset import build_smoke_dataset

import torch, lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 150   # 150 reliably reaches >=3x (see report)
OUT = REPO / "results/smoke_test"; OUT.mkdir(parents=True, exist_ok=True)

build_smoke_dataset()                                  # ensure the tiny dataset exists
# Overfit needs a CONSTANT, larger LR: the stock OneCycle scheduler (total_steps=400k) keeps LR in
# warmup (~1e-5) over a 50-step run -> no learning. Disable it and use a fixed 1e-3 (Stage-5 doc).
cfg = compose_smoke_config(max_epochs=MAX_EPOCHS, batch_size=2, experiment="puressm",
                           extra_overrides=["training.lr_scheduler.use=False",
                                            "training.learning_rate=1e-3",
                                            "model.backbone.checkpoint_blocks=True",
                                            # DropPath OFF for the overfit gate: stochastic depth
                                            # actively fights single-batch memorization and the 3x
                                            # threshold was calibrated on the DropPath-free ResNet
                                            # smoke. Training keeps 0.1 (this is smoke-only).
                                            "model.backbone.drop_path_rate=0.0"])

from modules.utils.fetch import fetch_data_module, fetch_model_module
dm = fetch_data_module(cfg)
dm.setup("fit")
train_loader = dm.train_dataloader()
module = fetch_model_module(cfg)
# overfit_batches=1 re-fetches index 0 every epoch via the REAL dataloader: a fixed target (no shuffle)
# with FRESH label objects each epoch. We cannot just repeat one captured batch -- training_step runs
# to_prophesee()->numpy_() which mutates label objects to numpy IN PLACE (irreversible), breaking the
# next epoch's device transfer. Pass only the train loader (no val) so Prophesee eval never runs.


class LossCurve(Callback):
    def __init__(self):
        self.losses = []
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and "loss" in outputs:
            self.losses.append(float(outputs["loss"].detach()))


curve = LossCurve()
trainer = pl.Trainer(
    accelerator="gpu", devices=1, precision="bf16-mixed",
    max_epochs=MAX_EPOCHS, overfit_batches=1,
    num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, gradient_clip_val=1.0, callbacks=[curve],
)
trainer.fit(module, train_dataloaders=train_loader)

losses = curve.losses
print(f"captured {len(losses)} loss points; first few: {[round(l, 3) for l in losses[:5]]}")
assert len(losses) >= 5, f"too few loss points: {len(losses)}"
assert all(torch.isfinite(torch.tensor(l)) for l in losses), "NaN/Inf loss encountered"
initial = sum(losses[:3]) / 3
final = sum(losses[-3:]) / 3
reduction = initial / max(final, 1e-9)

plt.figure(figsize=(6, 4))
plt.plot(losses, lw=1.5)
plt.xlabel("train step (1/epoch)"); plt.ylabel("total loss"); plt.yscale("log")
plt.title(f"Stage 12 PureSSM overfit ({MAX_EPOCHS} ep): {initial:.2f} -> {final:.2f}  ({reduction:.1f}x)")
plt.tight_layout(); plt.savefig(OUT / "puressm_overfit_loss_curve.png", dpi=120)
print(f"initial={initial:.3f} final={final:.3f} reduction={reduction:.1f}x")
print("wrote", OUT / "puressm_overfit_loss_curve.png")
print("PASS" if reduction >= 3.0 else "MARGINAL (<3x) -- see report")
# Gate (spec Unit 3 / plan): a smoke that cannot overfit a fixed real batch >=3x is a learning
# regression -- fail loudly so automation/CI catches it (the curve + numbers are still written above).
assert reduction >= 3.0, f"overfit reduction {reduction:.1f}x < 3x (no-learning regression)"
