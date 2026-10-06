"""Stage-19 overfit smoke: SpikingSSM (PureSSM skeleton + LIF temporal readout) through the REAL
Lightning stack (drop-in backbone + PAFPN + YOLOX + SimOTA) overfits a tiny real Gen1 batch via a
real pl.Trainer. Same gate as Stages 5/12 (>= 3x loss reduction, no NaN) plus, for the spike and
graded arms, a firing-rate band per spiking stage. The gates live in integration/spiking_smoke.py
(CPU-tested); this script only runs the model and records.

Writes results/smoke_test/spikingssm_<mode>_s<stages>_overfit[_e<epochs>][_run<n>].{png,json}
(reruns are numbered, never overwritten); exit code 1 on FAIL.

Run (events_signals, CUDA, idle GPU), ladder start first:
  python proofs/smoke_overfit_spikingssm.py --mode analog --stages 4
  python proofs/smoke_overfit_spikingssm.py --mode spike  --stages 4
"""
import argparse, json, os, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))      # parents[2] == repo/code (so `event_ssm` is importable)

ap = argparse.ArgumentParser(description="Stage-19 SpikingSSM overfit smoke")
ap.add_argument("--mode", choices=("analog", "graded", "spike"), default="analog",
                help="output_mode ablation arm (analog first: least optimisation risk)")
ap.add_argument("--stages", type=int, nargs="+", default=[4],
                help="spiking_stages (ladder rung): 4 | 3 4 | 2 3 4")
ap.add_argument("--epochs", type=int, default=150,
                help="1 step per epoch; 150 reliably reached >=3x for PureSSM (Stage 12)")
ap.add_argument("--monitor-every", type=int, default=10,
                help="[spk-monitor]/[monitor] cadence; the training default (200) never fires in a "
                     "150-step smoke")
args = ap.parse_args()
stages = sorted(set(args.stages))
if stages not in ([4], [3, 4], [2, 3, 4]):           # same rungs as scripts/stage19_short_local.sh
    ap.error(f"--stages must be one ladder rung: 4 | 3 4 | 2 3 4 (got {stages})")
tag = f"{args.mode}_s{''.join(map(str, stages))}"

# Monitors attach at backbone build time (register.py reads the env), so set them before building.
os.environ["SPIKING_MONITOR"] = "1"
os.environ["PURESSM_MONITOR"] = "1"
os.environ["SPIKING_MONITOR_EVERY"] = str(args.monitor_every)
os.environ["PURESSM_MONITOR_EVERY"] = str(args.monitor_every)
# Contamination guard (as the Stage-10 launcher): a Stage-9 inference-time dt hook must not leak
# into a training smoke.
if os.environ.pop("MAMBA_STEP_SCALE", None) is not None:
    print("[stage19] WARNING: MAMBA_STEP_SCALE was set in the environment; unset for this smoke")

from event_ssm.integration.smoke_harness import compose_smoke_config, REPO
from event_ssm.integration.make_smoke_dataset import build_smoke_dataset
from event_ssm.integration.spiking_smoke import (
    SpikingSmokeRecorder, find_spiking_backbone, json_safe, median_step_ms, output_stem, plot_smoke,
    smoke_verdict, verify_arm,
)
from event_ssm.models.spikingssm.lif import _BETA_EPS

import torch, lightning.pytorch as pl
from omegaconf import OmegaConf

OUT = REPO / "results/smoke_test"; OUT.mkdir(parents=True, exist_ok=True)

build_smoke_dataset()                                  # ensure the tiny dataset exists
# Smoke-only overrides, identical to the Stage-12 PureSSM smoke so the two are comparable:
#  * constant LR 1e-3: the stock OneCycle (total_steps=400k) stays in warm-up over 150 steps;
#  * checkpoint_blocks=True: the local 16 GB card (the training recipe leaves it False);
#  * drop_path_rate=0.0: stochastic depth fights single-batch memorisation (Stage-12 lesson).
# Then the arm: output_mode + the ladder rung.
cfg = compose_smoke_config(max_epochs=args.epochs, batch_size=2, experiment="spikingssm",
                           extra_overrides=["training.lr_scheduler.use=False",
                                            "training.learning_rate=1e-3",
                                            "model.backbone.checkpoint_blocks=True",
                                            "model.backbone.drop_path_rate=0.0",
                                            f"model.backbone.spiking.output_mode={args.mode}",
                                            "model.backbone.spiking.spiking_stages="
                                            f"[{','.join(map(str, stages))}]"])
spk_cfg = cfg.model.backbone.spiking
if spk_cfg.output_mode != args.mode or sorted(spk_cfg.spiking_stages) != stages:
    raise RuntimeError(f"arm override did not reach the config: {spk_cfg}")

from modules.utils.fetch import fetch_data_module, fetch_model_module
dm = fetch_data_module(cfg)
dm.setup("fit")
train_loader = dm.train_dataloader()
module = fetch_model_module(cfg)

# Fail closed on a mislabelled arm: the BUILT model, not the config, decides what was measured.
verify_arm(find_spiking_backbone(module), args.mode, stages)   # raises; never stripped by -O
print(f"[stage19] arm verified on the built model: output_mode={args.mode} spiking_stages={stages}")
# overfit_batches=1 re-fetches index 0 every epoch via the REAL dataloader (fresh label objects;
# training_step mutates labels to numpy in place, so one captured batch cannot be replayed).
# Only the train loader is passed, so Prophesee evaluation never runs.

rec = SpikingSmokeRecorder()
trainer = pl.Trainer(
    accelerator="gpu", devices=1, precision="bf16-mixed",
    max_epochs=args.epochs, overfit_batches=1,
    num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, gradient_clip_val=1.0, callbacks=[rec],
)
torch.cuda.reset_peak_memory_stats()
trainer.fit(module, train_dataloaders=train_loader)
peak_gb = torch.cuda.max_memory_allocated() / 2**30   # same unit as the Stage-11 probe's 8.55 GB
step_ms = median_step_ms(rec.step_s)

v = smoke_verdict(rec.losses, rec.rates, args.mode, expected_stages=stages)
try:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout.strip()
except Exception:
    sha = "unknown"
summary = dict(stage=19, arm=tag, spiking_stages=stages, epochs=args.epochs,   # `mode` comes from v
               n_steps=len(rec.losses), peak_vram_gb=round(peak_gb, 3),
               median_step_ms=round(step_ms, 1), git=sha,
               final_beta_mean={s: xs[-1] for s, xs in rec.beta_mean.items()},
               final_beta_max={s: xs[-1] for s, xs in rec.beta_max.items()},
               # the whole LIF block (threshold, residual, learn_*...), so provenance never rests
               # on config defaults
               spiking_cfg=OmegaConf.to_container(spk_cfg, resolve=True), **v)
stem = output_stem(OUT, tag, args.epochs)             # reruns are numbered, never overwritten
(OUT / f"{stem}.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
plot_smoke(rec, v, args.mode, stages, OUT / f"{stem}.png",
           beta_cap=1.0 - _BETA_EPS,
           footer=f"peak VRAM {peak_gb:.2f} GB · median step {step_ms:.0f} ms "
                  f"· {len(rec.losses)} steps · git {sha}")

print(f"captured {len(rec.losses)} loss points; first few: {[round(x, 3) for x in rec.losses[:5]]}")
print(f"loss {v['initial_loss']:.3f} -> {v['final_loss']:.3f}  reduction={v['reduction']:.1f}x")
for s in sorted(v["firing"]):
    print(f"stage {s}: final rate={v['final_rate'][s]:.4f} [{v['firing'][s]}"
          f"{'' if v['firing_gated'] else ', not gated'}]  beta mean/max="
          f"{summary['final_beta_mean'][s]:.4f}/{summary['final_beta_max'][s]:.4f}")
print(f"peak VRAM {peak_gb:.2f} GB  median step {step_ms:.0f} ms")
print("wrote", OUT / f"{stem}.png", "and .json")
if v["passed"]:
    print("PASS")
else:
    print("FAIL: " + "; ".join(v["failures"]))
    sys.exit(1)
