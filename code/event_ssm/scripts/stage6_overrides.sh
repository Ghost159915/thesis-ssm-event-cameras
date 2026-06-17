# Shared Hydra overrides for Stage-6 / Stage-7 training runs.
# Sourced by stage6_run_local.sh, stage6_short_train.slurm, stage7_midrun_local.sh
# (single source of truth -- edit once).
# Caller MUST set before sourcing: DATASET, MAX_STEPS, BATCH, PRECISION
# Optional knobs (defaults reproduce the Stage-6 short-run behaviour exactly):
#   MAX_EPOCHS (default 1)          -> training.max_epochs
#   VAL_EVERY  (default $MAX_STEPS) -> validation.val_check_interval (end-only by default)
#   VAL_FRAC   (default 1.0)        -> validation.limit_val_batches  (full val by default; = RVT default)
#   GROUP_NAME (default stage6_short_mamba2) -> wandb.group_name (offline -> just a label)
: "${MAX_EPOCHS:=1}"
: "${VAL_EVERY:=$MAX_STEPS}"
: "${VAL_FRAC:=1.0}"
: "${GROUP_NAME:=stage6_short_mamba2}"
STAGE6_OVERRIDES=(
  dataset=gen1 model=rnndet +experiment/gen1=resnet_mamba
  dataset.path="$DATASET"
  training.precision="$PRECISION"
  training.max_steps="$MAX_STEPS" training.max_epochs="$MAX_EPOCHS"
  batch_size.train="$BATCH" batch_size.eval="$BATCH"
  validation.val_check_interval="$VAL_EVERY" validation.check_val_every_n_epoch=null
  validation.limit_val_batches="$VAL_FRAC"
  hardware.gpus=0
  wandb.group_name="$GROUP_NAME"
)
