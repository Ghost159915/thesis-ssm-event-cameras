# Shared Hydra overrides for Stage-6 / Stage-7 training runs.
# Sourced by stage6_run_local.sh, stage6_short_train.slurm, stage7_midrun_local.sh
# (single source of truth -- edit once).
# Caller MUST set before sourcing: DATASET, MAX_STEPS, BATCH, PRECISION
# Optional knobs (defaults reproduce the Stage-6 short-run behaviour exactly):
#   MAX_EPOCHS (default 1)          -> training.max_epochs
#   VAL_EVERY  (default $MAX_STEPS) -> validation.val_check_interval (end-only by default)
#   VAL_FRAC   (default 1.0)        -> validation.limit_val_batches  (full val by default; = RVT default)
#   GROUP_NAME (default stage6_short_mamba2) -> wandb.group_name (offline -> just a label)
#   NUM_WORKERS_TRAIN (default 6) -> hardware.num_workers.train  (lower to cut HOST-RAM use; see below)
#   NUM_WORKERS_EVAL  (default 2) -> hardware.num_workers.eval
#   EXPERIMENT (default resnet_mamba) -> +experiment/gen1=$EXPERIMENT (e.g. `puressm` -- Stage 12/13)
# NOTE: each Gen1 train worker held ~3.6 GB RSS; on a 16 GB-RAM box 6 workers exhausted system memory
# and the kernel OOM-killer killed a pt_data_worker mid-run (2026-06-17). Lower these on small-RAM hosts.
: "${MAX_EPOCHS:=1}"
: "${VAL_EVERY:=$MAX_STEPS}"
: "${VAL_FRAC:=1.0}"
: "${GROUP_NAME:=stage6_short_mamba2}"
: "${NUM_WORKERS_TRAIN:=6}"
: "${NUM_WORKERS_EVAL:=2}"
: "${EXPERIMENT:=resnet_mamba}"
STAGE6_OVERRIDES=(
  dataset=gen1 model=rnndet +experiment/gen1="$EXPERIMENT"
  dataset.path="$DATASET"
  training.precision="$PRECISION"
  training.max_steps="$MAX_STEPS" training.max_epochs="$MAX_EPOCHS"
  batch_size.train="$BATCH" batch_size.eval="$BATCH"
  validation.val_check_interval="$VAL_EVERY" validation.check_val_every_n_epoch=null
  validation.limit_val_batches="$VAL_FRAC"
  hardware.gpus=0
  hardware.num_workers.train="$NUM_WORKERS_TRAIN" hardware.num_workers.eval="$NUM_WORKERS_EVAL"
  wandb.group_name="$GROUP_NAME"
)
