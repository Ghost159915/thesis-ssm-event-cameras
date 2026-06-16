# Shared Hydra overrides for the Stage-6 short training run.
# Sourced by stage6_run_local.sh and stage6_short_train.slurm (single source of truth — edit once).
# Caller must export/set: DATASET, MAX_STEPS, BATCH, PRECISION before sourcing.
STAGE6_OVERRIDES=(
  dataset=gen1 model=rnndet +experiment/gen1=resnet_mamba
  dataset.path="$DATASET"
  training.precision="$PRECISION"
  training.max_steps="$MAX_STEPS" training.max_epochs=1
  batch_size.train="$BATCH" batch_size.eval="$BATCH"
  validation.val_check_interval="$MAX_STEPS" validation.check_val_every_n_epoch=null
  hardware.gpus=0
  wandb.group_name=stage6_short_mamba2          # RVT requires this mandatory value (offline -> just a label)
)
