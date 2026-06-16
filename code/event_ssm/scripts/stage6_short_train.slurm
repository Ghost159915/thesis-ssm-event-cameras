#!/bin/bash
# Stage-6 SHORT training on Katana (single CUDA GPU). Same launcher + RVT train.py as the local run.
# Fill in the cluster's CUDA module + conda path (placeholders marked <-- below).
#SBATCH --job-name=ssm_stage6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=stage6_%j.out
set -euo pipefail

# module load cuda/12.8                          # <-- set to Katana's CUDA >=12.8 (Blackwell/cu128 stack)
set +u                                             # conda's activate.d (cuda-nvcc) references unbound vars
source ~/miniforge3/etc/profile.d/conda.sh && conda activate events_signals   # <-- adjust conda path
set -u
export WANDB_MODE=offline                          # RVT wandb cfg has no 'mode' key -> use the env var

REPO="$SLURM_SUBMIT_DIR"                          # submit from the repo root
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
cd "$REPO/external/ssms_event_cameras/RVT"

DATASET="$REPO/data/gen1_subset10"
MAX_STEPS=2000
BATCH=8

srun python "$REPO/code/event_ssm/scripts/stage6_train.py" \
  dataset=gen1 model=rnndet +experiment/gen1=resnet_mamba \
  dataset.path="$DATASET" \
  training.precision=bf16-mixed \
  training.max_steps="$MAX_STEPS" training.max_epochs=1 \
  batch_size.train="$BATCH" batch_size.eval="$BATCH" \
  validation.val_check_interval="$MAX_STEPS" validation.check_val_every_n_epoch=null \
  hardware.gpus=0 \
  wandb.group_name=stage6_short_mamba2
