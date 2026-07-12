#!/usr/bin/env bash
# Stage-13 CLOUD SHORT RUN (rented RTX 5090, sm_120). PureSSM backbone, 25k-step short training run --
# a quick "does the ablation train + does the pipeline hold up on rented cloud hardware" signal, well
# short of the full 400k baseline budget (Stage 7/8 already established that comparison locally for
# EventSSMDetector; this is the cloud smoke/short-run counterpart for PureSSM).
#
# Thin wrapper over stage7_midrun_local.sh -- inherits EVERY OOM/resume/Hydra fix (single source of
# truth; REPO/CONDA_SH are now env-overridable there specifically so this cloud wrapper can point at a
# different checkout/conda root without touching the local script's defaults). Only the experiment
# selection + step budget + labels + run dir + wandb mode differ (exported below).
#
# CONDA_SH default: stage7_midrun_local.sh falls back to a hardcoded local-dev-machine path
# (/home/ghost/miniforge3/...) when CONDA_SH is unset, which does not exist on a rented instance (root
# user, different $HOME). This wrapper exports a $HOME-relative default instead, matching where
# setup_env_5090.sh installs Miniforge on the cloud instance -- so training doesn't abort at the
# `source "$CONDA_SH"` line in stage7_midrun_local.sh. Still overridable via CONDA_SH=... if needed.
#
# Prerequisites (see code/event_ssm/scripts/cloud/): setup_env_5090.sh (env bootstrap) then
# pull_dataset.sh (train/val only -- test stays local-only) must have already run on this instance.
#
#   bash stage13_cloud_short.sh                 # PureSSM, 25k steps, on cloud GPU
#   bash stage13_cloud_short.sh --cfg job        # GPU-free dry-run (compose config + exit)
#   EXPERIMENT=resnet_mamba bash stage13_cloud_short.sh   # override the ablation arm if needed
set -euo pipefail

export REPO="${REPO:-$HOME/thesis-ssm-event-cameras}"
export CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

# --- cloud short-run knobs (25k steps, PureSSM) -- everything else inherits stage7_midrun_local.sh's
#     mid-run defaults (DATASET, PRECISION=bf16-mixed, MAX_EPOCHS=10000) ---
export EXPERIMENT=puressm                # +experiment/gen1=puressm (Stage 12 backbone; stage6_overrides.sh knob)
export MAX_STEPS=25000                   # short cloud smoke/signal run, not the full 400k budget
export VAL_EVERY=5000                    # 5 full-val mAP points across the run
export BATCH=4
export NUM_WORKERS_TRAIN=6               # cloud RAM allows the yaml defaults again (local was RAM-capped to 2)
export NUM_WORKERS_EVAL=2
export GROUP_NAME=stage13_cloud_puressm  # wandb label
export RUNDIR="$REPO/results/stage13_cloud"
export PURESSM_MONITOR=1                 # attach the PureSSM spatial-norm monitor (register.py; zero-overhead when unset)
export WANDB_MODE="${WANDB_MODE:-online}"   # cloud instance -> log live, not offline like the local runs

exec bash "$REPO/code/event_ssm/scripts/stage7_midrun_local.sh" "$@"
