#!/usr/bin/env bash
# Stage-7b LOCAL FULL-BUDGET RUN (RTX 5070 Ti). FROM SCRATCH, 400k steps = the FULL baseline budget
# (S5-RVT / RVT both train 400k). A FRESH OneCycle over the whole 400k horizon -- this is the run that
# yields the head-to-head-comparable number. It is deliberately NOT a resume of the 100k mid-run: that
# run's OneCycle already annealed to ~0 at step 100k, so resuming-and-extending it gives an undefined LR
# trajectory (the scheduler is reconstructed with the wrong total_steps). Start over instead.
#
# Thin wrapper over stage7_midrun_local.sh -- inherits EVERY OOM/resume/Hydra fix (single source of
# truth). Only the step budget + labels + run dir change (exported below). Per the terminal policy, the
# USER runs this.
#
#   bash stage7_fullrun_local.sh                 # fresh 400k run
#   bash stage7_fullrun_local.sh --cfg job       # GPU-free dry-run (compose config + exit)
#
# Expected wall-clock: PLAN FOR IT TO SPAN MORE THAN ONE NIGHT (~11-18h at the RAM-safe 2-worker
# throughput; confirm the real rate from the live tqdm step/s once it is warm). Launch DETACHED so a
# closed terminal can't SIGHUP it (tmux is not installed on GhostMachine):
#   setsid bash code/event_ssm/scripts/stage7_fullrun_local.sh >/dev/null 2>&1 &
# The tee'd results/stage7_fullrun/console_*.log still captures everything.
#
# RESUME (if it dies / you stop it): you MUST carry the same 400k budget or the restored OneCycle
# scheduler (total_steps=400000) won't match a default-100k reconstruction:
#   MAX_STEPS=400000 bash code/event_ssm/scripts/stage7_resume_local.sh   # auto-picks newest last_*.ckpt
set -euo pipefail
REPO=/home/ghost/Desktop/thesis-ssm-event-cameras

# --- only these differ from the mid-run; everything else inherits stage7_midrun_local.sh defaults ---
export MAX_STEPS=400000                  # full baseline budget (4x the mid-run)
export MAX_EPOCHS=100000                 # steps bind -> never a silent epoch truncation
export VAL_EVERY=20000                   # 20 full-val mAP points; overlays the mid-run curve at 40/60/80/100k
                                         #   (~20 vals x ~8 min ~= 2.5h overhead; raise to 40000 to halve it)
export GROUP_NAME=stage7_fullrun_mamba2  # offline wandb label
export RUNDIR="$REPO/results/stage7_fullrun"   # separate Hydra/.log dir from the mid-run
# BATCH=4 / PRECISION=bf16-mixed / NUM_WORKERS_TRAIN=2 / NUM_WORKERS_EVAL=1 inherit the VRAM/RAM-safe
# mid-run defaults. Bump workers (and add a ~32 GB swapfile first) only if `free -g` shows headroom.

exec bash "$REPO/code/event_ssm/scripts/stage7_midrun_local.sh" "$@"
