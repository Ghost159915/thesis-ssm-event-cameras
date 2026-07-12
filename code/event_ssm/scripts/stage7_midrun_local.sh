#!/usr/bin/env bash
# Stage-7a LOCAL MID-RUN (RTX 5070 Ti). Full Gen1 train, schedule-completing overnight run for a
# trustworthy "is this architecture competitive?" signal (25% of the 400k baseline budget on full data).
# Reuses RVT train.py UNMODIFIED via the stage6_train.py launcher (registers the Mamba-2 ResNetMamba
# backbone first). bf16 autocast, no GradScaler (ISSUE-09). wandb offline. Validation-driven
# checkpoints + rough val-mAP every VAL_EVERY steps. Resumable: set STAGE7_RESUME=/abs/last...ckpt.
# Per the terminal policy, the USER runs this. See STAGE7_MIDRUN_RUN.md.
#
# Offline GPU-free dry-run (compose config + exit):  bash stage7_midrun_local.sh --cfg job
set -euo pipefail

set +u                                               # conda's activate.d (cuda-nvcc) references unbound vars
source "${CONDA_SH:-/home/ghost/miniforge3/etc/profile.d/conda.sh}" && conda activate events_signals
set -u
export WANDB_MODE=offline                            # RVT wandb cfg has no 'mode' key -> use the env var
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation (real seq_len=21 is VRAM-heavy)
export PYTHONUNBUFFERED=1                             # flush stdout/stderr live -> the tqdm step/loss bar
                                                     # streams to the terminal + tee'd log (no buffering freeze)
REPO="${REPO:-/home/ghost/Desktop/thesis-ssm-event-cameras}"
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
cd "$REPO/external/ssms_event_cameras/RVT"           # hydra config_path="config" is relative to train.py

# ---------------- mid-run knobs (edit freely) ----------------
# Every knob is env-overridable (`: "${VAR:=default}"`) so thin wrappers (e.g. stage7_fullrun_local.sh)
# can change the budget/labels by exporting vars before exec'ing this script -- the OOM/resume/Hydra
# fixes below stay the single source of truth. Run directly with no env set => the mid-run defaults.
DATASET="${DATASET:-$REPO/data/gen1_raw/gen1}"   # FULL train (1458 rec) + full val/test
: "${MAX_STEPS:=100000}"             # 25% of baseline; OneCycle completes overnight even at ~2.8 it/s
: "${MAX_EPOCHS:=10000}"             # steps bind -> never a silent 1-epoch truncation
: "${VAL_EVERY:=20000}"              # FULL val + checkpoint every 20k steps (5 mAP points; ~38 min total val overhead)
: "${VAL_FRAC:=1.0}"                 # full val (clean mAP). Gen1 val is an IterableDataset -> limit_val_batches
                                     # MUST be 1.0 or an INT (num batches); a FRACTION (e.g. 0.25) is rejected by Lightning
: "${BATCH:=4}"                      # VRAM-safe at seq_len=21 on 16 GB
: "${PRECISION:=bf16-mixed}"         # ISSUE-09; fp32 fallback: PRECISION=32
: "${NUM_WORKERS_TRAIN:=2}"          # HOST-RAM safe on 16 GB. The default 6 (each worker ~3.6 GB RSS)
: "${NUM_WORKERS_EVAL:=1}"           #   exhausted system RAM -> OOM-killer killed the run 2026-06-17.
                                     #   Drop to 1/1 if it still OOMs; raise once a big swapfile exists.
: "${GROUP_NAME:=stage7_midrun_mamba2}"     # offline wandb label
: "${RUNDIR:=$REPO/results/stage7_midrun}"  # Hydra .hydra/ + train.log ONLY (hydra chdir=False in 1.3) --
                                     # CHECKPOINTS go to external/.../RVT/RVT/<runid>/checkpoints/ (logger-relative to cwd)
# -------------------------------------------------------------

# Knobs are visible to the sourced builder because it runs in this same shell.
source "$REPO/code/event_ssm/scripts/stage6_overrides.sh"   # builds STAGE6_OVERRIDES from the knobs above

# Optional offline full-state resume: STAGE7_RESUME=/abs/path/to/last_epoch=...-step=....ckpt
# (artifact_name just needs to be non-null; artifact_local_file makes get_ckpt_path load from disk;
#  resume_only_weights stays False -> optimizer/scheduler/global-step are restored, schedule continues.)
RESUME_OVERRIDES=()
if [[ -n "${STAGE7_RESUME:-}" ]]; then
  # Hydra's override grammar treats '=' as the key/value separator; the ckpt filename contains '='
  # (last_epoch=...-step=...), so the path value MUST be single-quoted for Hydra (it strips the quotes
  # during parse). Without this, Hydra aborts with "mismatched input '=' expecting <EOF>".
  RESUME_OVERRIDES+=( wandb.artifact_name=resume "wandb.artifact_local_file='$STAGE7_RESUME'" )
  echo "[stage7] resuming full training state from: $STAGE7_RESUME"
fi

# Mirror ALL console output (RVT/Lightning print to stdout, NOT to Hydra's train.log) to a timestamped
# file so a closed window / crash leaves a readable record. pipefail (set above) preserves python's exit
# code through the tee. TIP: launch inside tmux/screen so a SIGHUP from a closed terminal can't kill it.
mkdir -p "$RUNDIR"
LOGFILE="$RUNDIR/console_$(date +%Y%m%d_%H%M%S).log"
echo "[stage7] console log -> $LOGFILE"
PY_CMD=( python "$REPO/code/event_ssm/scripts/stage6_train.py"
         "${STAGE6_OVERRIDES[@]}" hydra.run.dir="$RUNDIR" "${RESUME_OVERRIDES[@]}" "$@" )
# LIVE step/loss bar: Lightning's tqdm bar uses '\r' and goes SILENT when piped to `tee` (tqdm sees a
# non-TTY and stops the live redraw -> the terminal looks "stuck" at the warnings while training runs
# fine; PYTHONUNBUFFERED does NOT fix this -- it's TTY detection, not buffering). When launched
# interactively, wrap python in `script`, which gives it a pseudo-TTY so the bar renders LIVE in the
# terminal; -f flushes the same bytes into $LOGFILE, -e propagates python's exit code, -q hides the
# banner. Non-interactive (backgrounded / piped to a file): no PTY needed -> plain tee fallback.
if [[ -t 1 ]] && command -v script >/dev/null 2>&1; then
  script -q -e -f -c "$(printf '%q ' "${PY_CMD[@]}")" "$LOGFILE"
else
  "${PY_CMD[@]}" 2>&1 | tee "$LOGFILE"
fi
