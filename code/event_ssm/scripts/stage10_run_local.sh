#!/usr/bin/env bash
# Stage-10 efficiency benchmark launcher. REFUSES to run unless the GPU is idle — benchmark numbers
# measured on a contended GPU are garbage (spec §6). Usage:
#   bash code/event_ssm/scripts/stage10_run_local.sh --smoke     # <2 min wiring check
#   bash code/event_ssm/scripts/stage10_run_local.sh             # the real ~20-30 min run
set -euo pipefail

REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
NVSMI="${NVSMI:-nvidia-smi}"   # override with a stub for guard tests

read -r UTIL MEM <<<"$($NVSMI --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')"
if ! [[ "${UTIL:-}" =~ ^[0-9]+$ ]] || ! [[ "${MEM:-}" =~ ^[0-9]+$ ]]; then
  echo "[stage10] ABORT: could not parse GPU state from nvidia-smi (util='${UTIL:-}' mem='${MEM:-}'). Driver hiccup? Fail-closed." >&2
  exit 1
fi
if (( UTIL >= 10 )) || (( MEM >= 1500 )); then
  echo "[stage10] ABORT: GPU not idle (util=${UTIL}%, mem=${MEM} MiB; need <10% and <1500 MiB)." >&2
  echo "[stage10] Wait for Stage-9 sweeps/renders to finish, then re-run." >&2
  exit 1
fi
echo "[stage10] GPU idle (util=${UTIL}%, mem=${MEM} MiB) — proceeding."

set +u; source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals; set -u
# Contamination guard (Important-2): the Stage-9 Delta_t hooks (MambaTemporalBlock.step_scale /
# baseline S5Block.step_scale, docs/patches/README.md) read MAMBA_STEP_SCALE / S5_STEP_SCALE ONCE
# at model construction. A leftover export from a Stage-9 sweep shell sourced into this one would
# silently construct a Delta_t-rescaled model here and Stage-10 would benchmark the wrong thing.
unset MAMBA_STEP_SCALE S5_STEP_SCALE
# Same for the training monitors (forward hooks that print and host-sync inside timed forwards) and the Stage-22
# eval-only arm override: never wanted in a benchmark.
unset SPIKING_MONITOR PURESSM_MONITOR SPIKING_ALLOW_ARM_OVERRIDE
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"

OUT="$REPO/results/stage10"; mkdir -p "$OUT"
LOG="$OUT/console_$(date +%Y%m%d_%H%M%S).log"
echo "[stage10] log -> $LOG"
python "$REPO/code/event_ssm/scripts/stage10_benchmark.py" "$@" 2>&1 | tee "$LOG"
echo "[stage10] next: python $REPO/code/event_ssm/scripts/stage10_report.py"
