#!/usr/bin/env bash
# Stage 22, item 2 — post-hoc firing-threshold sweep for ONE trained SpikingSSM checkpoint (IDLE GPU ONLY).
#
#   bash stage22_threshold_sweep_local.sh <ckpt> [threshold ...]      default thresholds: 0.5 0.75 1.0 1.5 2.0 3.0
#
# Per threshold T (string kept verbatim in the file names):
#   1. preflight (CPU, `--cfg job`): the composed eval config must carry threshold = T, else stop;
#   2. the Stage-21 test eval at T  -> results/stage22/threshold_sweep/<tag>/eval_thr<T>.txt   (~14 min)
#   3. firing rates at T            -> results/stage22/threshold_sweep/<tag>/rates_thr<T>.json (~minutes)
# Idempotent: a threshold whose log already has a test/AP, or whose rates file exists, is skipped. One failed
# threshold does not abort the others. T = 1.0 re-evaluates the trained operating point: its AP must equal the
# canonical Stage-21 test AP (a consistency check on the override path).
# Then: python -m event_ssm.eval.threshold_sweep --sweep <arm>=<dir> ... --out <stem>
set -uo pipefail                                      # NOT -e: one failed threshold must not stop the sweep

set +u
source "${CONDA_SH:-/home/ghost/miniforge3/etc/profile.d/conda.sh}" && conda activate events_signals
set -u
REPO="${REPO:-/home/ghost/Desktop/thesis-ssm-event-cameras}"
SCRIPTS="$REPO/code/event_ssm/scripts"
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [[ $# -lt 1 ]]; then
  echo "usage: bash stage22_threshold_sweep_local.sh <ckpt> [threshold ...]" >&2
  exit 2
fi
CKPT="$1"; shift
[[ "$CKPT" != /* ]] && CKPT="$PWD/$CKPT"
[[ -f "$CKPT" ]] || { echo "[stage22-thr] ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
if [[ $# -gt 0 ]]; then THRS=("$@"); else THRS=(0.5 0.75 1.0 1.5 2.0 3.0); fi

TAG="$(python -m event_ssm.integration.spiking_ckpt tag "$CKPT")" || exit 1
OUT="$REPO/results/stage22/threshold_sweep/$TAG"
mkdir -p "$OUT"
echo "[stage22-thr] ckpt $CKPT  arm $TAG  thresholds ${THRS[*]}  -> $OUT"

ap_of() { grep 'test/AP' "$1" 2>/dev/null | grep -v 'test/AP_' | grep -oE '[0-9]+\.[0-9]+' | head -1; }

for T in "${THRS[@]}"; do
  LOG="$OUT/eval_thr${T}.txt"
  if [[ -n "$(ap_of "$LOG")" ]]; then
    echo "[stage22-thr] θ=$T eval: done ($(ap_of "$LOG")), skipping"
  else
    # 1. preflight: the composed config must carry exactly this threshold (no duplicate-override surprises).
    #    `--cfg job` goes LAST: the launcher appends these after its own overrides, and Hydra's parser rejects
    #    overrides that follow an option ("unrecognized arguments").
    CFG="$(SPIKING_ALLOW_ARM_OVERRIDE=1 bash "$SCRIPTS/stage21_spikingssm_test_eval_local.sh" "$CKPT" \
           "model.backbone.spiking.threshold=$T" --cfg job 2>/dev/null)"
    GOT="$(awk '/^[[:space:]]*spiking:/{f=1} f && /^[[:space:]]+threshold:/{print $2; exit}' <<< "$CFG")"
    if ! python -c "import sys; sys.exit(0 if abs(float('$GOT' or 'nan') - float('$T')) < 1e-9 else 1)"; then
      echo "[stage22-thr] θ=$T PREFLIGHT FAILED: composed threshold '${GOT}' != $T — skipping this threshold" >&2
      continue
    fi
    # 2. the test eval at T
    echo "[stage22-thr] θ=$T eval -> $LOG"
    SPIKING_ALLOW_ARM_OVERRIDE=1 bash "$SCRIPTS/stage21_spikingssm_test_eval_local.sh" "$CKPT" \
      "model.backbone.spiking.threshold=$T" > "$LOG" 2>&1
    echo "[stage22-thr] θ=$T test/AP = $(ap_of "$LOG")"
  fi
  # 3. firing rates at T
  if [[ -f "$OUT/rates_thr${T}.json" ]]; then
    echo "[stage22-thr] θ=$T rates: done, skipping"
  else
    python "$SCRIPTS/stage22_threshold_rates.py" --ckpt "$CKPT" --thresholds "$T" --out "$OUT"
  fi
done

echo "================ θ sweep summary ($TAG) ================"
for T in "${THRS[@]}"; do
  R="$(python -c "import json; print(round(json.load(open('$OUT/rates_thr$T.json'))['weighted_rate'], 4))" 2>/dev/null)"
  printf "  θ=%-6s test/AP=%-8s weighted_rate=%s\n" "$T" "$(ap_of "$OUT/eval_thr${T}.txt")" "${R:-?}"
done
