#!/usr/bin/env bash
# Fact-check decision D1: RVT (ConvLSTM, RVT-B) Gen1 TEST-set eval, same recipe as the S5-RVT baseline eval
# (stage8_baseline_eval_local.sh = RVT's own published validation recipe: base experiment, conf 0.001, default
# precision, batch 8). The backbone is RVT's original ConvLSTM code, vendored in event_ssm/baselines/. Per the
# terminal policy the USER runs this on an idle GPU.
#
#   bash stage9_rvt_lstm_eval_local.sh                     # 1x: canonical dt=50 test set (expect ~47.2, RVT paper)
#   DATASET=$REPO/data/gen1_stage9/preproc_tr \
#     bash stage9_rvt_lstm_eval_local.sh "" "dataset.ev_repr_name='stacked_histogram_dt=5_nbins=10'"   # true 10x
set -euo pipefail
set +u
source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
unset MAMBA_STEP_SCALE S5_STEP_SCALE                 # Stage-9 inference-time hooks: never active here
REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
cd "$REPO/external/ssms_event_cameras/RVT"          # hydra config_path="config" is relative to validation.py

DATASET="${DATASET:-$REPO/data/gen1_raw/gen1}"      # dir CONTAINING test/
DEFAULT_CKPT="$REPO/checkpoints/rvt-b-gen1.ckpt"
CKPT="${1:-}"; [[ -z "$CKPT" ]] && CKPT="$DEFAULT_CKPT"
BATCH="${BATCH:-8}"
shift || true
[[ -f "$CKPT" ]] || { echo "[rvt-lstm] ERROR: checkpoint not found: $CKPT" >&2; exit 1; }

RESULTS="$REPO/results/stage9_rvt_lstm"; mkdir -p "$RESULTS"
TAG="$(basename "$DATASET")"; for a in "$@"; do [[ "$a" == *ev_repr_name* ]] && TAG="${TAG}_$(echo "$a" | grep -o 'dt=[0-9]*')"; done
OUT="$RESULTS/rvt_lstm_${TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "[rvt-lstm] ckpt:    $CKPT"; echo "[rvt-lstm] dataset: $DATASET"; echo "[rvt-lstm] log ->   $OUT"
python "$REPO/code/event_ssm/scripts/stage9_rvt_lstm_eval.py" \
  dataset=gen1 \
  dataset.path="$DATASET" \
  +experiment/gen1=base.yaml \
  "checkpoint='$CKPT'" \
  use_test_set=1 \
  hardware.gpus=0 \
  batch_size.eval="$BATCH" \
  model.postprocess.confidence_threshold=0.001 \
  "$@" 2>&1 | tee "$OUT"
