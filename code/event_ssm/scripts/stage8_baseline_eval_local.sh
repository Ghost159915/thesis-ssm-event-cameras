#!/usr/bin/env bash
# Stage-8 BASELINE (S5-RVT / S5-ViT) Gen1 TEST-set eval with per-class AP. Reproduces the MVP eval that
# scored 47.7 (VALIDATION_QUICKSTART.md) AND now prints per-class car/pedestrian AP, so EventSSMDetector
# and the baseline are compared on the SAME evaluator. Runs the stock S5-ViT (stage8_baseline_eval.py
# does setup_paths but NOT register -> no Mamba backbone). Per the terminal policy, the USER runs this.
#
#   bash stage8_baseline_eval_local.sh                 # eval checkpoints/gen1_base.ckpt (default)
#   bash stage8_baseline_eval_local.sh /abs/other.ckpt
#   bash stage8_baseline_eval_local.sh --cfg job       # GPU-free dry-run
set -euo pipefail

set +u
source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1            # PyTorch 2.6+ refuses to unpickle the Lightning ckpt otherwise
export PYTHONUNBUFFERED=1
REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
ORIG_PWD="$PWD"                                      # resolve a relative ckpt arg before the cd
cd "$REPO/external/ssms_event_cameras/RVT"          # hydra config_path="config" is relative to validation.py

# ---------------- knobs ----------------
DATASET="${DATASET:-$REPO/data/gen1_raw/gen1}"      # dir CONTAINING test/ (NOT test/ itself)
DEFAULT_CKPT="$REPO/checkpoints/gen1_base.ckpt"     # the S5-RVT baseline checkpoint (47.7 mAP)
CKPT="${1:-${CKPT:-$DEFAULT_CKPT}}"
[[ "$CKPT" == -* ]] && CKPT="$DEFAULT_CKPT"
[[ "$CKPT" != -* && "$CKPT" != /* ]] && CKPT="$ORIG_PWD/$CKPT"   # relative -> absolute
BATCH="${BATCH:-8}"                                 # baseline ViT is lighter at eval; MVP used 8
# ---------------------------------------

PASSTHRU=("$@"); [[ ${#PASSTHRU[@]} -ge 1 && "${PASSTHRU[0]}" != -* ]] && PASSTHRU=("${PASSTHRU[@]:1}")
if [[ "$CKPT" != -* && ! -f "$CKPT" ]]; then
  echo "[stage8-baseline] ERROR: checkpoint not found: $CKPT" >&2; exit 1
fi

RESULTS="$REPO/results/stage8_baseline_eval"
mkdir -p "$RESULTS"
OUT="$RESULTS/baseline_test_eval_$(date +%Y%m%d_%H%M%S).log"
echo "[stage8-baseline] ckpt:    $CKPT"
echo "[stage8-baseline] dataset: $DATASET"
echo "[stage8-baseline] log ->   $OUT"

# Mirrors the VALIDATION_QUICKSTART MVP command (base experiment, conf 0.001, default precision=16) so the
# aggregate AP reproduces 47.7; the per-class patch adds car/ped AP to stderr (merged via 2>&1 -> tee).
python "$REPO/code/event_ssm/scripts/stage8_baseline_eval.py" \
  dataset=gen1 \
  dataset.path="$DATASET" \
  +experiment/gen1=base.yaml \
  "checkpoint='$CKPT'" \
  use_test_set=1 \
  hardware.gpus=0 \
  batch_size.eval="$BATCH" \
  model.postprocess.confidence_threshold=0.001 \
  "${PASSTHRU[@]}" 2>&1 | tee "$OUT"
