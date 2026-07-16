#!/usr/bin/env bash
# Stage-16 Slice D — GT-vs-prediction overlay driver (the qualitative large-car visuals).
#
# For each of the 12 large-car test recordings (results/stage16/large_car_recs.txt), renders TWO
# GT-vs-pred videos per recording (one per model: EventSSM, PureSSM) through the shared EventCV
# renderer (stage9_render_event_video.py --boxes --pred ... --smooth --montage), then stitches
# every per-rec/per-model 8-frame montage into one results/stage16/large_car_contact_sheet.png.
#
# PREREQUISITE (GPU — run first, NOT by this script):
#   python code/event_ssm/scripts/stage16_dump_predictions.py --model eventssm
#   python code/event_ssm/scripts/stage16_dump_predictions.py --model puressm
# -> results/stage16/preds/{eventssm,puressm}/<rec>.npy
#
# This script itself is CPU-only (cv2 + numpy rendering — no model, no GPU touch). A recording
# whose prediction dump is missing is skipped with a warning rather than failing the whole run,
# so it is safe to run again after dumping the remaining model.
set -euo pipefail

REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
RAW="$REPO/data/gen1_stage9/raw/detection_dataset_duration_60s_ratio_1.0/test"
RECS_FILE="$REPO/results/stage16/large_car_recs.txt"
OUT="$REPO/results/stage16"
PREDS="$OUT/preds"
RENDER="$REPO/code/event_ssm/scripts/stage9_render_event_video.py"
MODELS=(eventssm puressm)

[[ -f "$RECS_FILE" ]] || {
  echo "ERROR: $RECS_FILE not found (run stage16_select_large_car_recs.py first)"; exit 1; }

set +u; source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals; set -u
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

n_recs=0
n_rendered=0
while IFS= read -r rec; do
  [[ -z "$rec" ]] && continue
  n_recs=$((n_recs + 1))
  dat="$RAW/${rec}_td.dat.h5"
  [[ -f "$dat" ]] || { echo "ERROR: missing raw recording $dat"; exit 1; }

  for model in "${MODELS[@]}"; do
    pred="$PREDS/$model/${rec}.npy"
    if [[ ! -f "$pred" ]]; then
      echo "[skip] $rec / $model — no prediction dump at $pred " \
           "(run stage16_dump_predictions.py --model $model first)"
      continue
    fi
    out_mp4="$OUT/${rec}__${model}_gt_vs_pred.mp4"
    echo
    echo "=== $rec / $model ==="
    python "$RENDER" "$dat" \
      --boxes --pred "$pred" --smooth --montage \
      --frame-dt-ms 33 --fps 30 --upscale 2 \
      --out "$out_mp4"
    n_rendered=$((n_rendered + 1))
  done
done < "$RECS_FILE"

echo
echo "[stitch] building the large-car contact sheet from the per-rec/per-model montages ..."
python - "$OUT" "$RECS_FILE" <<'PYEOF'
import sys
from pathlib import Path

import cv2
import numpy as np

out_dir = Path(sys.argv[1])
recs = [ln.strip() for ln in Path(sys.argv[2]).read_text().splitlines() if ln.strip()]
models = ["eventssm", "puressm"]

tiles, labels = [], []
for rec in recs:
    for model in models:
        mp = out_dir / f"{rec}__{model}_gt_vs_pred_montage.png"
        if mp.exists():
            img = cv2.imread(str(mp))
            if img is not None:
                tiles.append(img)
                labels.append(f"{rec} [{model}]")

if not tiles:
    print("[stitch] no per-rec montages found yet (dump predictions + rerun this script first) "
          "-- skipping contact sheet")
    sys.exit(0)

w = min(t.shape[1] for t in tiles)
h = min(t.shape[0] for t in tiles)
labeled = []
for img, lbl in zip(tiles, labels):
    img = cv2.resize(img, (w, h)).copy()
    cv2.putText(img, lbl, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    labeled.append(img)

sheet = np.vstack(labeled)
sheet_path = out_dir / "large_car_contact_sheet.png"
cv2.imwrite(str(sheet_path), sheet)
print(f"[stitch] wrote {sheet_path} ({len(labeled)} tiles, {sheet.shape[1]}x{sheet.shape[0]})")
PYEOF

echo
echo "DONE. $n_recs recordings scanned, $n_rendered videos rendered this run."
echo "Outputs: $OUT/<rec>__<model>_gt_vs_pred.mp4 (+ _montage.png), $OUT/large_car_contact_sheet.png"
