#!/usr/bin/env bash
# Stage-9 REGIME-2 render — TRUE rate-change test sets (stride = accumulation window).
#
# The fixed-stride sweep (regime 1) varied only the per-frame event window; the model still stepped
# every 50 ms, so Δt-compensation had nothing to fix (falsified 2026-07-10: 30.1/38.4/29.4 at 4x).
# Regime 2 renders the paper's ACTUAL experiment: frames tile the stream gap-free at their natural
# cadence, so the model genuinely steps faster -> the SSM Δt-rescaling mechanism becomes testable.
#
#   rate  window=stride  extraction config             frames/seq  est. size
#   2x    25 ms          const_duration_40hz.yaml      ~2382       ~26 GB
#   10x   5 ms           const_duration_200hz.yaml     ~11910      ~35 GB   (paper's 200 Hz headline)
#
# (Stride must divide gen1's 250 ms label grid -> 4x/12 ms is impossible; 1x true-rate == the existing
#  dt=50 fixed-stride render, reused as the shared anchor.)
#
# Uses the TS_STEP_EV_REPR_MS env hook patched into preprocess_dataset.py (docs/patches/
# preprocess_full_stage9.patch). Output goes to a SEPARATE tree (preproc_tr/) so the validated
# regime-1 renders are untouched; representation names collide by design (stacked_histogram_dt=25...)
# but live under different parents.
#
# RESUMABLE: the preprocessor skips fully-rendered sequences and cleans in-progress files.
# Per the terminal policy the USER runs this, FOREGROUND (live "sequences" tqdm bar):
#   bash code/event_ssm/scripts/stage9_render_truerate.sh 2x     # ~1-3 h, do this first
#   bash code/event_ssm/scripts/stage9_render_truerate.sh 10x    # longer (writes ~10x frames)
#   bash code/event_ssm/scripts/stage9_render_truerate.sh        # both, sequentially
set -euo pipefail

REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
GENX="$REPO/external/ssms_event_cameras/RVT/scripts/genx"
RAW="$REPO/data/gen1_stage9/raw/detection_dataset_duration_60s_ratio_1.0"   # test/ has 470 .dat.h5; train/ val/ empty
OUT="$REPO/data/gen1_stage9/preproc_tr"
NP="${NP:-3}"          # render workers; 16 GB RAM box -> keep modest (OOM history)

declare -A EXTRACT=( [2x]=const_duration_40hz.yaml  [10x]=const_duration_200hz.yaml )
declare -A STRIDE=(  [2x]=25                        [10x]=5 )

PASSES=("$@"); [[ ${#PASSES[@]} -eq 0 ]] && PASSES=(2x 10x)
for P in "${PASSES[@]}"; do
  [[ -n "${EXTRACT[$P]:-}" ]] || { echo "unknown pass '$P' (use 2x and/or 10x)"; exit 1; }
done

n=$(find "$RAW/test" -name '*_td.dat.h5' 2>/dev/null | wc -l)
[[ "$n" -eq 470 ]] || { echo "ERROR: expected 470 .dat.h5 under $RAW/test, found $n"; exit 1; }
grep -q "TS_STEP_EV_REPR_MS" "$GENX/preprocess_dataset.py" || {
  echo "ERROR: stride hook missing — apply docs/patches/preprocess_full_stage9.patch first"; exit 1; }
mkdir -p "$OUT"
df -h "$REPO" | tail -1 | awk '{print "[disk] free: "$4}'

set +u; source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals; set -u
export PYTHONUNBUFFERED=1
cd "$GENX"    # conf_preprocess paths are relative to the genx dir

for P in "${PASSES[@]}"; do
  S="${STRIDE[$P]}"
  echo
  echo "================================================================================"
  echo "  RENDER $P — window=${S} ms, stride=${S} ms (TRUE rate; gap-free)   np=$NP"
  echo "================================================================================"
  TS_STEP_EV_REPR_MS="$S" python preprocess_dataset.py \
    "$RAW" "$OUT" \
    conf_preprocess/representation/stacked_hist.yaml \
    "conf_preprocess/extraction/frequencies/${EXTRACT[$P]}" \
    conf_preprocess/filter_gen1.yaml \
    -ds gen1 -np "$NP"
  # the python header must show BOTH the window (value: ms in the printed config) AND
  # "[stage9] ts_step_ev_repr_ms = ${S}" — that pair is the true-rate engagement proof.
  done_n=$(find "$OUT/test" -path "*stacked_histogram_dt=${S}_nbins=10/event_representations.h5" 2>/dev/null | wc -l)
  echo "[render $P] rendered sequences: ${done_n}/469"
done

echo
echo "DONE. Next: bash code/event_ssm/scripts/stage9_truerate_sweep.sh   (the regime-2 eval grid)"
