#!/usr/bin/env bash
# =============================================================================
# Stage 9 prep - raw Gen1 TEST events ready for re-rendering at multiple dt.
#
#   1) extract the two Prophesee OG archives (test_a.7z + test_b.7z) -> raw .dat
#   2) create empty train/ val/ (the RVT pre-processor asserts all 3 splits exist)
#   3) convert every *_td.dat  ->  *_td.dat.h5  (the format H5Reader consumes)
#
# Run it FOREGROUND so you see live per-file progress:
#       bash code/event_ssm/scripts/stage9_prep_test.sh
#
# Resumable: 7z skips already-extracted files (-aos); the converter skips any
# recording whose .dat.h5 already exists. Safe to re-run if interrupted.
# =============================================================================
set -euo pipefail

REPO="/home/ghost/Desktop/thesis-ssm-event-cameras"
cd "$REPO"
ENV=events_signals
RAW="$REPO/data/gen1_stage9/raw"
DDR="$RAW/detection_dataset_duration_60s_ratio_1.0"
TEST="$DDR/test"

echo "==> [1/3] extracting test_a.7z + test_b.7z  ->  $RAW"
mkdir -p "$RAW"
7z x data/test_a.7z -o"$RAW" -aos
7z x data/test_b.7z -o"$RAW" -aos

echo "==> creating empty train/ val/ (pre-processor asserts they exist)"
mkdir -p "$DDR/train" "$DDR/val"

n_dat=$(find "$TEST" -name '*_td.dat'   | wc -l)
n_box=$(find "$TEST" -name '*_bbox.npy' | wc -l)
echo "==> [2/3] extracted: $n_dat .dat  +  $n_box _bbox.npy  (expect 470 each)"

echo "==> [3/3] converting .dat -> .dat.h5 (live progress below)"
conda run --no-capture-output -n "$ENV" \
  python code/event_ssm/scripts/stage9_dat_to_h5.py "$TEST"

echo ""
echo "==> DONE. Raw test events ready at:"
echo "      $TEST"
echo "    Disk note: .dat are kept. After the re-render step is verified you can"
echo "    reclaim ~142 GB with:  rm $TEST/*.dat"
