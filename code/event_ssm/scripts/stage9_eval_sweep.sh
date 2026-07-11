#!/usr/bin/env bash
# Stage-9 temporal-generalisation EVAL SWEEP.
# Evaluates BOTH models — EventSSM (ours, Mamba) and the S5-RVT baseline (S5-ViT) — on the rebuilt Gen1
# TEST set at every dt window {200,100,50,25,12} ms. Both go through the SAME stage9 preprocessing pipeline
# (data/gen1_stage9/preproc) so the resulting degradation curves are directly comparable.
#
# SAFE TO RUN ANY TIME (even mid-render):
#   * skips a rate whose representations are not fully rendered (<469 sequences) — re-run later to pick it up
#   * skips a (model,rate) whose log already has a test/AP — so re-running only does the new rates (idempotent)
# Per the terminal policy, the USER runs this. Foreground; ~10 evals x ~5-9 min ≈ 60-90 min when all rates ready.
#
#   bash code/event_ssm/scripts/stage9_eval_sweep.sh
#
# Output: per-(model,rate) logs + a final test/AP summary table under results/stage9/sweep/.
set -uo pipefail        # NOT -e: one failed eval must not abort the whole sweep

REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
SCRIPTS="$REPO/code/event_ssm/scripts"
PREPROC="$REPO/data/gen1_stage9/preproc"            # dir CONTAINING test/ (the rebuilt stage9 set)
OURS_CKPT="$REPO/external/ssms_event_cameras/RVT/RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt"
BASE_CKPT="$REPO/checkpoints/gen1_base.ckpt"        # S5-RVT baseline (47.7 mAP on canonical data)
OUT="$REPO/results/stage9/sweep"
mkdir -p "$OUT"

RATES=(200 100 50 25 12)                            # 0.25x 0.5x 1x 2x 4x  (window ms; grid stride fixed at 50ms)
NSEQ=469

ap_of() { grep 'test/AP' "$1" 2>/dev/null | grep -v 'test/AP_' | grep -oE '[0-9]+\.[0-9]+' | head -1; }
done_log() { [[ -s "$1" ]] && grep -q 'test/AP' "$1"; }   # log exists and contains a result

echo "================ Stage-9 eval sweep — preflight ================"
for V in "${RATES[@]}"; do
  n=$(find "$PREPROC/test" -path "*stacked_histogram_dt=${V}_nbins=10/event_representations.h5" 2>/dev/null | wc -l)
  printf "  dt=%-3s ms : %s/%s rendered\n" "$V" "$n" "$NSEQ"
done
echo "================================================================"

for V in "${RATES[@]}"; do
  EVR="stacked_histogram_dt=${V}_nbins=10"
  n=$(find "$PREPROC/test" -path "*${EVR}/event_representations.h5" 2>/dev/null | wc -l)
  if [[ "$n" -lt "$NSEQ" ]]; then
    echo "[skip] dt=${V} ms: only ${n}/${NSEQ} rendered — skipping (re-run sweep once it finishes)"
    continue
  fi

  # --- EventSSM (ours) — stage7 wrapper registers ResNetMamba; ckpt MUST precede the override ---
  OLOG="$OUT/eventssm_dt${V}.log"
  if done_log "$OLOG"; then
    echo "[done] EventSSM dt=${V} ms already evaluated (AP=$(ap_of "$OLOG")) — skipping"
  else
    echo "===== dt=${V} ms | EventSSM (ours) ====="
    DATASET="$PREPROC" bash "$SCRIPTS/stage7_test_eval_local.sh" "$OURS_CKPT" \
      "dataset.ev_repr_name='${EVR}'" 2>&1 | tee "$OLOG"
  fi

  # --- S5-RVT baseline — stage8 wrapper, stock S5-ViT (no register); ckpt MUST precede the override ---
  BLOG="$OUT/baseline_dt${V}.log"
  if done_log "$BLOG"; then
    echo "[done] S5-RVT baseline dt=${V} ms already evaluated (AP=$(ap_of "$BLOG")) — skipping"
  else
    echo "===== dt=${V} ms | S5-RVT baseline ====="
    DATASET="$PREPROC" bash "$SCRIPTS/stage8_baseline_eval_local.sh" "$BASE_CKPT" \
      "dataset.ev_repr_name='${EVR}'" 2>&1 | tee "$BLOG"
  fi
done

echo
echo "================ SUMMARY: test/AP (COCO mAP) by model x rate ================"
printf "%-9s %-10s %-12s %-12s\n" "mult" "dt(ms)" "EventSSM" "S5-RVT"
declare -A MULT=( [200]=0.25x [100]=0.5x [50]=1x [25]=2x [12]=4x )
for V in "${RATES[@]}"; do
  a=$(ap_of "$OUT/eventssm_dt${V}.log"); b=$(ap_of "$OUT/baseline_dt${V}.log")
  printf "%-9s %-10s %-12s %-12s\n" "${MULT[$V]}" "$V" "${a:-—}" "${b:-—}"
done
echo "============================================================================="
echo "logs: $OUT/   |   per-class car/ped AP is in each log (grep '\\[per-class\\]')"
echo "published RVT(ConvLSTM)/S5-RVT degradation numbers for the comparison table: stages/Stage_09_Temporal_Generalisation.md"
