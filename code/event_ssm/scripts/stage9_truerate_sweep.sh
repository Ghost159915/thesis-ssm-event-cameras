#!/usr/bin/env bash
# Stage-9 REGIME-2 eval grid — TRUE rate-change test sets (stride = window, from stage9_render_truerate.sh).
#
# Runs the full 2x2 at each true rate: {S5-RVT baseline, EventSSM (ours)} x {no-comp, Δt-compensated}.
# In THIS regime the model genuinely steps faster, so the compensation convention is the original
# paper one: step_scale = window/50  (2x -> 0.5, 10x -> 0.1) — smaller Δt for faster stepping.
#
#   references: paper @200 Hz (=our 10x): S5+comp = 39.84, RVT(ConvLSTM, no knob) = 8.35
#               shared 1x anchor (window=stride=50): ours 46.2 / baseline 47.7 (reuse, no re-eval)
#
# Eval cost scales with frames/seq: 2x ~2382 (~30 min/eval), 10x ~11910 (~2.5 h/eval).
# Default order runs ALL FOUR 2x evals first (fast feedback), then the four 10x (overnight-friendly).
# Idempotent: skips any eval whose log already contains a test/AP; skips unrendered rates.
# Per the terminal policy the USER runs this, FOREGROUND:
#   bash code/event_ssm/scripts/stage9_truerate_sweep.sh          # both rates (25 then 5)
#   bash code/event_ssm/scripts/stage9_truerate_sweep.sh 25       # just 2x
#   bash code/event_ssm/scripts/stage9_truerate_sweep.sh 5        # just 10x
set -uo pipefail        # NOT -e: one failed eval must not abort the grid

REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
SCRIPTS="$REPO/code/event_ssm/scripts"
PREPROC="$REPO/data/gen1_stage9/preproc_tr"         # dir CONTAINING test/ (the TRUE-RATE renders)
OURS_CKPT="$REPO/external/ssms_event_cameras/RVT/RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt"
BASE_CKPT="$REPO/checkpoints/gen1_base.ckpt"
OUT="$REPO/results/stage9/sweep_tr"
mkdir -p "$OUT"

RATES=("$@"); [[ ${#RATES[@]} -eq 0 ]] && RATES=(25 5)
NSEQ=469
declare -A MULT=( [25]=2x [5]=10x )

ap_of()   { grep 'test/AP' "$1" 2>/dev/null | grep -v 'test/AP_' | grep -oE '[0-9]+\.[0-9]+' | head -1; }
done_log(){ [[ -s "$1" ]] && grep -q 'test/AP' "$1"; }
step_scale_of(){ python3 -c "print(f'{$1/50:.4f}')"; }   # window/50: the paper convention, correct in regime 2

echo "================ Stage-9 TRUE-RATE eval grid — preflight ================"
for V in "${RATES[@]}"; do
  n=$(find "$PREPROC/test" -path "*stacked_histogram_dt=${V}_nbins=10/event_representations.h5" 2>/dev/null | wc -l)
  printf "  %-4s window=stride=%-3s ms : step_scale(comp)=%-7s : %s/%s rendered\n" \
    "${MULT[$V]:-?}" "$V" "$(step_scale_of "$V")" "$n" "$NSEQ"
done
echo "=========================================================================="

run_eval() {  # $1 model-key  $2 dt  $3 step_scale ('' = no-comp)
  local K="$1" V="$2" SS="$3"
  local EVR="stacked_histogram_dt=${V}_nbins=10"
  local TAG LOG WRAP ENVX
  if [[ -z "$SS" ]]; then TAG="nc"; else TAG="ss${SS}"; fi
  LOG="$OUT/${K}_dt${V}_${TAG}.log"
  if done_log "$LOG"; then
    echo "[done] $K ${MULT[$V]} ${TAG} already evaluated (AP=$(ap_of "$LOG")) — skipping"
    return 0
  fi
  echo "===== ${MULT[$V]} | dt=stride=${V} ms | $K | ${TAG} ====="
  if [[ "$K" == "baseline" ]]; then
    WRAP="$SCRIPTS/stage8_baseline_eval_local.sh"; local CKPT="$BASE_CKPT"; ENVX="S5_STEP_SCALE"
  else
    WRAP="$SCRIPTS/stage7_test_eval_local.sh";     local CKPT="$OURS_CKPT"; ENVX="MAMBA_STEP_SCALE"
  fi
  # comp runs must show the engagement lines ([S5Block]/[MambaTemporalBlock] step_scale=...) in the log
  if [[ -z "$SS" ]]; then
    DATASET="$PREPROC" bash "$WRAP" "$CKPT" "dataset.ev_repr_name='${EVR}'" 2>&1 | tee "$LOG"
  else
    env "$ENVX=$SS" DATASET="$PREPROC" bash "$WRAP" "$CKPT" "dataset.ev_repr_name='${EVR}'" 2>&1 | tee "$LOG"
  fi
}

for V in "${RATES[@]}"; do
  EVR="stacked_histogram_dt=${V}_nbins=10"
  n=$(find "$PREPROC/test" -path "*${EVR}/event_representations.h5" 2>/dev/null | wc -l)
  if [[ "$n" -lt "$NSEQ" ]]; then
    echo "[skip] dt=${V} ms: only ${n}/${NSEQ} rendered — run stage9_render_truerate.sh first"
    continue
  fi
  SS=$(step_scale_of "$V")
  run_eval baseline "$V" ""       # S5-RVT, no compensation  (expect real degradation here)
  run_eval baseline "$V" "$SS"    # S5-RVT + step_scale      (paper mechanism; 10x target ≈ 39.8)
  run_eval eventssm "$V" ""       # EventSSM, no compensation (does Mamba's input-dependent Δt self-adapt?)
  run_eval eventssm "$V" "$SS"    # EventSSM + delta-scaling  (the Mamba analog)
done

echo
echo "================= SUMMARY: TRUE-RATE regime (stride = window) ================="
printf "%-5s %-9s %-12s %-12s %-12s %-12s\n" "mult" "dt(ms)" "S5 no-comp" "S5 comp" "ours no-comp" "ours comp"
for V in "${RATES[@]}"; do
  SS=$(step_scale_of "$V")
  printf "%-5s %-9s %-12s %-12s %-12s %-12s\n" "${MULT[$V]:-?}" "$V" \
    "$(ap_of "$OUT/baseline_dt${V}_nc.log")"  "$(ap_of "$OUT/baseline_dt${V}_ss${SS}.log")" \
    "$(ap_of "$OUT/eventssm_dt${V}_nc.log")"  "$(ap_of "$OUT/eventssm_dt${V}_ss${SS}.log")"
done
echo "==============================================================================="
echo "anchors (1x, shared with regime 1): ours 0.4620 / baseline 0.4769"
echo "paper @200 Hz (=10x): S5+comp 39.84 · RVT(ConvLSTM) 8.35   |   logs: $OUT/"
