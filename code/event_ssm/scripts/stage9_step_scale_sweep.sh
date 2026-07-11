#!/usr/bin/env bash
# Stage-9 COMPENSATED baseline sweep — the model-side half of the temporal-generalisation experiment.
#
# Re-evaluates the S5-RVT baseline on the rebuilt Gen1 TEST set at each dt window, this time with the
# S5 discretisation step rescaled per rate:  step_scale = test_window_ms / 50  (see s5_step_scale_hook.patch).
# The upstream repo never plumbs step_scale through detection, so our FIRST sweep (stage9_eval_sweep.sh)
# ran everything at the training Delta_t (step_scale=1.0) and S5 collapsed to ~38.4 mAP at 4x. This script
# engages the mechanism and is the "with-compensation" curve; the earlier sweep is the "no-compensation"
# baseline. Logs go to a SEPARATE dir so both curves survive.
#
# FALSIFIABLE GATE: with step_scale set, S5-RVT @4x (dt=12) must RECOVER from 38.4 toward the paper's ~40-44.
#   * 1x (dt=50) uses step_scale=1.0 and MUST reproduce ~47.7 exactly (proves the hook is inert at 1x).
#   * If 4x gets WORSE instead of better, the convention is inverted -> try step_scale = 50 / window.
#
# Per the terminal policy, the USER runs this (foreground).
#   Mini-gate first (ONE ~6-9 min eval):   bash code/event_ssm/scripts/stage9_step_scale_sweep.sh 12
#   Then the full compensated sweep:        bash code/event_ssm/scripts/stage9_step_scale_sweep.sh
#
# Idempotent: skips a (rate) whose _ss log already has a test/AP, and skips a rate not fully rendered.
set -uo pipefail        # NOT -e: one failed eval must not abort the sweep

REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
SCRIPTS="$REPO/code/event_ssm/scripts"
PREPROC="$REPO/data/gen1_stage9/preproc"            # dir CONTAINING test/ (the rebuilt stage9 set)
BASE_CKPT="$REPO/checkpoints/gen1_base.ckpt"        # S5-RVT baseline (47.7 mAP on canonical data)
OUT="$REPO/results/stage9/sweep_ss"                 # SEPARATE from the no-compensation sweep/ dir
NOCOMP="$REPO/results/stage9/sweep"                 # where the uncompensated baseline logs live (for the delta col)
mkdir -p "$OUT"

RATES=("$@"); [[ ${#RATES[@]} -eq 0 ]] && RATES=(200 100 50 25 12)   # default: all; or pass a subset for the gate
NSEQ=469
declare -A MULT=( [200]=0.25x [100]=0.5x [50]=1x [25]=2x [12]=4x )

ap_of()   { grep 'test/AP' "$1" 2>/dev/null | grep -v 'test/AP_' | grep -oE '[0-9]+\.[0-9]+' | head -1; }
done_log(){ [[ -s "$1" ]] && grep -q 'test/AP' "$1"; }
step_scale_of(){ python3 -c "print(f'{$1/50:.4f}')"; }   # convention: window_ms / 50

echo "================ Stage-9 COMPENSATED baseline sweep — preflight ================"
for V in "${RATES[@]}"; do
  n=$(find "$PREPROC/test" -path "*stacked_histogram_dt=${V}_nbins=10/event_representations.h5" 2>/dev/null | wc -l)
  printf "  %-6s dt=%-3s ms : step_scale=%-7s : %s/%s rendered\n" "${MULT[$V]:-?}" "$V" "$(step_scale_of "$V")" "$n" "$NSEQ"
done
echo "================================================================================"

for V in "${RATES[@]}"; do
  EVR="stacked_histogram_dt=${V}_nbins=10"
  SS=$(step_scale_of "$V")
  n=$(find "$PREPROC/test" -path "*${EVR}/event_representations.h5" 2>/dev/null | wc -l)
  if [[ "$n" -lt "$NSEQ" ]]; then
    echo "[skip] dt=${V} ms: only ${n}/${NSEQ} rendered — skipping (re-run once it finishes)"
    continue
  fi

  BLOG="$OUT/baseline_dt${V}_ss${SS}.log"
  if done_log "$BLOG"; then
    echo "[done] S5-RVT ${MULT[$V]} (dt=${V}, step_scale=${SS}) already evaluated (AP=$(ap_of "$BLOG")) — skipping"
    continue
  fi

  echo "===== ${MULT[$V]} | dt=${V} ms | step_scale=${SS} | S5-RVT baseline ====="
  S5_STEP_SCALE="$SS" DATASET="$PREPROC" bash "$SCRIPTS/stage8_baseline_eval_local.sh" "$BASE_CKPT" \
    "dataset.ev_repr_name='${EVR}'" 2>&1 | tee "$BLOG"
done

echo
echo "============ SUMMARY: S5-RVT baseline — no-comp vs step_scale-compensated ============"
printf "%-7s %-8s %-12s %-14s %-14s\n" "mult" "dt(ms)" "step_scale" "no-comp AP" "comp AP"
for V in "${RATES[@]}"; do
  SS=$(step_scale_of "$V")
  comp=$(ap_of "$OUT/baseline_dt${V}_ss${SS}.log")
  nocomp=$(ap_of "$NOCOMP/baseline_dt${V}.log")
  printf "%-7s %-8s %-12s %-14s %-14s\n" "${MULT[$V]:-?}" "$V" "$SS" "${nocomp:-—}" "${comp:-—}"
done
echo "======================================================================================"
echo "GATE CHECK: comp AP at 4x (dt=12) should be > no-comp 38.4 and approach ~40-44."
echo "            comp AP at 1x (dt=50) should equal ~47.7 (hook inert at step_scale=1.0)."
echo "logs: $OUT/"
