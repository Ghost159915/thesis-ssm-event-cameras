#!/usr/bin/env bash
# Stage-9 COMPENSATED EventSSM (ours, Mamba) sweep — the second half of the with-compensation curve.
#
# Mirrors stage9_step_scale_sweep.sh (S5-RVT baseline) for OUR model: re-evaluates the Stage-8
# EventSSM checkpoint on the rebuilt Gen1 TEST set at each dt window with the selective-scan
# Delta_t rescaled per rate via MAMBA_STEP_SCALE = test_window_ms / 50 (post-softplus delta scaling
# in event_ssm/temporal/_scan.py — the Mamba analog of S5's step_scale; mamba-ssm has no native knob).
# The earlier stage9_eval_sweep.sh results remain the "no-compensation" curve; logs here go to the
# SAME sweep_ss/ dir as the compensated baseline so the 4-curve plot reads one no-comp dir + one comp dir.
#
# RUN ONLY AFTER the parity tests pass:
#   pytest code/event_ssm/tests/test_step_scale_parity.py -v      (needs idle-ish GPU, ~seconds)
# Then (per the terminal policy, the USER runs this; foreground; 5 evals x ~6-9 min):
#   bash code/event_ssm/scripts/stage9_mamba_scale_sweep.sh          # all rates
#   bash code/event_ssm/scripts/stage9_mamba_scale_sweep.sh 12       # single rate (e.g. 4x first)
#
# Idempotent: skips a rate whose _ss log already has a test/AP; skips rates not fully rendered.
set -uo pipefail        # NOT -e: one failed eval must not abort the sweep

REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
SCRIPTS="$REPO/code/event_ssm/scripts"
PREPROC="$REPO/data/gen1_stage9/preproc"            # dir CONTAINING test/ (the rebuilt stage9 set)
OURS_CKPT="$REPO/external/ssms_event_cameras/RVT/RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt"
OUT="$REPO/results/stage9/sweep_ss"                 # shared with the compensated-baseline sweep
NOCOMP="$REPO/results/stage9/sweep"                 # uncompensated logs (for the summary delta col)
mkdir -p "$OUT"

RATES=("$@"); [[ ${#RATES[@]} -eq 0 ]] && RATES=(200 100 50 25 12)
NSEQ=469
declare -A MULT=( [200]=0.25x [100]=0.5x [50]=1x [25]=2x [12]=4x )

ap_of()   { grep 'test/AP' "$1" 2>/dev/null | grep -v 'test/AP_' | grep -oE '[0-9]+\.[0-9]+' | head -1; }
done_log(){ [[ -s "$1" ]] && grep -q 'test/AP' "$1"; }
step_scale_of(){ python3 -c "print(f'{$1/50:.4f}')"; }   # same convention as the S5 sweep

echo "================ Stage-9 COMPENSATED EventSSM sweep — preflight ================"
for V in "${RATES[@]}"; do
  n=$(find "$PREPROC/test" -path "*stacked_histogram_dt=${V}_nbins=10/event_representations.h5" 2>/dev/null | wc -l)
  printf "  %-6s dt=%-3s ms : MAMBA_STEP_SCALE=%-7s : %s/%s rendered\n" "${MULT[$V]:-?}" "$V" "$(step_scale_of "$V")" "$n" "$NSEQ"
done
echo "================================================================================"

for V in "${RATES[@]}"; do
  EVR="stacked_histogram_dt=${V}_nbins=10"
  SS=$(step_scale_of "$V")
  n=$(find "$PREPROC/test" -path "*${EVR}/event_representations.h5" 2>/dev/null | wc -l)
  if [[ "$n" -lt "$NSEQ" ]]; then
    echo "[skip] dt=${V} ms: only ${n}/${NSEQ} rendered — skipping"
    continue
  fi

  OLOG="$OUT/eventssm_dt${V}_ss${SS}.log"
  if done_log "$OLOG"; then
    echo "[done] EventSSM ${MULT[$V]} (dt=${V}, step_scale=${SS}) already evaluated (AP=$(ap_of "$OLOG")) — skipping"
    continue
  fi

  echo "===== ${MULT[$V]} | dt=${V} ms | MAMBA_STEP_SCALE=${SS} | EventSSM (ours) ====="
  # sanity: the eval log MUST contain '[MambaTemporalBlock] Stage-9: step_scale=...' lines
  # (except at 1x where the hook is silent by design) — proof the knob engaged in-process.
  MAMBA_STEP_SCALE="$SS" DATASET="$PREPROC" bash "$SCRIPTS/stage7_test_eval_local.sh" "$OURS_CKPT" \
    "dataset.ev_repr_name='${EVR}'" 2>&1 | tee "$OLOG"
done

echo
echo "============ SUMMARY: EventSSM (ours) — no-comp vs delta-scaled ============"
printf "%-7s %-8s %-12s %-14s %-14s\n" "mult" "dt(ms)" "step_scale" "no-comp AP" "comp AP"
for V in "${RATES[@]}"; do
  SS=$(step_scale_of "$V")
  comp=$(ap_of "$OUT/eventssm_dt${V}_ss${SS}.log")
  nocomp=$(ap_of "$NOCOMP/eventssm_dt${V}.log")
  printf "%-7s %-8s %-12s %-14s %-14s\n" "${MULT[$V]:-?}" "$V" "$SS" "${nocomp:-—}" "${comp:-—}"
done
echo "============================================================================"
echo "no-comp EventSSM 4x reference: 35.4 — comp should recover toward its 1x peak 46.2."
echo "logs: $OUT/"
