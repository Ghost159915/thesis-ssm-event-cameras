#!/usr/bin/env bash
# Final-results chain of 2026-11-10 (GhostMachine). Redoes what the 2026-10-17 night could not:
#   the true-rate test set had been deleted (data/gen1_stage9/), so every true-rate eval was skipped.
# Order = priority, so whatever is cut short is the least important:
#   1. RVT ConvLSTM checkpoint (download if missing) + its 1x test eval  (decision D1; expect ~47.2)
#   2. rebuild the true-rate test set: extract + convert raw test events, render 10x then 2x
#   3. -> touch ~/render_done.flag  (the CPU bootstrap session starts then; it would compete for RAM with rendering)
#   4. graded 400k true-rate sweep  (like-for-like with PureSSM 400k's 69.7 % retention)
#   5. RVT ConvLSTM at true 10x and 2x  (D1)
#   6. bonus: graded 100k, spike 100k, PureSSM 100k true-rate sweeps (the ladder's robustness)
# Accuracy evals only (no benchmark), so VS Code may stay open. Everything is idempotent: re-running skips finished
# work. Independent steps are joined without `set -e`, so one failure does not stop the night. Full log:
#   results/overnight_2026-11-10/night.log   (survives a reboot, unlike the tmux scrollback)
# Run detached:
#   tmux new -d -s night "bash ~/Desktop/thesis-ssm-event-cameras/code/event_ssm/scripts/overnight_2026-11-10_gpu.sh; bash"
cd ~/Desktop/thesis-ssm-event-cameras || exit 1
REPO=$PWD
S=$REPO/code/event_ssm/scripts
R=$REPO/external/ssms_event_cameras/RVT/RVT
TR=$REPO/data/gen1_stage9/preproc_tr
RAWTEST=$REPO/data/gen1_stage9/raw/detection_dataset_duration_60s_ratio_1.0/test
LOGDIR=$REPO/results/overnight_2026-11-10
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/night.log") 2>&1
rm -f ~/render_done.flag
say() { echo; echo "[night] $(date '+%F %T') $*"; }
rendered() { find "$TR/test" -path "*stacked_histogram_dt=$1_nbins=10/event_representations.h5" 2>/dev/null | wc -l; }
lstm_done() { grep -l 'test/AP' "$REPO"/results/stage9_rvt_lstm/rvt_lstm_"$1"_*.log 2>/dev/null | head -1; }

# ── 1. ConvLSTM checkpoint + 1x eval ───────────────────────────────────────────────────────────────────────────
say "1. RVT ConvLSTM checkpoint"
if [[ ! -s $REPO/checkpoints/rvt-b-gen1.ckpt ]]; then
  wget -c -O "$REPO/checkpoints/rvt-b-gen1.ckpt" https://download.ifi.uzh.ch/rpg/RVT/checkpoints/gen1/rvt-b.ckpt
fi
ls -la "$REPO/checkpoints/rvt-b-gen1.ckpt"
if [[ -n "$(lstm_done gen1)" ]]; then say "ConvLSTM 1x already done: $(lstm_done gen1)"
else say "ConvLSTM 1x test eval (canonical dt=50 test set; RVT paper: 47.2)"; bash $S/stage9_rvt_lstm_eval_local.sh; fi

# ── 2. rebuild the true-rate test set ──────────────────────────────────────────────────────────────────────────
say "2a. extract + convert raw test events (resumable)"
bash $S/stage9_prep_test.sh
n_h5=$(find "$RAWTEST" -name '*_td.dat.h5' 2>/dev/null | wc -l)
if [[ "$n_h5" -eq 470 ]]; then
  say "470/470 .dat.h5 present -> deleting the raw .dat (≈ 142 GB; re-extractable from data/test_{a,b}.7z)"
  rm -f "$RAWTEST"/*_td.dat
else
  say "ERROR: only $n_h5/470 .dat.h5 — keeping the .dat; the renders below will refuse to start"
fi
df -h "$REPO" | tail -1

say "2b. render true 10x (window = stride = 5 ms)"
[[ "$(rendered 5)" -ge 469 ]] && say "10x already rendered" || bash $S/stage9_render_truerate.sh 10x
say "2c. render true 2x (window = stride = 25 ms)"
[[ "$(rendered 25)" -ge 469 ]] && say "2x already rendered" || bash $S/stage9_render_truerate.sh 2x
say "rendered: 10x $(rendered 5)/469, 2x $(rendered 25)/469"
for V in 5 25; do [[ "$(rendered $V)" -ge 469 ]] || say "ERROR: dt=$V incomplete — its evals below will be skipped"; done
touch ~/render_done.flag                                   # the bootstrap session starts now

# ── 3. graded 400k true-rate sweep (the like-for-like with PureSSM 400k) ───────────────────────────────────────
say "3. graded 400k true-rate sweep (2x then 10x, uncompensated)"
OURS_WRAP=$S/stage21_spikingssm_test_eval_local.sh OURS_CKPT="$R/a43jctgx/checkpoints/epoch=006-step=350000-val_AP=0.47.ckpt" \
  OURS_KEY=graded_s234_400k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh

# ── 4. RVT ConvLSTM on the true-rate set (D1) ──────────────────────────────────────────────────────────────────
for V in 5 25; do
  if [[ -n "$(lstm_done "preproc_tr_dt=$V")" ]]; then say "ConvLSTM dt=$V already done"; continue; fi
  if [[ "$(rendered $V)" -lt 469 ]]; then say "ConvLSTM dt=$V skipped: not rendered"; continue; fi
  say "4. ConvLSTM true-rate dt=$V"
  DATASET=$TR bash $S/stage9_rvt_lstm_eval_local.sh "" "dataset.ev_repr_name='stacked_histogram_dt=${V}_nbins=10'"
done

# ── 5. bonus: the 100k ladder's robustness ─────────────────────────────────────────────────────────────────────
say "5. bonus: graded 100k / spike 100k / PureSSM 100k true-rate sweeps"
OURS_WRAP=$S/stage21_spikingssm_test_eval_local.sh OURS_CKPT=$R/rnak3sms/checkpoints/last_epoch=000-step=100000.ckpt \
  OURS_KEY=graded_s234_100k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh
OURS_WRAP=$S/stage21_spikingssm_test_eval_local.sh OURS_CKPT=$R/5ykgl80y/checkpoints/last_epoch=002-step=100000.ckpt \
  OURS_KEY=spike_s234_100k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh
OURS_WRAP=$S/stage14_puressm_test_eval_local.sh OURS_CKPT=$R/igk6myyb/checkpoints/last_epoch=002-step=100000.ckpt \
  OURS_KEY=puressm_100k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh

say "ALL DONE"
grep -h 'test/AP' "$REPO"/results/stage9/sweep_tr/{graded_s234_400k,graded_s234_100k,spike_s234_100k,puressm_100k}_dt*_nc.log \
  "$REPO"/results/stage9_rvt_lstm/*.log 2>/dev/null | grep -v 'AP_' | head -20
