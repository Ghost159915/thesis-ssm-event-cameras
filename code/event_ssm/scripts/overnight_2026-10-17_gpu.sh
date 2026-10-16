#!/usr/bin/env bash
# Overnight GPU chain of 2026-10-17 (docs/runbooks/Overnight_2026-10-16.md §1 + §5), as a file because pasting the
# multi-line block into a terminal garbled it. Run detached in tmux:
#   tmux new -d -s night "bash ~/Desktop/thesis-ssm-event-cameras/code/event_ssm/scripts/overnight_2026-10-17_gpu.sh; bash"
# Independent jobs are joined with ';' so one failure does not stop the night. Waits for VS Code to exit first
# (the benchmark's idle-GPU guard; memory alone is not a reliable signal), then signals the bootstrap via a flag file.
cd ~/Desktop/thesis-ssm-event-cameras || exit 1
S=$PWD/code/event_ssm/scripts
R=$PWD/external/ssms_event_cameras/RVT/RVT
rm -f ~/bench_done.flag

echo "[night] $(date '+%F %T') waiting for VS Code to close..."
while pgrep -x code >/dev/null; do sleep 5; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1450 ]; do sleep 5; done
sleep 20

# 1. per-arm SOP benchmarks (spiking model only; idle GPU)
echo "[night] $(date '+%F %T') benchmarks"
bash $S/stage10_run_local.sh --models spikingssm --spikingssm-ckpt $R/5ykgl80y/checkpoints/last_epoch=002-step=100000.ckpt \
  --spikingssm-test-ap 0.4018 --out $PWD/results/stage22/spike_100k
bash $S/stage10_run_local.sh --models spikingssm --spikingssm-ckpt $R/rnak3sms/checkpoints/last_epoch=000-step=100000.ckpt \
  --spikingssm-test-ap 0.4072 --out $PWD/results/stage22/graded_100k
bash $S/stage10_run_local.sh --models spikingssm --spikingssm-ckpt $R/sx14rpo8/checkpoints/last_epoch=002-step=100000.ckpt \
  --spikingssm-test-ap 0.4225 --out $PWD/results/stage22/analog_100k
touch ~/bench_done.flag                                    # the bootstrap session starts now

# 2. post-hoc threshold sweeps (≈ 1.5 h each)
echo "[night] $(date '+%F %T') threshold sweeps"
bash $S/stage22_threshold_sweep_local.sh $R/5ykgl80y/checkpoints/last_epoch=002-step=100000.ckpt
bash $S/stage22_threshold_sweep_local.sh $R/rnak3sms/checkpoints/last_epoch=000-step=100000.ckpt

# 3. true-rate robustness, uncompensated (≈ 3 h each)
echo "[night] $(date '+%F %T') true-rate sweeps"
OURS_WRAP=$S/stage21_spikingssm_test_eval_local.sh OURS_CKPT=$R/rnak3sms/checkpoints/last_epoch=000-step=100000.ckpt \
  OURS_KEY=graded_s234_100k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh
OURS_WRAP=$S/stage21_spikingssm_test_eval_local.sh OURS_CKPT=$R/5ykgl80y/checkpoints/last_epoch=002-step=100000.ckpt \
  OURS_KEY=spike_s234_100k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh
OURS_WRAP=$S/stage21_spikingssm_test_eval_local.sh OURS_CKPT="$R/a43jctgx/checkpoints/epoch=006-step=350000-val_AP=0.47.ckpt" \
  OURS_KEY=graded_s234_400k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh
OURS_WRAP=$S/stage14_puressm_test_eval_local.sh OURS_CKPT=$R/igk6myyb/checkpoints/last_epoch=002-step=100000.ckpt \
  OURS_KEY=puressm_100k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh

echo "[night] $(date '+%F %T') ALL DONE"
