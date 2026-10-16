#!/usr/bin/env bash
# Overnight bootstrap of 2026-10-17 (docs/runbooks/Overnight_2026-10-16.md §2), CPU only. Waits for the GPU chain's
# benchmarks to finish (flag file) so it cannot disturb their timings. Run detached in tmux:
#   tmux new -d -s boot "bash ~/Desktop/thesis-ssm-event-cameras/code/event_ssm/scripts/overnight_2026-10-17_bootstrap.sh; bash"
cd ~/Desktop/thesis-ssm-event-cameras/code || exit 1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
export PYTHONPATH="$PWD:$PWD/../external/ssms_event_cameras/RVT"   # RVT's evaluator must be importable
CAP=../results/stage21_bootstrap/captures

echo "[boot] $(date '+%F %T') waiting for the benchmarks..."
until [ -f ~/bench_done.flag ]; do sleep 60; done

echo "[boot] $(date '+%F %T') ladder_100k"
python -m event_ssm.eval.bootstrap --capture puressm=$CAP/puressm_100k.pkl --capture analog=$CAP/analog_100k.pkl \
  --capture graded=$CAP/graded_100k.pkl --capture binary=$CAP/spike_100k.pkl \
  --pair puressm:analog --pair analog:graded --pair graded:binary \
  --n-boot 1000 --workers 8 --out ../results/stage21_bootstrap/ladder_100k

echo "[boot] $(date '+%F %T') models_400k"
python -m event_ssm.eval.bootstrap --capture s5rvt=$CAP/s5rvt.pkl --capture eventssm=$CAP/eventssm_400k.pkl \
  --capture puressm=$CAP/puressm_400k.pkl --capture graded=$CAP/graded_400k.pkl \
  --pair s5rvt:eventssm --pair s5rvt:puressm --pair s5rvt:graded --pair eventssm:puressm --pair puressm:graded \
  --n-boot 1000 --workers 8 --out ../results/stage21_bootstrap/models_400k

echo "[boot] $(date '+%F %T') ALL DONE"
