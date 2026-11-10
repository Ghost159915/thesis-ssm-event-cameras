#!/usr/bin/env bash
# 400k-model bootstrap of 2026-11-10 (CPU only). Redoes the 2026-10-17 run, which failed at once with
# CaptureMismatch: the S5-RVT capture (RVT recipe, batch 8) orders recordings differently from ours. --align-by-labels
# matches recordings by their ground truth; PureSSM goes first so it is the layout reference and the three own models
# keep their exact official point estimates (only S5-RVT is reordered; its AP still rounds to 47.72).
# Waits for the GPU chain's renders (~/render_done.flag): rendering and 6 bootstrap workers (≈ 0.6 GB each) would
# compete for the 15 GB of RAM. Run detached:
#   tmux new -d -s boot "bash ~/Desktop/thesis-ssm-event-cameras/code/event_ssm/scripts/overnight_2026-11-10_bootstrap.sh; bash"
# Add NOW=1 in front of `bash` to skip the wait (e.g. if the GPU chain is not running).
cd ~/Desktop/thesis-ssm-event-cameras/code || exit 1
LOGDIR=../results/overnight_2026-11-10
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/boot.log") 2>&1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
export PYTHONPATH="$PWD:$PWD/../external/ssms_event_cameras/RVT"   # RVT's evaluator must be importable
CAP=../results/stage21_bootstrap/captures

if [[ "${NOW:-0}" != "1" ]]; then
  echo "[boot] $(date '+%F %T') waiting for the renders (~/render_done.flag)..."
  until [[ -f ~/render_done.flag ]]; do sleep 60; done
fi

echo "[boot] $(date '+%F %T') models_400k"
python -m event_ssm.eval.bootstrap --capture puressm=$CAP/puressm_400k.pkl --capture s5rvt=$CAP/s5rvt.pkl \
  --capture eventssm=$CAP/eventssm_400k.pkl --capture graded=$CAP/graded_400k.pkl --align-by-labels \
  --pair s5rvt:eventssm --pair s5rvt:puressm --pair s5rvt:graded --pair eventssm:puressm --pair puressm:graded \
  --n-boot 1000 --workers 6 --out ../results/stage21_bootstrap/models_400k

echo "[boot] $(date '+%F %T') ALL DONE"
