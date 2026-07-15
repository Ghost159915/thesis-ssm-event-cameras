#!/usr/bin/env bash
# Stage-14 GEN1 TEST-SET eval (RTX 5070 Ti) for the PURE-SSM (BiMamba-spatial) detector.
# Exact clone of stage7_test_eval_local.sh (the Stage-8 EventSSM test eval that produced test/AP=46.2),
# with ONLY two changes so the number is a fair apples-to-apples comparison:
#   1. backbone experiment:  +experiment/gen1=resnet_mamba  ->  +experiment/gen1=puressm
#   2. default checkpoint  :  the 400k-run best-val ckpt copied home to results/stage14_cloud/ckpts/
# Everything else (use_test_set=1 -> 470-rec test split, bf16-mixed, conf 0.001, batch 4) is untouched.
# Runs RVT validation.py via stage7_eval.py (registers the drop-in backbone first). The test/AP here is
# DIRECTLY comparable to: S5-RVT 47.7, EventSSM 46.2 (all test/AP; the W&B curve reports val/AP, a
# different split). Per the terminal policy, the USER runs this.
#
#   bash stage14_puressm_test_eval_local.sh                 # eval the 400k best ckpt (default below)
#   bash stage14_puressm_test_eval_local.sh /abs/ckpt.ckpt  # eval a specific ckpt
#   CKPT=/abs/ckpt.ckpt BATCH=2 bash stage14_puressm_test_eval_local.sh   # drop BATCH if VRAM-tight
#   bash stage14_puressm_test_eval_local.sh --cfg job       # GPU-free dry-run (compose config + exit)
set -euo pipefail

set +u                                               # conda activate.d references unbound vars
source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1            # PyTorch 2.6+ refuses to unpickle the Lightning ckpt otherwise
export PYTHONUNBUFFERED=1                            # stream the COCO eval / per-class AP table live
REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
ORIG_PWD="$PWD"                                      # capture invocation cwd BEFORE the cd, so a RELATIVE
                                                    # ckpt arg still resolves after we cd into RVT/
cd "$REPO/external/ssms_event_cameras/RVT"          # hydra config_path="config" is relative to validation.py

# ---------------- knobs ----------------
DATASET="${DATASET:-$REPO/data/gen1_raw/gen1}"      # dir CONTAINING test/ (NOT test/ itself) -> else mAP=0/NaN
DEFAULT_CKPT="$REPO/results/stage14_cloud/ckpts/epoch=002-step=310000-val_AP=0.48.ckpt"  # 400k-run best (val/AP 0.48 @ step 310k)
CKPT="${1:-${CKPT:-$DEFAULT_CKPT}}"                 # positional arg > $CKPT env > default
# A leading flag (e.g. --cfg) is NOT a checkpoint path: fall back to the default and let it pass through.
[[ "$CKPT" == -* ]] && CKPT="$DEFAULT_CKPT"
# Resolve a RELATIVE ckpt path against the invocation cwd (we have since cd'd into RVT/), so you can pass
# e.g. `results/.../foo.ckpt` from the repo root and it still works. Absolute paths (/...) pass through.
[[ "$CKPT" != -* && "$CKPT" != /* ]] && CKPT="$ORIG_PWD/$CKPT"
BATCH="${BATCH:-4}"                                 # VRAM-safe at seq_len=21; drop to 2/1 if PureSSM OOMs on 16GB
# ---------------------------------------

# If the first arg was a real ckpt path we consumed it; drop it so the rest pass through to hydra.
PASSTHRU=("$@"); [[ ${#PASSTHRU[@]} -ge 1 && "${PASSTHRU[0]}" != -* ]] && PASSTHRU=("${PASSTHRU[@]:1}")

if [[ "$CKPT" != -* && ! -f "$CKPT" ]]; then
  echo "[stage14-eval] ERROR: checkpoint not found: $CKPT" >&2
  echo "  Copy the 400k ckpt home first (scp from the instance), or pass one explicitly:" >&2
  echo "    bash stage14_puressm_test_eval_local.sh /abs/path/to/<ckpt>.ckpt" >&2
  exit 1
fi

RESULTS="$REPO/results/stage14_test_eval"
mkdir -p "$RESULTS"
OUT="$RESULTS/test_eval_$(date +%Y%m%d_%H%M%S).txt"
echo "[stage14-eval] ckpt:    $CKPT"
echo "[stage14-eval] dataset: $DATASET  (expects test/ with 470 recs)"
echo "[stage14-eval] log ->   $OUT"

# checkpoint value is single-quoted: Lightning ckpt names contain '=' (epoch=...-step=...-val_AP=...),
# which Hydra's override grammar rejects unless the value is quoted (same fix as the resume path).
python "$REPO/code/event_ssm/scripts/stage7_eval.py" \
  dataset=gen1 \
  dataset.path="$DATASET" \
  model=rnndet +experiment/gen1=puressm \
  "checkpoint='$CKPT'" \
  use_test_set=1 \
  hardware.gpus=0 \
  hardware.num_workers.eval=2 \
  batch_size.eval="$BATCH" \
  training.precision="bf16-mixed" \
  model.postprocess.confidence_threshold=0.001 \
  "${PASSTHRU[@]}" 2>&1 | tee "$OUT"
