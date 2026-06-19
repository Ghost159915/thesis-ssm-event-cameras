#!/usr/bin/env bash
# Stage-7 GEN1 TEST-SET eval (RTX 5070 Ti). Runs RVT validation.py (use_test_set=1 -> trainer.test on the
# held-out test split, 470 recs) on our drop-in ResNetMamba detector via stage7_eval.py (registers the
# backbone first). Produces the test/AP that is DIRECTLY comparable to the published baselines
# (S5-ViT repro = 47.7, RVT = 47.2 -- BOTH are test/AP; the training curve reports val/AP, which is a
# different split). Mirrors VALIDATION_QUICKSTART.md (TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, conf 0.001).
# Per the terminal policy, the USER runs this.
#
#   bash stage7_test_eval_local.sh                 # eval the mid-run best ckpt (default below)
#   bash stage7_test_eval_local.sh /abs/ckpt.ckpt  # eval a specific ckpt (e.g. the 400k best, once it exists)
#   CKPT=/abs/ckpt.ckpt BATCH=8 bash stage7_test_eval_local.sh
#   bash stage7_test_eval_local.sh --cfg job       # GPU-free dry-run (compose config + exit)
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
DEFAULT_CKPT="$REPO/external/ssms_event_cameras/RVT/RVT/8zotrwjw/checkpoints/epoch=003-step=320000-val_AP=0.46.ckpt"  # 400k best (Stage-8 headline)
CKPT="${1:-${CKPT:-$DEFAULT_CKPT}}"                 # positional arg > $CKPT env > mid-run best
# A leading flag (e.g. --cfg) is NOT a checkpoint path: fall back to the default and let it pass through.
[[ "$CKPT" == -* ]] && CKPT="$DEFAULT_CKPT"
# Resolve a RELATIVE ckpt path against the invocation cwd (we have since cd'd into RVT/), so you can pass
# e.g. `external/.../foo.ckpt` from the repo root and it still works. Absolute paths (/...) pass through.
[[ "$CKPT" != -* && "$CKPT" != /* ]] && CKPT="$ORIG_PWD/$CKPT"
BATCH="${BATCH:-4}"                                 # VRAM-safe at seq_len=21; raise if headroom
# ---------------------------------------

# If the first arg was a real ckpt path we consumed it; drop it so the rest pass through to hydra.
PASSTHRU=("$@"); [[ ${#PASSTHRU[@]} -ge 1 && "${PASSTHRU[0]}" != -* ]] && PASSTHRU=("${PASSTHRU[@]:1}")

if [[ "$CKPT" != -* && ! -f "$CKPT" ]]; then
  echo "[stage7-eval] ERROR: checkpoint not found: $CKPT" >&2
  echo "  Pass one explicitly:  bash stage7_test_eval_local.sh /abs/path/to/<ckpt>.ckpt" >&2
  exit 1
fi

RESULTS="$REPO/results/stage7_test_eval"
mkdir -p "$RESULTS"
OUT="$RESULTS/test_eval_$(date +%Y%m%d_%H%M%S).txt"
echo "[stage7-eval] ckpt:    $CKPT"
echo "[stage7-eval] dataset: $DATASET  (expects test/ with 470 recs)"
echo "[stage7-eval] log ->   $OUT"

# checkpoint value is single-quoted: Lightning ckpt names contain '=' (epoch=...-step=...-val_AP=...),
# which Hydra's override grammar rejects unless the value is quoted (same fix as the resume path).
python "$REPO/code/event_ssm/scripts/stage7_eval.py" \
  dataset=gen1 \
  dataset.path="$DATASET" \
  model=rnndet +experiment/gen1=resnet_mamba \
  "checkpoint='$CKPT'" \
  use_test_set=1 \
  hardware.gpus=0 \
  hardware.num_workers.eval=2 \
  batch_size.eval="$BATCH" \
  training.precision="bf16-mixed" \
  model.postprocess.confidence_threshold=0.001 \
  "${PASSTHRU[@]}" 2>&1 | tee "$OUT"
