#!/usr/bin/env bash
# Stage-7a RESUME helper. Restores FULL training state (optimizer + scheduler + global-step) from the
# newest 'last' checkpoint and continues the OneCycle schedule to MAX_STEPS via stage7_midrun_local.sh.
# (See that launcher + STAGE7_MIDRUN_RUN.md for why checkpoints live under the vendored RVT tree.)
# Per the terminal policy, the USER runs this.
#
#   bash stage7_resume_local.sh                 # auto-pick the newest last_*.ckpt across runs
#   bash stage7_resume_local.sh /abs/path.ckpt  # resume from a specific checkpoint
#   bash stage7_resume_local.sh --cfg job       # dry-run (passed through to the launcher)
set -euo pipefail
REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
CKPT_GLOB="$REPO/external/ssms_event_cameras/RVT/RVT/*/checkpoints/last_epoch=*.ckpt"

# A leading non-flag arg is treated as an explicit checkpoint path; flags (e.g. --cfg) pass through.
CKPT=""
if [[ $# -ge 1 && "$1" != -* ]]; then
  CKPT="$1"; shift
else
  # newest by mtime across all <runid> dirs
  CKPT="$(ls -t $CKPT_GLOB 2>/dev/null | head -n1 || true)"
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "[stage7-resume] ERROR: no 'last' checkpoint found." >&2
  echo "  Looked for: $CKPT_GLOB" >&2
  echo "  Pass one explicitly:  bash stage7_resume_local.sh /abs/path/to/last_epoch=...ckpt" >&2
  exit 1
fi

echo "[stage7-resume] resuming full training state from:"
echo "    $CKPT"
ls -la --time-style=long-iso "$CKPT" >&2 || true
export STAGE7_RESUME="$CKPT"
exec bash "$REPO/code/event_ssm/scripts/stage7_midrun_local.sh" "$@"
