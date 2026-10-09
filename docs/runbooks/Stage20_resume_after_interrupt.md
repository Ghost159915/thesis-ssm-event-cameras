# Stage 20 — Restart the training after the 2026-10-09 interruption

**Where things stand.** The graded 100k arm was stopped with Ctrl-C on 2026-10-09 at 19:02 (step ≈ 48.1k) because the
GPU was needed. Its last checkpoint is at **step 40k**, so the resume repeats ≈ 8k steps (≈ 1 h). Spike 100k is done.
Still to run: graded 100k (resume), analog 100k, PureSSM 100k anchor, graded 400k. Record: `docs/notes/Stage20_fullrun_notes.md` §3.2.

## 1. Free the GPU

Close whatever was using it, then check:

```bash
nvidia-smi
```

The process list at the bottom should show no Python/compute process, and memory should be back to the desktop's
≈ 1.5 GB. Training needs ≈ 13 GB of the 16 GB.

## 2. Open the tmux session

```bash
tmux attach -t stage20        # if it says "no sessions": tmux new -s stage20
```

## 3. Paste the restart chain

```bash
S=/home/ghost/Desktop/thesis-ssm-event-cameras/code/event_ssm/scripts
CKPT=/home/ghost/Desktop/thesis-ssm-event-cameras/external/ssms_event_cameras/RVT/RVT/5d1y0skm/checkpoints/last_epoch=000-step=40000.ckpt
STAGE7_RESUME=$CKPT ARM=graded STAGES=2,3,4 BUDGET=100k bash $S/stage20_full_local.sh && \
ARM=analog STAGES=2,3,4 BUDGET=100k bash $S/stage20_full_local.sh && \
bash $S/stage20_puressm_local.sh && \
ARM=graded STAGES=2,3,4 BUDGET=400k bash $S/stage20_full_local.sh
```

What each line does:

| # | Run | Time |
|---|---|---|
| 1 | graded 100k, **resumed** from the 40k checkpoint (weights, optimizer, learning-rate schedule and step count restored) | ≈ 9 h |
| 2 | analog 100k, fresh | ≈ 14 h |
| 3 | PureSSM 100k anchor, fresh (Stage-20 D6) | ≈ 14 h |
| 4 | graded 400k headline, fresh | ≈ 56 h |

`STAGE7_RESUME` is set only for line 1; the later lines start from scratch, as they should. If any line fails, the
`&&` stops the chain there.

## 4. Check the resume worked (first ~2 minutes)

These three lines should appear near the top of the output:

```
[stage7] resuming full training state from: .../5d1y0skm/checkpoints/last_epoch=000-step=40000.ckpt
[stage6_train] local-file resume: loading checkpoint directly from ...
[stage6_train] resume guard: checkpoint schedule matches 100000 steps
```

Then the progress bar should start moving at ≈ 2.2 it/s. The first validation after the resume is **step 50,000**,
≈ 1 h 20 min later, logged as `global step 50000: 'val/AP' ...`. It should land above the 40k value (0.352).

If it fails instead:
- *"resume checkpoint was trained on a …-step schedule"*: the resume guard rejected the checkpoint. Check that
  `CKPT` is the graded 40k file above and that `BUDGET=100k`.
- *State-dict / `ValueError` about the arm*: the checkpoint is from another model or arm. Same check.

## 5. Leave it running

Detach with **Ctrl-b, then d**. It keeps going with the terminal closed.

## If you need the GPU again

Stopping just **after** a checkpoint loses the least: checkpoints are written every 10k steps, right after each
`global step …0000: 'val/AP'` line. Ctrl-C at any other time loses everything since the last one. To resume again,
point `CKPT` at the newest `last_*.ckpt` of the run that was stopped (`ls -t <run>/checkpoints/last_*.ckpt | head -1`)
and use that run's `ARM`/`BUDGET`.

## Afterwards

Tell Claude the time you restarted. It goes into the notes (§3.2, "resume time"), since the resumed graded arm sees a
different data order from 40k on, and the thesis reports that.
