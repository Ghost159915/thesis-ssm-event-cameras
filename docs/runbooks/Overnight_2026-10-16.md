# Overnight run, 2026-10-16 → 17 (GhostMachine)

Where we stopped (2026-10-16 ≈ 14:45): all Stage-21 test evals, the graded-400k benchmark + SOP accounting and the
firing-pattern figure are done and recorded (`docs/results/Stage21_test_results.md` §1–5). All 8 bootstrap captures
exist in `results/stage21_bootstrap/captures/` (spike/graded/analog/puressm 100k, puressm/graded/eventssm 400k,
s5rvt). Tonight: per-arm SOP benchmarks → threshold sweeps → true-rate sweeps on the GPU, and the bootstrap on the CPU.

**PC vs M4 Mac (decided 2026-10-16):** everything runs on the PC tonight. The GPU jobs need CUDA and the sm_120
mamba-ssm kernels (PC only; benchmark numbers must come from the 5070 Ti). The bootstrap is CPU-only and would be
roughly 2× faster on the M4 (Ryzen 5 5600X = 6 cores, 2020) and would free the PC's cores, but needs the captures
(≈ 135 MB, not in git), RVT's evaluator and an identity check on the Mac first; not worth the setup when both jobs
finish before morning on the PC.

## 1. GPU chain — `tmux new -s night` (≈ 12–13 h)

Separate terminal (Ctrl+Alt+T), not VS Code. Paste, then `Ctrl-b d`. Independent jobs are joined with `;` so one
failure does not stop the night; the first loop waits for the VS Code process (`pgrep -x code`) to exit, the second
for the idle guard (< 1500 MiB). Memory alone is not a reliable signal: after the 2026-10-17 01:05 reboot the GPU
showed 1404 MiB with VS Code open, so a memory-only wait would have started the benchmarks with VS Code running.
**Started 2026-10-17 ≈ 01:15** (the 16th's plan slipped; machine rebooted 01:05, earlier tmux sessions gone).

```bash
cd ~/Desktop/thesis-ssm-event-cameras; S=$PWD/code/event_ssm/scripts; R=$PWD/external/ssms_event_cameras/RVT/RVT; rm -f ~/bench_done.flag; \
echo "waiting for VS Code to close..."; while pgrep -x code >/dev/null; do sleep 5; done; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1450 ]; do sleep 5; done; sleep 20; \
bash $S/stage10_run_local.sh --models spikingssm --spikingssm-ckpt $R/5ykgl80y/checkpoints/last_epoch=002-step=100000.ckpt --spikingssm-test-ap 0.4018 --out $PWD/results/stage22/spike_100k; \
bash $S/stage10_run_local.sh --models spikingssm --spikingssm-ckpt $R/rnak3sms/checkpoints/last_epoch=000-step=100000.ckpt --spikingssm-test-ap 0.4072 --out $PWD/results/stage22/graded_100k; \
bash $S/stage10_run_local.sh --models spikingssm --spikingssm-ckpt $R/sx14rpo8/checkpoints/last_epoch=002-step=100000.ckpt --spikingssm-test-ap 0.4225 --out $PWD/results/stage22/analog_100k; \
touch ~/bench_done.flag; \
bash $S/stage22_threshold_sweep_local.sh $R/5ykgl80y/checkpoints/last_epoch=002-step=100000.ckpt; \
bash $S/stage22_threshold_sweep_local.sh $R/rnak3sms/checkpoints/last_epoch=000-step=100000.ckpt; \
OURS_WRAP=$S/stage21_spikingssm_test_eval_local.sh OURS_CKPT=$R/rnak3sms/checkpoints/last_epoch=000-step=100000.ckpt OURS_KEY=graded_s234_100k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh; \
OURS_WRAP=$S/stage21_spikingssm_test_eval_local.sh OURS_CKPT=$R/5ykgl80y/checkpoints/last_epoch=002-step=100000.ckpt OURS_KEY=spike_s234_100k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh; \
OURS_WRAP=$S/stage21_spikingssm_test_eval_local.sh OURS_CKPT="$R/a43jctgx/checkpoints/epoch=006-step=350000-val_AP=0.47.ckpt" OURS_KEY=graded_s234_400k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh
```

Order and rough times: 3 per-arm SOP benchmarks (spiking model only, minutes each) → threshold sweeps spike,
graded 100k (6 thresholds × ≈ 15 min each ≈ 1.5 h per arm) → true-rate sweeps graded 100k, spike 100k, graded 400k
(≈ 3 h each, uncompensated only; the 400k run is the like-for-like comparison with PureSSM 400k's 69.7 %).

## 2. Bootstrap — `tmux new -s boot` (CPU only, ≈ 10 h)

Paste, then `Ctrl-b d`. Waits for `~/bench_done.flag` so it cannot disturb the benchmark timings.

```bash
cd ~/Desktop/thesis-ssm-event-cameras/code && source ~/miniforge3/etc/profile.d/conda.sh && conda activate events_signals; \
export PYTHONPATH="$PWD:$PWD/../external/ssms_event_cameras/RVT"; CAP=../results/stage21_bootstrap/captures; \
echo "waiting for the benchmarks..."; until [ -f ~/bench_done.flag ]; do sleep 60; done; \
python -m event_ssm.eval.bootstrap --capture puressm=$CAP/puressm_100k.pkl --capture analog=$CAP/analog_100k.pkl \
  --capture graded=$CAP/graded_100k.pkl --capture binary=$CAP/spike_100k.pkl \
  --pair puressm:analog --pair analog:graded --pair graded:binary \
  --n-boot 1000 --workers 8 --out ../results/stage21_bootstrap/ladder_100k && \
python -m event_ssm.eval.bootstrap --capture s5rvt=$CAP/s5rvt.pkl --capture eventssm=$CAP/eventssm_400k.pkl \
  --capture puressm=$CAP/puressm_400k.pkl --capture graded=$CAP/graded_400k.pkl \
  --pair s5rvt:eventssm --pair s5rvt:puressm --pair s5rvt:graded --pair eventssm:puressm --pair puressm:graded \
  --n-boot 1000 --workers 8 --out ../results/stage21_bootstrap/models_400k
```

## 3. Then close VS Code

The benchmarks start ≈ 20 s later. Watch with `tmux attach -t night` / `tmux attach -t boot`; leave with `Ctrl-b d`
(never Ctrl-C).

## 4. Next morning

Reopen VS Code, `claude --continue`, say "update". Expected outputs:

- `results/stage22/{spike,graded,analog}_100k/bench_results.json` (+ run the reporter for `sop_table.md`):
  energy side of the ladder (analog dense MACs, graded sparse MACs, spike ACs).
- `results/stage22/threshold_sweep/{spike,graded}_s234/` → threshold-Pareto figure
  (`python -m event_ssm.eval.threshold_sweep …`, GhostMachine checklist §E item 5). θ = 1.0 must reproduce the
  canonical Stage-21 test AP.
- `results/stage9/sweep_tr/{graded_s234_100k,spike_s234_100k,graded_s234_400k}_dt{25,5}_nc.log` →
  `fig_robustness` panel (c) appears automatically (`python thesis/figures/fig_robustness.py`).
- `results/stage21_bootstrap/{ladder_100k,models_400k}/bootstrap.{json,md}` → ladder figure
  (`python -m event_ssm.eval.ladder_plot …`, bootstrap runbook step 5) and error bars in `fig_accuracy_by_size`.
- Still open: `! wandb login` → PureSSM 400k training curves; restyle the firing-pattern figure; tidy label
  overlaps in `fig_training_400k` / `fig_spiking_firing`; fill §5.7 `\res{}` in `draft_main.tex`; the user's
  `draft_main.tex` has an accidental edit (`\usepackage{arra` on line 15) to fix.

## 5. Add-on: PureSSM 100k anchor on the true-rate set (added 2026-10-16, Mac session)

The chain in §1 runs the true-rate sweep for graded 100k, spike 100k and graded 400k only. §5.7.5 of `draft_main.tex`
compares the 100k readouts' true-10× retention with **the PureSSM 100k anchor under the same protocol**, and that run
is not in the chain (PureSSM 400k's 69.7 % is the reference for graded 400k only). After the chain has finished (or
appended to it), in the same tmux window (≈ 3 h, GPU idle):

```bash
cd ~/Desktop/thesis-ssm-event-cameras; S=$PWD/code/event_ssm/scripts; R=$PWD/external/ssms_event_cameras/RVT/RVT; \
OURS_WRAP=$S/stage14_puressm_test_eval_local.sh OURS_CKPT=$R/igk6myyb/checkpoints/last_epoch=002-step=100000.ckpt OURS_KEY=puressm_100k RUN_BASELINE=0 RUN_COMP=0 bash $S/stage9_truerate_sweep.sh
```

Output: `results/stage9/sweep_tr/puressm_100k_dt{25,5}_nc.log`; its 1× reference is the canonical Stage-21 test AP
42.52. `fig_robustness` panel (c) does not plot it yet (it uses PureSSM 400k as the reference line).
