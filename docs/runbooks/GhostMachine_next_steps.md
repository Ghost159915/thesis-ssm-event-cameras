# GhostMachine — next steps checklist

**Written 2026-10-11.** One list of everything that has to happen on the workstation, in order. Details live in the
linked runbooks; this page is the order of play. Tick items off as you go.

Rule throughout: **while a training run is on the GPU, do only CPU work** (no test evals, no benchmark, no
`pytest -m gpu`). A plain `pytest` is CPU-only and safe: every CUDA test carries the `gpu` mark (modules since 2026-10-11; tests
using the `device` fixture via a `conftest.py` hook since 2026-10-12).

---

## A. On arrival (CPU only, safe while training runs)

- [ ] `cd ~/Desktop/thesis-ssm-event-cameras && git pull`
- [ ] Check the training chain: `tmux attach -t stage20` (detach with Ctrl-b, d). Note which run is going and its
      latest `val/AP` line. If it was stopped: resume from the newest checkpoint as in
      `docs/runbooks/Stage20_resume_after_interrupt.md` ("If you need the GPU again"), and record the restart time
      in `docs/notes/Stage20_fullrun_notes.md` §3.2.
- [ ] Analysis-tool tests (CPU): `conda activate events_signals && cd code && pytest event_ssm/tests/eval -q`
      → expect `27 passed`.
- [ ] Item-3 launcher tests (CPU; they need bash ≥ 4, so they cannot run on the Mac):
      `pytest event_ssm/tests/models/spikingssm/test_spiking_ckpt.py event_ssm/tests/models/spikingssm/test_stage21_eval_script.py
      event_ssm/tests/models/spikingssm/test_truerate_sweep_spiking.py -q` → all pass.

## B. Back up the results (CPU only)

All Stage 8–22 outputs (benchmark JSONs, rate curves, SOP tables) exist only on this machine (`/results/` is
gitignored).

- [ ] Pull the `.gitignore` change that tracks small result files (figures, JSON, CSV, markdown, test-eval .txt; checkpoints
      and logs stay ignored) — pushed from the Mac on 2026-10-11.
- [ ] Dry run and check the size: `git add -n results/ | wc -l` and
      `git add -n results/ | sed "s/^add '//;s/'$//" | xargs du -ch | tail -1`. Exclude any unexpectedly large file.
- [ ] Commit and push. This also brings the Stage-9 true-rate curve to the Mac for slide 7.

## C. Bootstrap Part A (CPU / code work)

Follow `docs/runbooks/Stage21_bootstrap_ghostmachine.md` steps 2–3:

- [x] Read the RVT evaluator and the detection test step; confirm the hook points. (2026-10-12)
- [x] Write `code/event_ssm/eval/bootstrap_capture.py` + its CPU tests; add the opt-in hook line to
      `stage7_eval.py` and `stage8_baseline_eval.py`.
- [x] Commit and push. (2026-10-12)

## D. Let the training chain finish

Order (Stage-20 plan): graded 100k (resumed) → analog 100k → PureSSM 100k anchor → graded 400k.

- [ ] **Decide on the graded 400k run** (≈ 56 h). It is the pre-registered headline, but the ladder and §5.7 only
      need the 100k runs. Recommendation: let the three 100k runs + anchor finish, do section E, and start the 400k
      run only if the calendar allows.

## E. When the GPU is free (one job at a time)

1. [ ] **Identity check** (≈ 14 min): PureSSM 400k test eval with `BOOTSTRAP_CAPTURE` set, then
       `--identity --expect puressm=0.4643` (bootstrap runbook step 4). **Must match before any interval is used.**
2. [ ] **Stage 21 test evals with capture**, one per 100k arm (spike, graded, analog via
       `stage21_spikingssm_test_eval_local.sh <ckpt>`; PureSSM anchor via `stage14_puressm_test_eval_local.sh <ckpt>`),
       each with `BOOTSTRAP_CAPTURE=…/captures/<arm>_100k.pkl` (bootstrap runbook step 5).
3. [ ] **Bootstrap** (CPU, can overlap the next GPU job): the ladder command in the bootstrap runbook step 5 →
       `results/stage21_bootstrap/ladder_100k/bootstrap.{json,md}`.
4. [ ] **Stage 22 benchmark + SOP energy** for the spiking model:
       `stage10_run_local.sh --models all+spikingssm --spikingssm-ckpt <ckpt> --spikingssm-test-ap 0.x`
       (writes to `results/stage22/`, never overwrites the Stage-10/16 JSON). Firing rates come from the FINAL
       checkpoints (trained rates ≈ 2× the 25k pilot).
5. [ ] **Threshold sweep** (item 2, ≈ 1.5 h per checkpoint): for the spike and graded 100k checkpoints,
       `bash code/event_ssm/scripts/stage22_threshold_sweep_local.sh <ckpt>` (thresholds 0.5 0.75 1.0 1.5 2.0 3.0;
       idempotent, preflight-checked). θ = 1.0 must reproduce the canonical Stage-21 test AP. Then the figure:
       `cd code && python -m event_ssm.eval.threshold_sweep --sweep graded=../results/stage22/threshold_sweep/graded_s234
       --sweep binary=../results/stage22/threshold_sweep/spike_s234 --out ../results/stage22/threshold_sweep/threshold_pareto`.
6. [ ] **Event-rate robustness of the spiking arms** (item 3, ≈ 3 h per checkpoint at 10×): the true-rate protocol,
       uncompensated only, e.g. for graded:
       `OURS_WRAP=$PWD/code/event_ssm/scripts/stage21_spikingssm_test_eval_local.sh OURS_CKPT=<graded 100k ckpt>
       OURS_KEY=graded_s234 RUN_BASELINE=0 RUN_COMP=0 bash code/event_ssm/scripts/stage9_truerate_sweep.sh`
       (same for spike). Logs land in `results/stage9/sweep_tr/<key>_dt{25,5}_nc.log` and the eval copies under
       `results/stage21_test_eval/<tag>_dt{25,5}/`. Retention = AP at 10× ÷ the arm's canonical 1× test AP; compare
       with PureSSM's 69.7 %. (The sweep refuses RUN_COMP=1 here: the spiking launcher unsets MAMBA_STEP_SCALE.)
7. [x] **β per channel** (done 2026-10-13 → `results/stage22/beta/beta_100k.*`; every channel within 0.044 of 0.9) (item 6; CPU only, safe during training):
       `cd code && python -m event_ssm.eval.beta_report --ckpt binary=<spike 100k ckpt> --ckpt graded=<graded 100k ckpt>
       --out ../results/stage22/beta/beta_100k` → table + histogram per stage (fills the β \res{} in §5.7).
   [ ] **Firing-pattern figure** (item 5; idle GPU, minutes): `python code/event_ssm/scripts/stage22_firing_pattern.py
       --ckpt <ckpt>` per arm → `results/stage22/firing/<tag>_<rec>.png` (events per frame vs per-stage firing rate,
       with the correlation r).
8. [ ] Optional: capture EventSSM 400k for the paired EventSSM vs PureSSM interval (46.2 vs 46.4, AP_L +3.0).

## F. After each result

- [ ] Commit the small result files (section B makes them trackable) and push, so the Mac has them for the thesis
      and the slides.
- [ ] Tell Claude the numbers / paths: they go into §5.7, the abstract, contribution 6, slide 8 and slide 10.
