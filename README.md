# State-Space Models for Event-Camera Object Detection

Final-year robotics thesis (Thesis C · MMAN4953 · UNSW Sydney · Benas Vaiciulis; supervisor Dr Will Midgley).
The question: **what does a state-space model (SSM) actually contribute to event-camera object detection?**

The S5-RVT baseline (Zubić et al., CVPR 2024) was reproduced exactly, then own backbones were swapped into the same
frozen pipeline (YOLO-PAFPN neck, YOLOX head, loss, Gen1 data, Prophesee evaluator), one change at a time. Any
difference between neighbouring models comes from that one change alone.

| Model | Spatial | Temporal | Gen1 test AP | Notes |
|---|---|---|---|---|
| S5-RVT (reproduced) | MaxViT attention | S5 | **47.72** | exact reproduction of the paper |
| **EventSSM** (ours) | ResNet-18 conv | Mamba | 46.22 | fastest (12.6 ms/frame eager) |
| **PureSSM** (ours) | BiMamba | Mamba | 46.43 | smallest (16.3 M), fewest FLOPs (10.0 G), **AP_L +2.95**, most rate-robust (69.7 % at true 10×) |
| **SpikingSSM** (ours) | BiMamba | Mamba + LIF spiking readout | *training* | analog / graded / binary readouts; 100k ladder in progress |

Metric: COCO mAP at IoU 0.50:0.95 on the Gen1 test set (AP_50 of the reproduction = 75.3).

## Main findings

- **Accuracy.** EventSSM and PureSSM tie overall (within single-seed noise). PureSSM closes about half the
  large-object gap to the attention baseline; a trained effective-receptive-field probe shows the SSM spatial mixer
  sees up to twice as far as the CNN and widens its field during training.
- **Event-rate robustness.** At a true 10× event rate PureSSM keeps 69.7 % of its accuracy (EventSSM 63.0 %,
  S5-RVT 62.2 %). The inference-time Δt-rescaling credited for this in the literature is **not reproducible under
  the published checkpoint and code** (it costs 9–13 mAP at true 10×); the robustness is present without it.
- **Efficiency.** PureSSM has the best accuracy per GFLOP (4.63) but is launch-bound in eager PyTorch (26.9 ms;
  181 Hz with CUDA graphs). The Mamba models carry 144 MB of streaming state per stream vs 4.7 MB for S5-RVT.
- **Spiking (Thesis C, in progress).** A leaky integrate-and-fire readout on PureSSM's temporal path, with three
  readouts sharing one membrane, so PureSSM → analog → graded → binary separates the cost of the neuron dynamics,
  of sparsity and of binarisation (order pre-registered: analog ≥ graded > binary). 100k-step validation AP so far:
  binary 41.8 (complete); graded 41.6 vs binary 40.4 at 80k. Spikes feed only ~1 % of the detector's operations, so
  the energy case needs a spiking spatial backbone too (Future Work; Loihi 2 via NIR).

Full write-ups: [`docs/results/`](docs/results/) (Stage 8/9/15/16 results, `Thesis_Progress_Writeup.md` for
EventSSM + PureSSM) and the full draft [`thesis/latex/draft_main.tex`](thesis/latex/draft_main.tex).

## Status (2026-10-11)

- **Done:** baseline reproduction; EventSSM and PureSSM (all pillars); spiking model built, integrated and tested
  (Stages 17–19, kill-switch passed); binary 100k run complete; INRC Loihi 2 proposal finalised
  ([`thesis/inrc_proposal/`](thesis/inrc_proposal/), 2 pages, vLab access only).
- **In progress:** Stage 20 training chain on the workstation (graded 100k → analog 100k → PureSSM 100k anchor →
  optional graded 400k); bootstrap confidence intervals (Part B done, Part A on the workstation).
- **Next:** Stage 21 test evals, Stage 22 SOP/energy, spiking analysis (bootstrap CIs, threshold sweep, rate
  robustness, firing figure), then the final report (`main.tex`).
- **Submission:** the standard thesis report, written by the author in `thesis/latex/main.tex` (~50 pages of body,
  bibliography and appendices excluded;
  structure guide in [`docs/plans/2026-10-13-final-report-structure.md`](docs/plans/2026-10-13-final-report-structure.md)), plus a ~10-minute presentation.

Where to look next:
- Workstation to-do list: [`docs/runbooks/GhostMachine_next_steps.md`](docs/runbooks/GhostMachine_next_steps.md)
- Post-training analysis plan: [`docs/plans/2026-10-11-post-training-spiking-analysis.md`](docs/plans/2026-10-11-post-training-spiking-analysis.md)
- Term schedule: [`docs/plans/Thesis_C_Project_Timeline.md`](docs/plans/Thesis_C_Project_Timeline.md)
- Training log and decisions: [`docs/notes/Stage20_fullrun_notes.md`](docs/notes/Stage20_fullrun_notes.md)

## Repository map

```
code/event_ssm/                 the detectors (drop-in backbones for the vendored RVT pipeline)
  ├─ models/eventssm/             ResNet-18 spatial stages
  ├─ models/puressm/              BiMamba spatial stages
  ├─ models/spikingssm/           LIF readout + spiking backbone (Stages 17–18; sibling package, nothing else edited)
  ├─ backbone/  temporal/         shared recurrent skeleton + Mamba-2 temporal path (frozen)
  ├─ baselines/                   RVT ConvLSTM baselines (Stage-9 rate study)
  ├─ integration/                 Hydra registration, smoke harness, resume guard, spiking checkpoint contract
  ├─ benchmark/                   latency/energy/FLOP harness (Stage 10/16) + SOP accounting (Stage 22)
  ├─ eval/                        bootstrap confidence intervals for test AP (Stage 21)
  ├─ scripts/                     stage launchers (stageNN_*.sh / .py), cloud/ setup for a rented RTX 5090
  └─ configs · proofs · viz · tests/
env/                            conda/pip lockfiles (Blackwell sm_120, cu128)
docs/                           all prose — see docs/README.md
thesis/
  ├─ latex/                       main.tex (final report: headings + literature review, being written), draft_main.tex
  │                               (full 64-page draft, source material), LitReview_ThesisA.tex, references.bib
  ├─ inrc_proposal/               2-page Intel INRC Loihi 2 proposal
  ├─ presentation/                talk deck (source + offline preview), presentation guide, model explainer
  ├─ supervisor_report/           August progress report
  └─ diagrams/ · admin/
external/                       vendored S5-RVT/RVT baseline (gitignored; rebuild with docs/patches/)
data/ results/ checkpoints/     inputs/outputs (gitignored; results/ exists only on the workstation)
```

## Quick pointers

- **Tests (workstation):** `cd code && pytest event_ssm/tests` (CPU set; add `-m gpu` on an idle GPU for the
  GPU-marked tests). Do not run the full suite during a training run: some unmarked tests use CUDA.
- **Tests (Mac, no torch):** only the bootstrap tests run there:
  `cd code && PYTHONPATH=$PWD python3 -m pytest --noconftest -c event_ssm/pytest.ini --rootdir event_ssm event_ssm/tests/eval -q`
- **Evaluate** (launchers in `code/event_ssm/scripts/`): EventSSM `stage7_test_eval_local.sh` · PureSSM
  `stage14_puressm_test_eval_local.sh` · SpikingSSM `stage21_spikingssm_test_eval_local.sh <ckpt>` (reads the arm
  from the checkpoint) · S5-RVT `stage8_baseline_eval_local.sh`.
- **Train a spiking arm:** `ARM=graded STAGES=2,3,4 BUDGET=100k bash code/event_ssm/scripts/stage20_full_local.sh`;
  resuming after an interruption: [`docs/runbooks/Stage20_resume_after_interrupt.md`](docs/runbooks/Stage20_resume_after_interrupt.md).
- **Environment:** [`docs/runbooks/MVP_Setup_Guide_Complete.md`](docs/runbooks/MVP_Setup_Guide_Complete.md),
  [`docs/runbooks/VALIDATION_QUICKSTART.md`](docs/runbooks/VALIDATION_QUICKSTART.md); rented RTX 5090:
  [`docs/runbooks/Cloud_Runbook_5090.md`](docs/runbooks/Cloud_Runbook_5090.md).
- **Assistant instructions and full project state:** [`CLAUDE.md`](CLAUDE.md).
