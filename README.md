# State-Space Models for Event-Camera Object Detection

Final-year robotics thesis (Thesis B · MMAN4952 · UNSW Sydney · Benas Vaiciulis) on **State-Space Models
(Mamba/SSM) for event-camera object detection**, targeting low-power micro-UAV / automotive perception.

The work builds two original detector backbones on top of a verified S5-RVT baseline and compares them in a
**controlled four-model ablation** (only the backbone's spatial mixer changes):

| Model | Spatial mixer | Temporal | Gen1 test/AP | Notes |
|---|---|---|---|---|
| S5-RVT (Zubić 2024, reproduced) | MaxViT | S5 | **47.72** | reference baseline (exact repro) |
| **EventSSM** (ours) | ResNet-18 conv | Mamba | 46.22 | CNN–SSM hybrid |
| **PureSSM** (ours) | BiMamba | Mamba | 46.43 | pure SSM; **AP_L +2.95**, smallest (16.33 M), most rate-robust (69.7% @ true-10×) |

**Start here:** [`docs/results/Thesis_Progress_Writeup.md`](docs/results/Thesis_Progress_Writeup.md) — the full
thesis-shaped synthesis (motivation → method → all result pillars → discussion → next steps).

## Repository map

```
code/event_ssm/   the detector:
  ├─ models/{eventssm,puressm}/   the two contributions' spatial backbones (ResNet-18 conv | BiMamba SSM)
  ├─ backbone/                    shared recurrent skeleton (ResNetMamba, hosts spatial mixer + temporal)
  ├─ temporal/                    Mamba temporal path — shared by both models
  └─ integration · benchmark · viz · configs · scripts · tests/{models/{eventssm,puressm}, core} · proofs
env/              conda/pip lockfiles (Blackwell sm_120, cu128)
docs/             all prose — see docs/README.md for the full index
  ├─ roadmap/       stage-by-stage narrative (Stage_00 … Stage_16) + stage reports
  ├─ results/       per-stage result & comparison docs + the progress write-up
  ├─ plans/         implementation plans (superpowers)
  ├─ specs/         design specs
  ├─ research/      lit-reviews / deep-dives (SSM, Loihi/INRC, datasets, resolution, UAV)
  ├─ runbooks/      setup & ops guides (MVP setup, validation, cloud-5090, env reuse)
  ├─ notes/         technical notes, rationale, verdicts
  └─ patches/       re-applies external/ modifications after any re-clone
thesis/           dissertation artifacts — latex/ (main.tex + references.bib), diagrams/, admin/
external/         vendored S5-RVT/RVT baseline (gitignored — reconstruct via docs/patches + cloud setup)
data/ results/ checkpoints/   experiment inputs/outputs (gitignored bulk)
```

## Quick pointers

- **Run the test suite:** `pytest code/event_ssm/tests/` (92 pass; `+ -m gpu` for the 1 GPU-marked test on an idle GPU).
- **Reproduce PureSSM eval:** `bash code/event_ssm/scripts/stage14_puressm_test_eval_local.sh`
- **Environment / setup:** [`docs/runbooks/MVP_Setup_Guide_Complete.md`](docs/runbooks/MVP_Setup_Guide_Complete.md),
  [`docs/runbooks/VALIDATION_QUICKSTART.md`](docs/runbooks/VALIDATION_QUICKSTART.md);
  cloud training on a rented RTX 5090: [`docs/runbooks/Cloud_Runbook_5090.md`](docs/runbooks/Cloud_Runbook_5090.md).
- **AI-assistant project instructions:** [`CLAUDE.md`](CLAUDE.md) (current status + conventions).

## Status (2026-07-16)

EventSSM and PureSSM investigations are **complete** across accuracy, temporal-robustness, and efficiency pillars.
Next directions: integrate the write-up into the LaTeX chapters; **1Mpx/Gen4 resolution study** (planned, deferred);
and a **Loihi 2 / spiking-SSM** fork (pursuing INRC access — see
[`docs/research/INRC_Loihi_Access_Notes.md`](docs/research/INRC_Loihi_Access_Notes.md)).
