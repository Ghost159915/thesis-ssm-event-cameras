# docs/ — index & map

All prose lives here, one level of folders. Code is in `code/`; dissertation artifacts in `thesis/`.

## Where things live

| Folder | Contents | Key files |
|---|---|---|
| **roadmap/** | Stage-by-stage narrative (the story) + stage reports | `Stage_00`…`Stage_16`, `README.md` (Stages 17–23 are recorded in `notes/`) |
| **results/** | Per-stage result & comparison docs + the synthesis | **`Thesis_Progress_Writeup.md`** (start here), `Stage8/9/15/16_*` |
| **plans/** | Implementation plans + the Thesis C schedule | **`Thesis_C_Project_Timeline.md`**, `2026-09-07-thesis-c-plan.md`, `2026-10-11-post-training-spiking-analysis.md`, `2026-*-stageNN-*.md` |
| **specs/** | Design specs (the "what") | `2026-*-design.md`, `design_specification.md` |
| **research/** | Lit-reviews / deep-dives (the reading) | SSM, **Loihi/INRC** (`INRC_Loihi_Access_Notes.md`, `DeepResearch_Loihi_SpikingSSM.md`), datasets (`Augmentation_Dataset_and_Resolution_Research.md`, `Event_Dataset_Survey_Breakdown.md`, `HighRes_EventCameras_Gehrig2022_analysis.md`), UAV, EventCV |
| **runbooks/** | Setup & ops guides (the "how to run") | **`GhostMachine_next_steps.md`** (workstation to-do), `Stage20_resume_after_interrupt.md`, `Stage21_bootstrap_ghostmachine.md`, `MVP_Setup_Guide_Complete.md`, `VALIDATION_QUICKSTART.md`, `Cloud_Runbook_5090.md`, `Stage14_env_reuse_recipe.md` |
| **notes/** | Technical notes / rationale / verdicts | Spiking model: `Stage17_build_notes.md`, `Stage18_integration_notes.md`, `Stage19_smoke_notes.md`, `Stage20_fullrun_notes.md`, `Stage21_22_tooling_notes.md`; `Thesis_factcheck_2026-10-07.md`, `Stage9_Zubic_methodology_verdict.md`, `Stage11/12/13_*_notes.md`, `architecture_blueprint.md`, `codebase_audit.md`, `yolox_head_interface.md`, `Supervisor_Update_*` |
| **patches/** | Re-applies `external/` modifications after any re-clone | `*.patch`, `README.md` |

## Reading paths

- **"What is this thesis and what were the results?"** → `../README.md`, then `results/Thesis_Progress_Writeup.md`
  (EventSSM + PureSSM) and `notes/Stage17`–`Stage20` (spiking model).
- **"How was each stage done?"** → `roadmap/Stage_XX_*.md` (narrative) + matching `plans/` + `specs/`.
- **"Where are the numbers?"** → `results/` (accuracy/robustness/efficiency tables).
- **"How do I run/set up?"** → `runbooks/`.
- **"What happens next?"** → `plans/Thesis_C_Project_Timeline.md`, `plans/2026-10-11-post-training-spiking-analysis.md`,
  `runbooks/GhostMachine_next_steps.md`; longer-term directions in `research/` (1Mpx, Loihi/INRC).

## Notes on the restructure (2026-07-16)

- The 3 root design docs (`design_specification.md` → `specs/`; `architecture_blueprint.md`, `codebase_audit.md`
  → `notes/`) **overlap** their fuller `roadmap/Stage_00/01/02` counterparts but are **not identical** (shorter
  distilled versions). They were **preserved, not merged** — deciding which is canonical is a manual call.
- Research-paper PDFs live in `thesis/references/papers/` (gitignored bulk), paired with `thesis/latex/references.bib`.
