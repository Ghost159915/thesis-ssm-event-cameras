# docs/ — index & map

All prose lives here, one level of folders. Code is in `code/`; dissertation artifacts in `thesis/`.

## Where things live

| Folder | Contents | Key files |
|---|---|---|
| **roadmap/** | Stage-by-stage narrative (the story) + stage reports | `Stage_00`…`Stage_16`, `README.md` |
| **results/** | Per-stage result & comparison docs + the synthesis | **`Thesis_Progress_Writeup.md`** (start here), `Stage8/9/15/16_*` |
| **plans/** | Implementation plans (superpowers workflow) | `2026-*-stageNN-*.md`, `Repo_Restructure_Proposal.md` |
| **specs/** | Design specs (the "what") | `2026-*-design.md`, `design_specification.md` |
| **research/** | Lit-reviews / deep-dives (the reading) | SSM, **Loihi/INRC** (`INRC_Loihi_Access_Notes.md`, `DeepResearch_Loihi_SpikingSSM.md`), datasets (`Augmentation_Dataset_and_Resolution_Research.md`, `Event_Dataset_Survey_Breakdown.md`, `HighRes_EventCameras_Gehrig2022_analysis.md`), UAV, EventCV |
| **runbooks/** | Setup & ops guides (the "how to run") | `MVP_Setup_Guide_Complete.md`, `VALIDATION_QUICKSTART.md`, `Cloud_Runbook_5090.md`, `Stage14_env_reuse_recipe.md` |
| **notes/** | Technical notes / rationale / verdicts | `Stage9_Zubic_methodology_verdict.md`, `Stage11/12/13_*_notes.md`, `architecture_blueprint.md`, `codebase_audit.md`, `yolox_head_interface.md`, `Supervisor_Update_*` |
| **patches/** | Re-applies `external/` modifications after any re-clone | `*.patch`, `README.md` |

## Reading paths

- **"What is this thesis and what were the results?"** → `results/Thesis_Progress_Writeup.md`.
- **"How was each stage done?"** → `roadmap/Stage_XX_*.md` (narrative) + matching `plans/` + `specs/`.
- **"Where are the numbers?"** → `results/` (accuracy/robustness/efficiency tables).
- **"How do I run/set up?"** → `runbooks/`.
- **"Where's the thesis heading next?"** → `research/` (1Mpx augmentation, Loihi/INRC access) + `CLAUDE.md` §Next Steps.

## Notes on the restructure (2026-07-16)

- The 3 root design docs (`design_specification.md` → `specs/`; `architecture_blueprint.md`, `codebase_audit.md`
  → `notes/`) **overlap** their fuller `roadmap/Stage_00/01/02` counterparts but are **not identical** (shorter
  distilled versions). They were **preserved, not merged** — deciding which is canonical is a manual call.
- Research-paper PDFs live in `thesis/references/papers/` (gitignored bulk), paired with `thesis/latex/references.bib`.
