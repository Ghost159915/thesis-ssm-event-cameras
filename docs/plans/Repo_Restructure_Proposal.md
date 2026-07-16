# Repo Restructure Proposal (for review — execute after Stage 16 merges)

**Goal:** one obvious home for every kind of file — code, roadmap, plans, results, research, thesis, env —
so the repo is clean, navigable, and self-explanatory. Executed *after* Stage 16 is merged, all tests pass,
and we've pushed a clean baseline (per your sequencing).

---

## 1. What's messy today (the survey)

- **Root clutter — 10 loose files:** `architecture_blueprint.md`, `codebase_audit.md`, `design_specification.md`,
  `MVP_Setup_Guide_Complete.md`, `VALIDATION_QUICKSTART.md`, `yolox_head_interface.md`, two `requirements_*.txt`,
  `events_signals_5070ti.yml` — design docs, setup guides, a code note, and env files all dumped at root.
- **Duplication:** root `architecture_blueprint.md` ↔ `stages/Stage_01_Architecture_Blueprint.md` ↔
  `Thesis_Plan/architecture_blueprint.dot/png` (architecture in **3** places); root `codebase_audit.md` ↔
  `stages/Stage_02_Codebase_Audit.md`; root `design_specification.md` ↔ `stages/Stage_00` ↔ `docs/superpowers/specs/`.
- **Docs sprawl (40 files, flat):** stage result docs, plans, specs, research (spiking/Loihi/UAV/EventCV), runbooks,
  patches, and misc notes all mixed in one `docs/` level.
- **Two parallel roadmap systems:** `stages/Stage_00–10` (EventSSM narrative) **and** `docs/superpowers/plans`
  (PureSSM plans) — same idea, two places; Stages 11–16 only exist as plans, never added to `stages/`.
- **Stage outputs scattered across 4 dirs:** `stages/`, `reports/`, `docs/`, `results/` all hold "stage X" material.
- **Dead code:** `code/ssm_event_detection/` (13 files) — the superseded standalone sketch (CLAUDE.md says the
  drop-in backbone *supersedes* it). Still tracked.
- **Thesis-writing split across 3 dirs:** `overleaf/` (LaTeX), `Thesis_Plan/` (diagrams), `reports/` (rubric).

## 2. Hard constraints — what must NOT move (or things break)

These have entanglements; moving them is high-risk and low-reward:
- **`code/event_ssm/`** package internals — imported everywhere; ~40 scripts, CLAUDE.md, and configs hardcode
  `code/event_ssm/...` paths. **Keep `code/` as-is** (a `src/` rename would churn every path for zero readability gain).
- **`code/event_ssm/configs/`** — symlinked from `external/.../config/experiment/gen1/*.yaml`; moving breaks the
  Hydra symlinks (recreated by `docs/patches`). Leave in place.
- **`docs/patches/`** — referenced by name in scripts + its own README. Keep the name.
- **`results/`, `data/`, `external/`, `checkpoints/`, `graphify-out/`** — stay (paths, gitignore, vendored code).

**Where the real cleanup is:** docs + root clutter + thesis dirs + dead code — all **low-risk** (prose/assets, not
imported). That's ~80% of the mess, movable safely with `git mv`.

## 3. Proposed target structure

```
thesis-ssm-event-cameras/
├── README.md              ← NEW: 1-page orientation (what's where, how to run, how to navigate)
├── CLAUDE.md              (stays — AI project instructions; update its path refs)
├── .gitignore
│
├── code/                  ← ALL code (internals unchanged — see constraints)
│   └── event_ssm/         (backbone · spatial · temporal · benchmark · integration · viz · configs · scripts · tests · proofs)
│   # DELETE code/ssm_event_detection/  (superseded sketch)
│
├── env/                   ← NEW: environment & dependency lockfiles
│   ├── events_signals_5070ti.yml
│   ├── requirements_5070ti_lock.txt
│   └── requirements_5070ti_mamba_lock.txt
│
├── docs/                  ← ALL prose, sub-organized (one level of folders)
│   ├── README.md          ← docs index / map
│   ├── roadmap/           ← stage-by-stage narrative  (was stages/ + reports/Stage_0X_*Report); ADD Stage_11–16
│   ├── results/           ← per-stage result & comparison docs (Stage8/9/15/16, smoke, stage6)
│   ├── plans/             ← implementation plans     (was docs/superpowers/plans)
│   ├── specs/             ← design specs             (was docs/superpowers/specs + root design_specification.md)
│   ├── research/          ← lit-reviews / deep-dives (spiking · Loihi · SSSM · UAV · EventCV)
│   ├── runbooks/          ← setup & ops guides       (MVP setup · validation quickstart · cloud runbook · env-reuse)
│   ├── notes/             ← technical notes          (gen1 res · stage9 rationale/verdict · build/integration notes · yolox interface)
│   └── patches/           ← external/ patches        (STAYS — referenced by scripts)
│
├── thesis/                ← ALL dissertation-writing artifacts
│   ├── latex/             ← main.tex · references.bib · images/   (was overleaf/)
│   ├── diagrams/          ← architecture .dot/.png                (was Thesis_Plan/)
│   └── admin/             ← marking rubric PDF                    (was reports/)
│
├── results/               ← experiment outputs (gitignored bulk + few tracked figures)  — STAYS
├── checkpoints/  ├── data/  ├── external/  └── graphify-out/       — STAY
```

## 4. Move map (high-level)

| From | → To | Action |
|---|---|---|
| `stages/Stage_*.md` | `docs/roadmap/` | git mv; add Stage_11–16 (from plans) so the narrative is complete |
| `reports/Stage_0X_*_Report.md` | `docs/roadmap/` (or `docs/results/`) | git mv; consolidate stage outputs |
| `reports/MMAN4951_Marking_Rubrics.pdf` | `thesis/admin/` | git mv |
| `overleaf/` | `thesis/latex/` | git mv |
| `Thesis_Plan/` | `thesis/diagrams/` | git mv |
| `docs/superpowers/plans/` | `docs/plans/` | git mv (flatten superpowers/) |
| `docs/superpowers/specs/` | `docs/specs/` | git mv |
| `docs/Spiking*`, `spiking_ssm_references`, `SSSMDetector_Loihi`, `UAV_*`, `EventCV_Assessment` | `docs/research/` | git mv |
| `docs/Cloud_Runbook_5090.md`, root `MVP_Setup_Guide`, `VALIDATION_QUICKSTART` | `docs/runbooks/` | git mv |
| `docs/Stage*_results*/comparison` | `docs/results/` | git mv |
| `docs/Stage*_notes`, `Stage9_rationale/verdict`, `Gen1_resolution`, root `yolox_head_interface.md` | `docs/notes/` | git mv |
| root `design_specification.md` | `docs/specs/` | dedup vs Stage_00 first |
| root `architecture_blueprint.md`, `codebase_audit.md` | merge into `docs/roadmap/Stage_01/02` | **dedup** (verify duplicate, keep one) |
| root `*.yml`, `requirements_*.txt` | `env/` | git mv |
| `code/ssm_event_detection/` | — | **DELETE** (superseded) |

## 5. DRY / dedup opportunities (the "reduce" half)

- **Delete** `code/ssm_event_detection/` (dead).
- **Merge** the 3 duplicated design docs (architecture/audit/spec) — one canonical copy each.
- **Unify** the two roadmap systems (`stages/` + `superpowers/plans`) under `docs/roadmap` + `docs/plans`.
- **Prose dedup:** the Stage result docs repeat context/numbers ("for the record" blocks) — trim to single-source
  tables + cross-links.
- **Scripts (40 flat in `code/event_ssm/scripts/`):** *candidate* to group by stage/purpose, but they cross-call
  each other and hardcode `$SCRIPTS` — **defer** unless done very carefully (low reward, real risk).

## 6. Execution approach (when we do it)

1. **Baseline first:** finish Stage 16 → run the FULL test suite (`pytest code/event_ssm/tests/`, incl. `-m gpu`) →
   all green → merge `stage16-pillars-visuals` → push. That commit is the refactor's starting point.
2. **Phased, low-risk-first, each phase its own commit + `graphify update .`:**
   - Phase 1 — delete dead code (`ssm_event_detection`).
   - Phase 2 — `env/` + `thesis/` (no code references → safe).
   - Phase 3 — `docs/` reorg (git mv into subfolders) + dedup the duplicated design docs.
   - Phase 4 — update **all cross-references**: CLAUDE.md path refs, doc-to-doc links, `.claude/` graphify config,
     new `README.md` + `docs/README.md` index.
3. **Verify after each phase:** `pytest` still green (code untouched by design), scripts still run (`bash -n`),
   `graphify update .` clean. Nothing in `code/` moves, so tests can't break from the moves.
4. **`git mv` throughout** (preserves history/blame).

**Net:** code stays put (safe), everything else gets one clear home, duplicates collapse, dead code goes. The repo
becomes: *code in `code/`, the story in `docs/roadmap`, the how in `docs/plans`+`docs/specs`, the findings in
`docs/results`, the reading in `docs/research`, the ops in `docs/runbooks`, the dissertation in `thesis/`.*

---

## 7. Code structure — separate the two models clearly (your adjustment)

**Reality first (why this is DRY-sensitive):** EventSSM and PureSSM are the *same detector* — identical YOLO-PAFPN
neck, YOLOX head, Mamba **temporal** path, data pipeline, and evaluator — differing in **one thing only: the
spatial mixer** (ResNet-18 vs BiMamba). That shared-everything design *is* the controlled experiment, so we
separate **what actually differs** (the spatial backbones) and keep the rest shared — we do **not** duplicate the
model end-to-end (that would fight DRY and break the "only the backbone changed" claim).

**Proposed `code/event_ssm/` layout:**
```
code/event_ssm/
├── models/                ← the two contributions, clearly separated
│   ├── eventssm/             ResNet-18 spatial backbone (CNN)     [from backbone/resnet_spatial.py]
│   └── puressm/              BiMamba spatial backbone (SSM)       [from spatial/]
├── backbone.py            ← shared recurrent skeleton hosting a spatial mixer + temporal  [from backbone/resnet_mamba.py]
├── temporal/              ← Mamba temporal — SHARED by both models
├── integration/           ← RVT register/glue (registers both)   — SHARED
├── benchmark/  · viz/     ← efficiency harness · EventCV rendering — SHARED
├── configs/
│   ├── eventssm/             resnet_mamba_yolox + experiment/gen1/resnet_mamba
│   └── puressm/              puressm_yolox + experiment/gen1/puressm   (symlink-aware — see below)
├── scripts/               ← runnable scripts (optionally grouped: scripts/{cloud,stage09,stage10,…})
├── tests/
│   ├── models/
│   │   ├── eventssm/         test_resnet_mamba, …
│   │   └── puressm/          test_bimamba_spatial, …
│   └── core/                temporal/scan · integration · benchmark tests (shared)
└── proofs/
```

**Honest note:** the per-model folders are intentionally *thin* (just the spatial backbone) — that's the
controlled-experiment design showing through; it's a feature (DRY), not a gap to fill by duplicating shared code.

**This is a higher-risk phase than the docs moves — and how we de-risk it:**
- Reorganising the package **breaks imports** (`event_ssm.spatial` → `event_ssm.models.puressm`, etc.), the
  **hardcoded `code/event_ssm/...` paths** in ~40 scripts + CLAUDE.md, and the **Hydra config symlinks**
  (`external/` → `configs/`). All fixable, but must be done carefully.
- **De-risk:** do it *only* on the green Stage-16 baseline → update imports (find/replace) + script paths →
  re-point the config symlinks (`docs/patches`) → **the full test suite is the safety net**: if
  `pytest code/event_ssm/tests/` stays green, the reorg is correct. One phase, one commit, verify, `graphify update .`.
  This becomes **Phase 5** (after the doc moves), because it's the riskiest.

**Open naming choice:** `models/{eventssm,puressm}/` (by model) vs `spatial/{resnet,bimamba}/` (by architecture).
Leaning model-name since you asked for *model* separation — tell me which reads better.

---

## 8. Everything else — the full-repo sweep (papers, CLAUDE.md, untracked docs)

**Research papers — `papers_correct/` (gitignored, ~40 PDFs):** already topic-organised (`01_event_cameras`,
`02_ssm_mamba`, `04_uav_navigation`, `05_datasets`) — good structure, ugly name, wrong place.
→ **`thesis/references/papers/{01_event_cameras, …}`** with `references.bib` beside it (the bibliography those PDFs
back). Keep **gitignored** (big binaries — the organisation lives in the filesystem + the tracked `.bib`). Drop the
`_correct` suffix. *(Decision for you: `thesis/references/` vs `docs/research/papers/` — I lean `thesis/references/`
since they're the dissertation's cited sources, paired with the `.bib`.)*

**Two `CLAUDE.md` files — NOT duplicate clutter (keep both):**
- `./CLAUDE.md` — main project instructions; Claude Code reads it from repo root → **must stay at root**.
- `./.claude/CLAUDE.md` — a tiny graphify-skill trigger → belongs in `.claude/`.
Both functional and correctly placed. (Optional: fold the 2-line `.claude/` one into root — low value, leave it.)

**Untracked docs floating in the tree — give homes + commit (part of reaching the clean baseline):**
- `docs/DeepResearch_Loihi_SpikingSSM.md`, `docs/Katana_Migration_GapAnalysis.md` → `docs/research/` — commit
  (finished research, left uncommitted "for review").
- `docs/superpowers/plans/2026-07-15-stage16-pillars-visuals.md` → `docs/plans/` — commit (Stage-16 plan).
- `docs/Supervisor_Update_2026-07-15.md` → `docs/notes/` (or `thesis/admin/`) — commit.
- `docs/Repo_Restructure_Proposal.md` (this file) → `docs/plans/` — commit.
- `PLAN_FIXES_FOR_CLAUDE.md` (root) → **stays untracked / delete when done** (the "never-commit" scratch file).

**Gitignore stays the boss for bulk:** `external/` (vendored RVT), `data/` (datasets), `checkpoints/`, `results/`
bulk, `papers/` PDFs, `.pytest_cache`, `proofs/out/*.mp4` — all stay gitignored. Tidying these is zero-risk to git.

**Key distinction for the whole job:** *tracked* files (code, prose) move via `git mv` (history preserved).
*Untracked/gitignored* files (papers, checkpoints, data) move via plain `mv` — git neither sees nor cares. Both get
tidied; only the tracked moves touch history. So the "clean up everything" splits cleanly into: (a) git-tracked
reorg (history-aware), (b) filesystem tidy of gitignored bulk (free).

## 9. Final target — the whole repo at a glance

```
thesis-ssm-event-cameras/
├── README.md · CLAUDE.md · .gitignore          (root = 3 files, not 10+)
├── code/event_ssm/{models/{eventssm,puressm}, backbone.py, temporal, integration, benchmark, viz, configs, scripts, tests, proofs}
├── env/                                          (.yml + requirements)
├── docs/{roadmap, results, plans, specs, research, runbooks, notes, patches, README.md}
├── thesis/{latex, diagrams, admin, references/{papers/, references.bib}}
├── results/ · checkpoints/ · data/ · external/ · graphify-out/   (gitignored bulk, tidied on disk)
└── .claude/ · .superpowers/                      (tooling)
```
