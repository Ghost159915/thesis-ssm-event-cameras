# PureSSMDetector Roadmap — Stages 11–16 (Milestones & Gates)

**Authoritative spec:** `docs/superpowers/specs/2026-07-11-puressm-backbone-design.md` (approved 2026-07-11)
**In-house baseline to beat:** EventSSMDetector — Gen1 test/AP 46.2 (AP_L 44.7), 70 Hz, 0.40 J/frame.
**Published baseline:** S5-RVT — 47.7 AP (AP_L 50.7), 51 Hz, 0.73 J/frame.

Each stage gets its own detailed implementation plan in `docs/superpowers/plans/` **written when the stage starts** (superpowers:writing-plans), executed task-by-task with superpowers:subagent-driven-development, gated by superpowers:requesting-code-review + superpowers:verification-before-completion before merge (superpowers:finishing-a-development-branch). Stage 11's plan exists: `2026-07-11-stage11-puressm-backbone.md`.

| Stage | Mirrors (EventSSM) | Deliverable | Exit gate (all must hold) |
|---|---|---|---|
| **11 — Backbone build** | Stage 3 | `code/event_ssm/spatial/` (BiMamba1DScan, BiMamba2DBlock, BiMambaSpatialStages); spatial-injection hook in `ResNetMambaBackbone`; probe + untrained-ERF figures | All new tests green, existing 60 unaffected; **probe gates: projected pipeline ≥ 51 Hz AND train step < 16 GB** (fail → fallback ladder, user decision); spatial params 8–16 M |
| **12 — Integration + smoke** | Stages 4+5 | `PureSSM` branch in `integration/register.py`; Hydra config pair `puressm_yolox/default.yaml` + `experiment/gen1/puressm.yaml` (symlinked); overfit-one-real-Gen1-batch smoke | Hydra-selected model trains on one batch to near-zero loss; RVT contract integration tests green; no `external/` edits |
| **13 — Short training** | Stage 6 | ~25k-step run with live monitors (per-stage feature norms, NaN guard) | val/AP in the EventSSM-short-run band (≈ 0.10–0.15 at comparable steps); no NaN/instability; monitors show no Mamba-R norm blow-up |
| **14 — Full run** | Stage 7 | 400k-step run, Stage-7 recipe byte-identical (OneCycle 2e-4, seq 21, batch 4, bf16-mixed, workers 2) — **user-run, foreground/tmux** | Run completes; best val/AP checkpoint identified; expected wall-time 40–50 h |
| **15 — Evaluation** | Stage 8 | Gen1 **test** AP + size-stratified + per-class table; verdict vs pre-registered bands (spec §1); trained-ERF figure | Results table committed; AP_L reported explicitly; interpretation written against the bands (incl. pretraining caveat if in 44–46.2) |
| **16 — Pillars + visuals** | Stages 9+10 | Stage-9 two-regime re-eval row; Stage-10 bench re-run row (launcher unsets `MAMBA_STEP_SCALE`; ~5 min); EventCV GT-vs-pred videos + large-car contact sheet (`docs/EventCV_Assessment.md` install plan); **upgrade-lever decision**: cross-scan on stages 3–4 iff probe/bench headroom AND accuracy warrants | Three pillar artifacts + videos/figures shipped; CLAUDE.md status + thesis-chapter notes updated |

## Standing constraints (apply to every stage)

- Controlled experiment: **only the spatial mixer changes.** Temporal blocks, LstmStates contract, PAFPN/YOLOX/losses, Gen1 tensors, evaluator, training recipe — untouched.
- Environment: torch 2.11.0+cu128 (sm_120); mamba-ssm 2.3.2.post1 + causal-conv1d 1.6.2.post1; **no new torch-dependent packages; `--no-deps --no-build-isolation` if ever needed**. `eventcv` (numpy-only) is the sole approved new install, Stage 16 only, downstream-of-science only.
- Visual proof per stage (user is a visual learner): every unit ships a figure/table to `code/event_ssm/proofs/out/` or `results/`.
- Non-obvious changes/bugfixes get documented in the stage doc, not just commits.
- Commits: conventional style, no assistant names.
