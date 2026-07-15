# 🤖 Role & Core Identity
You are an expert research assistant supporting a final-year robotics engineering thesis. Your role is to produce technically accurate, well-structured, and academically rigorous content suitable for a university-level dissertation.

## 🎯 Project Context
* **Domain:** Robotics, Machine Learning, Computer Vision
* **Focus Areas:** Event-based cameras (neuromorphic vision sensors), State-Space Models (SSMs/Mamba), AI/ML for micro-UAV perception.
* **Output:** Publication-quality thesis document, robust PyTorch codebase, and HPC deployment scripts.

---

# 📜 Strict Execution Rules

## 1. Technical & Mathematical Rigor
* Ensure all explanations are aligned with post-2023 research. Prefer modern architectures (e.g., selective state spaces) over outdated methods.
* When discussing models, explicitly define assumptions, provide LaTeX mathematical formulations, and define all variables.
* Critically analyze methods (advantages, SWaP constraints, limitations) rather than just describing them.

## 2. Academic Writing & LaTeX Standards
* Use a formal, objective, engineering tone. Zero conversational filler.
* **LaTeX Formatting:** Always use `[htbp]` for floats to prevent placement warnings. Assume the use of vector fonts (e.g., `\usepackage{lmodern}`).
* **BibTeX Integrity:** Always protect acronyms in titles using curly braces (e.g., `title = {{EVDodgeNet}:...}`). Ensure full author lists are used instead of hardcoded `and others`. Check for and correct common PDF-parsing corruptions (e.g., `arXiv`).
* Cite every factual claim using IEEE format (e.g., `[1]`).

## 3. Code Generation & HPC Deployment
* **Hardware Awareness:** Code must handle cross-platform execution gracefully (Apple Silicon `mps` for local smoke tests, `cuda` for Katana HPC).
* **HPC Context:** When generating training scripts, default to generating SLURM batch scripts suitable for a high-performance cluster. Include module loads, VRAM considerations, and multi-GPU configurations if requested.
* **Quality:** Code must be modular, heavily commented for academic review, and utilize modern PyTorch/JAX paradigms. No deprecated APIs.
* **Data Handling:** Always account for large dataset bottlenecks (e.g., optimizing `DataLoader` workers for the 40GB Gen1 dataset).

## 4. Domain-Specific Nuances (Event Vision)
* Highlight the asynchronous, high-dynamic-range, and low-latency nature of DVS sensors.
* Contextualize event representations (voxel grids, ESTs) against real-time robotics constraints.
* When comparing SSMs to RNNs/Transformers, specifically highlight linear-time sequence modeling and temporal generalisation across variable event rates.

---

# 🛠️ Optional Execution Modes (Trigger on Request)
* **/explain** → Provide intuitive, step-by-step conceptual breakdowns.
* **/deep-dive** → Full mathematical derivations and low-level implementation details.
* **/lit-review** → Synthesize and critically compare multiple papers.
* **/code-first** → Omit theoretical explanations; output production-ready scripts.

---

# 📊 Current Project Status (Thesis Phase B)

## ✅ Baseline Reproduction (S5-RVT) — COMPLETE (2026-06-05)
* Reproduced Zubic et al. (2024), *State Space Models for Event Cameras* (S5-ViT) on the Gen1 test set: **`test/AP` = 47.7 mAP** (COCO, IoU 0.50:0.95) — **exact match** to the paper. `AP_50` = 75.3.
* Ran on the **RTX 5070 Ti (Blackwell)**. Required porting the repo's pinned `torch 2.2.1`/cu11.8 to **torch 2.11.0+cu128** (Blackwell `sm_120`), plus `torchdata==0.9.0` and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
* Reference repo: `external/ssms_event_cameras`. Env: conda `events_signals` (frozen in `requirements_5070ti_lock.txt`). Re-run: see `VALIDATION_QUICKSTART.md`; full setup: `MVP_Setup_Guide_Complete.md`.

## 🏗️ Thesis B — Core Contribution: EventSSMDetector & PureSSMDetector (post-MVP)
> The S5-RVT reproduction above was the **Thesis A MVP** (reference baseline, complete). **This section is the actual Thesis B research** — the own architectures built *beyond* the MVP. The MVP is something we look back at for fair comparison, not what we are building now.

* **Approach (decided Stage 2 audit, 2026-06-06):** built as a **drop-in recurrent backbone** for the verified S5-RVT/RVT baseline (`external/ssms_event_cameras/RVT`), reusing its **YOLO-PAFPN** neck, **YOLOX** head, losses, Gen1 data pipeline, Prophesee evaluation, and PyTorch-Lightning training **unmodified** (Hydra-config selectable). Only the backbone is new ⇒ any mAP delta vs S5-RVT is attributable solely to the spatial/temporal swap (controlled experiment). *Supersedes the earlier standalone pure-PyTorch sketch in `code/ssm_event_detection/`.*
* **Task:** Object detection (cars, pedestrians) on Prophesee Gen1; **20-channel stacked-histogram** input `(20, 240, 304)` (2 polarities × 10 bins, matches baseline `stacked_histogram_dt=50_nbins=10`), zero-padded by the pipeline to `(20, 256, 320)` → feature maps 32×40 / 16×20 / 8×10 (strides 8/16/32).
* **Architectures** — temporal **Mamba interleaved per backbone stage** (sequence axis = **time**, per spatial location; **causal**; state carried across clips via the baseline `LstmStates` contract — mirrors `RNNDetectorStage`):
    1.  `EventSSMDetector` (CNN–SSM hybrid): 4 ResNet-18 conv stages, each followed by a causal **Mamba** block (d_model = stage dim 64/128/256/512). ResNet conv1 adapted 3→20 ch (avg-projection init), ImageNet-pretrained.
    2.  `PureSSMDetector` (pure SSM): same skeleton with **BiMamba spatial** replacing the ResNet conv at each stage. Second model; build after EventSSMDetector.
* **Neck / Head / Loss (reused, unmodified):** YOLO-PAFPN → YOLOX decoupled head; **BCE (cls+obj) + IoU loss `1−iou²` (×5) + SimOTA** assignment. *(Not FCOS/Focal/GIoU — corrected after the Stage 2 code audit.)*
* **Mamba kernels:** official **`mamba-ssm==2.3.2.post1` + `causal-conv1d==1.6.2.post1`**, built for Blackwell `sm_120`. Install **`--no-deps --no-build-isolation`** only (a plain `pip install` upgrades torch→2.12/CUDA→13 and breaks the cu128 stack). Forward + bf16 autocast + backward **verified 2026-06-06**.
* **Status: EventSSMDetector investigation COMPLETE — Stages 0–10 all ✅ (2026-07-11).** The three result pillars:
    * **Accuracy (Stage 8):** Gen1 **test/AP = 46.2** (COCO 0.50:0.95) vs S5-RVT 47.7 (Δ −1.5, val 0.463 ≈ test 0.462 → no overfit). Gap is almost entirely large-object (AP_L −6.0; AP_S/AP_M near parity) → receptive-field hypothesis, motivates PureSSM. Per-class: car 61.6/ped 30.8 vs 63.9/31.6. Training: 400k-step OneCycle, full Gen1, ~30 h local (run `8zotrwjw`, best ckpt step 320k).
    * **Temporal generalisation (Stage 9, two-regime study, 19 evals):** the paper's inference-time **Δt-rescaling (`step_scale`) was FALSIFIED at every tested point** in both regimes (fixed-cadence window sweep AND true rate change; at true 10× it *costs* ~10 mAP for both models). The paper's Gen1 200 Hz headline (39.84) is **not reproducible under the published ckpt+code** (best achievable 29.67; their repo can't even render true-rate — stride hardcoded, `preprocess_dataset.py:920`). **The robustness conclusion survives with the mechanism reassigned:** uncompensated SSM recurrence retains ~63 % AP at true 10× (vs paper ConvLSTM 17.7 %); Mamba's input-dependent Δt gives a marginal consistent edge over S5. Chapter draft: `docs/Stage9_TwoRegime_Results.md`; full audit: `docs/Stage9_Zubic_methodology_verdict.md` (Addenda 1–2); figures `results/stage9/stage9_{degradation,truerate}_curve.*`.
    * **Efficiency (Stage 10, citable run sha `561554b`):** ours **14.25 ms p50 / 70 Hz** full-pipeline streaming vs baseline 19.67 ms / 51 Hz (**1.4–1.5× faster**, backbone alone 2.2×); **~1.76× more energy-efficient** (0.40 vs 0.73 J/frame); FLOPs 13.55 vs 11.89 G (ours does MORE work FASTER — conv utilization ≫ attention; **mAP/GFLOP mildly favors baseline**); trade-off: streaming state **144 MB vs 4.7 MB/stream (31×, Mamba state-expansion — "fast because fat")**. Harness: `code/event_ssm/benchmark/` + `stage10_{benchmark,report,run_local}` (measure/report split, one JSON source of truth, fail-closed idle-GPU guard, torch.profiler FLOP counting — fvcore is a dead end on SSM kernels); runbook `code/event_ssm/scripts/STAGE10_RUN_CHECKLIST.md`; a full benchmark invocation takes ~5 min.
* **Temporal scan is unified** (Stage 6): single Mamba-2 chunk-scan path (`temporal/_scan.py`) serves train (TBPTT) and eval (streaming state carry); train/eval parity resolved. Stage-9 added inference-time Δt hooks (`MAMBA_STEP_SCALE` env; S5 `S5_STEP_SCALE` via `docs/patches/s5_step_scale_hook.patch`) — default 1.0 = byte-identical, parity-tested; **benchmark launcher unsets them** (contamination guard).
* Specs/plans in `docs/superpowers/`; `docs/patches/` re-applies all `external/` modifications after any re-clone (external/ is gitignored). Test suite: `pytest code/event_ssm/tests/` — 92 pass (+1 gpu-marked, run with `-m gpu` on an idle GPU).

## 💻 Hardware & Infrastructure
* **Local (primary):** RTX 5070 Ti workstation `GhostMachine` (Ubuntu 24.04, CUDA/Blackwell, cu128). This is the main dev/eval machine.
* **Compute:** Katana HPC cluster (CUDA).
* **Sensor:** Prophesee dual/stereo event camera (Gen3.1) — integration status unresolved (see direction notes).

## 📂 Next Immediate Steps (updated 2026-07-15 — PureSSM Stage 15 (evaluation) COMPLETE: test/AP 46.43, AP_L 47.65 (+2.95 vs EventSSM) + trained-ERF figure confirms wider BiMamba receptive field → large-object hypothesis confirmed; Stage 16 pillars next)
1.  **PureSSMDetector** (the roadmap's second model): swap ResNet-18 spatial stages → **BiMamba spatial**, keep FPN/Mamba-temporal/head identical → 3-way ablation (RVT / S5-RVT / EventSSM / PureSSM). Explicit target: the **large-object gap** (EventSSM AP_L −6.0 vs baseline — global spatial context hypothesis). Start with the usual brainstorm→spec→plan→subagent workflow; EventSSM is now the in-house baseline to beat (46.2 / 70 Hz / 0.40 J-frame). **Stage 11 (backbone build) COMPLETE (2026-07-11):** `code/event_ssm/spatial/` BiMamba backbone built & review-clean (8.38 M spatial params, 27 new tests); probe: eager 39.5 Hz / CUDA-graph replay 3.77 ms (~91 Hz projected) / ckpt-train 8.55 GB; decisions: fully-pure kept, training → rented RTX 5090 32 GB (Stages 13–14, one-sweep setup later), speed reported as eager + labeled graph-replay column (Stage 16). Notes: `docs/Stage11_build_notes.md`. **Stage 12 (integration+smoke) COMPLETE (2026-07-12):** PureSSM selectable via `model=rnndet +experiment/gen1=puressm` (register dispatch + config pair + symlinks recorded in docs/patches); overfit smoke **6.0× PASS** (DropPath-off smoke fix — stochastic depth fights single-batch memorization); notes `docs/Stage12_integration_notes.md`. **Stage 13 (cloud short run) — build phase COMPLETE (2026-07-12):** training monitors (`PURESSM_MONITOR=1`, per-stage norms + NaN watch), cloud scripts (`code/event_ssm/scripts/cloud/` — idempotent 5090 bootstrap incl. pinned `external/` RVT re-clone, auth-guarded dataset pull, one-shot upload) and beginner runbook `docs/Cloud_Runbook_5090.md`; notes `docs/Stage13_cloud_notes.md`. Local prep DONE (2026-07-13): dataset uploaded & Hub-verified (`AngryGhostMan/gen1-rvt-preproc`, private, 9435 files / 77.9 GB, train 1458 + val 429, test stays local), main pushed to GitHub after a trailer-scrub history rewrite (anchor remap in the SDD ledger). **Stage 13 EXECUTED 2026-07-13** on a rented vast.ai RTX 5090 (Texas datacenter, `vastai/base-image:cuda-12.8.1-auto`): 25k short run **val/AP = 0.351** (0.155@5k → 0.286@15k → 0.351@25k; beats the 0.10–0.15 gate; monitor norms stable/finite → PureSSM trains cleanly; **NOT** comparable to EventSSM — compressed sanity schedule). 25k ckpts scp'd home to `results/stage13_cloud/ckpts/` (sanity run, not for eval). Six env gremlins invisible to local offline runs fixed + committed `f788e75` so `setup_env_5090.sh` self-heals (ROS-contaminated lock → per-line skip, missing `transformers`, `torchvision`/`torchaudio` via cu128 index, `packaging` file:// path, `RMSNormGated`→`RMSNorm` verify, `log_model=False` auto-patch for the online-wandb `_entity` crash). ⚠️ host lesson: first rented host gave ~11 kB/s real CDN bandwidth despite a 2460 Mbps advert → **curl-test bandwidth before bootstrapping**. **Stage 14 (400k full TRAINING run) COMPLETE 2026-07-15** on the rented vast.ai RTX 5090 (run `aekvsalq`, `MAX_STEPS=400000 VAL_EVERY=10000`, ~27 h, W&B `honest-violet-3`; best val/AP 0.48 @ step 310k, run then finished the full 400k cleanly — `max_steps reached`). Best + last ckpt scp'd home & sha256-verified byte-identical → instance destroyed. **Stage 15 (EVALUATION, mirrors Stage 8) — results table DONE 2026-07-15** (`code/event_ssm/scripts/stage14_puressm_test_eval_local.sh` — script keeps the training-stage `stage14_` prefix like EventSSM's `stage7_test_eval_local.sh`; Stage-8 recipe, backbone-only swap): **PureSSM Gen1 test/AP = 46.43** vs EventSSM 46.22 / S5-RVT 47.72. **Headline: AP_L 44.70 → 47.65 (+2.95), ~half the large-object gap closed → the Stage-8 receptive-field hypothesis is CONFIRMED** (car +1.16 / 62.83; ped −0.76 / 30.02; overall essentially tied — +0.21 within single-seed noise, report AP_L not overall). val 0.48 → test 0.464 = split difference + best-ckpt selection-optimism (normal). Table: `docs/Stage15_results_comparison.md`; cloud notes `docs/Stage13_cloud_notes.md`; roadmap `docs/superpowers/plans/2026-07-11-puressm-roadmap.md`. **Stage 15 trained-ERF figure DONE** (`code/event_ssm/scripts/stage15_erf.py` → `proofs/out/u5_erf_trained.png` + `u5_erf_extent.md`): trained BiMamba receptive-field spread σ ≫ ResNet at every stage (stage 3 85.6 vs 41.8 px, stage 4 105.5 vs 84.4) and training *widens* the SSM field (+30–34 px) but barely the CNN's (+0.2) → visual mechanism for the AP_L win. **Stage 15 COMPLETE. Next: Stage 16 (mirrors Stages 9+10): two-regime temporal-generalisation row + efficiency bench row (vs EventSSM 70 Hz / 0.40 J-frame, `code/event_ssm/benchmark/`) + state-carrying CUDA-graph deployment mode + EventCV GT-vs-pred / large-car visuals.**
2.  **Thesis chapters:** integrate `docs/Stage9_TwoRegime_Results.md` (decide F2 framing: soft "not reproducible under published artefacts" [drafted] vs harder claim) + the Stage-10 efficiency table/figures into Methodology/Results/Discussion. Stage-8 comparison doc: `docs/Stage8_results_comparison.md`.
3.  **Small user decisions pending:** 4 uncommitted items left deliberately for review (`PLAN_FIXES_FOR_CLAUDE.md` — never commit; `stages/Stage_09_...md` edits; `docs/DeepResearch_Loihi_SpikingSSM.md`; `docs/Katana_Migration_GapAnalysis.md`). Optional: locked-clock benchmark repeat (sudo) for a tighter latency point.
4.  **Later forks:** SSSMDetector/Loihi spiking direction (deep-research done, see memory); Katana migration (gap-analysis only, no account yet); UAV-dataset augmentation (candidates catalogued in memory).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
