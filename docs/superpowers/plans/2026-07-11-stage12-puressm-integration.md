# Stage 12 — PureSSM Integration + Overfit Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PureSSM selectable through the unmodified RVT training stack (`model=rnndet +experiment/gen1=puressm`) and prove the full pipeline learns by overfitting one real Gen1 batch.

**Architecture:** Three additive touches, mirroring the Stage-4 ResNetMamba pattern exactly: (1) a `PureSSM` branch in `integration/register.py`'s monkeypatched builder + config-modifier; (2) a tracked Hydra config pair (`configs/puressm_yolox/default.yaml`, `configs/experiment/gen1/puressm.yaml`) symlinked into the gitignored RVT config tree; (3) the Stage-5 overfit-smoke harness parameterized by experiment and re-run for PureSSM. Zero edits to files under `external/` (symlink additions are the established mechanism, recorded in `docs/patches/`).

**Tech Stack:** Hydra/OmegaConf compose, PyTorch Lightning (RVT's trainer), the Stage-11 `BiMambaSpatialStages` + `ResNetMambaBackbone(spatial=…)` seam, pytest.

## Global Constraints

- `PY=/home/ghost/miniforge3/envs/events_signals/bin/python`; CUDA-only tests on the idle RTX 5070 Ti.
- **Controlled-experiment invariant:** no file content under `external/` may change; `code/event_ssm/spatial/`, `temporal/`, and the ResNetMamba code paths are frozen — this stage only ADDS dispatch/config/scripts. The experiment yaml must mirror `configs/experiment/gen1/resnet_mamba.yaml` byte-for-byte except the model-group default (recipe comparability).
- Spec §3 registration contract: dispatch key `backbone_cfg.name == "PureSSM"`; temporal blocks only on `in_stages` [2,3,4]; padding multiple-of-32 → in_res_hw (256,320) for Gen1.
- Spatial hyperparameters (spec §4.2/§4.3, locked): depths (2,2,8,2), spatial d_state 16, drop_path_rate 0.1, temporal d_state 64 — config keys: `depths`, `spatial_d_state`, `drop_path_rate`, `checkpoint_blocks` (default False).
- **Terminal policy:** implementer agents may run unit tests and one ≤10-epoch mini-smoke; the OFFICIAL 150-epoch overfit run is executed by the USER (command handed over, output pasted back).
- NEVER pip install. NEVER commit red. Conventional commits, no assistant names. Never commit: `PLAN_FIXES_FOR_CLAUDE.md`, `docs/DeepResearch_Loihi_SpikingSSM.md`, `docs/Katana_Migration_GapAnalysis.md`.
- Exit gate (roadmap Stage-12 row): Hydra-selected PureSSM overfits one real batch ≥3× loss reduction, no NaN; integration tests green; full suite green.

## File Structure

```
code/event_ssm/integration/register.py            # MODIFY: PureSSM branch (builder + modifier condition)
code/event_ssm/integration/smoke_harness.py       # MODIFY: experiment kwarg (default "resnet_mamba")
code/event_ssm/configs/puressm_yolox/default.yaml # CREATE: tracked model config
code/event_ssm/configs/experiment/gen1/puressm.yaml # CREATE: tracked experiment config
external/.../config/model/puressm_yolox/default.yaml     # SYMLINK (gitignored tree)
external/.../config/experiment/gen1/puressm.yaml          # SYMLINK (gitignored tree)
docs/patches/README.md                            # MODIFY: record the two symlinks for re-clone
code/event_ssm/proofs/smoke_overfit_puressm.py    # CREATE: Stage-12 overfit smoke
code/event_ssm/tests/test_register_puressm.py     # CREATE: registration + compose tests
```

---

### Task 1: `PureSSM` branch in `register.py`

**Files:**
- Modify: `code/event_ssm/integration/register.py:22-35` (builder) and `:59` (modifier condition)
- Test: `code/event_ssm/tests/test_register_puressm.py` (create)

**Interfaces:**
- Consumes: `ResNetMambaBackbone(in_channels, pretrained, d_state, num_layers_per_stage, temporal_stages, spatial)` (Stage-11 seam); `BiMambaSpatialStages(in_channels, depths, d_state, d_conv, expand, headdim, drop_path_rate, checkpoint_blocks)`.
- Produces: `build_recurrent_backbone(cfg)` returning a PureSSM backbone for `cfg.name == "PureSSM"`; modifier handling both names. Task 2's compose test and Task 3's smoke rely on this dispatch.

- [ ] **Step 1: Write the failing tests**

`code/event_ssm/tests/test_register_puressm.py`:
```python
# Stage 12: PureSSM registration through the RVT monkeypatch (spec §3, roadmap Stage-12)
import torch
from omegaconf import OmegaConf


def _register():
    from event_ssm.integration.smoke_harness import setup_paths, register
    setup_paths()
    register()


def _puressm_cfg(**kw):
    base = dict(name="PureSSM", input_channels=20, d_state=64, num_layers_per_stage=1,
                in_stages=[2, 3, 4], depths=[2, 2, 8, 2], spatial_d_state=16,
                drop_path_rate=0.1, checkpoint_blocks=False)
    base.update(kw)
    return OmegaConf.create(base)


def test_builder_dispatch_puressm(device):
    _register()
    import models.detection.recurrent_backbone as rb
    from event_ssm.spatial import BiMambaSpatialStages
    bb = rb.build_recurrent_backbone(_puressm_cfg())
    assert isinstance(bb.spatial, BiMambaSpatialStages)
    assert bb.spatial.depths == (2, 2, 8, 2)
    assert set(bb.temporal.keys()) == {"2", "3", "4"}


def test_builder_config_keys_flow(device):
    _register()
    import models.detection.recurrent_backbone as rb
    bb = rb.build_recurrent_backbone(_puressm_cfg(depths=[1, 1, 2, 1], spatial_d_state=8,
                                                  checkpoint_blocks=True))
    assert bb.spatial.depths == (1, 1, 2, 1)
    assert bb.spatial.checkpoint_blocks is True
    assert bb.spatial.stages[0][0].scan.d_state == 8


def test_resnet_mamba_branch_unaffected(device):
    _register()
    import models.detection.recurrent_backbone as rb
    from event_ssm.backbone.resnet_spatial import ResNetSpatialStages
    cfg = OmegaConf.create(dict(name="ResNetMamba", input_channels=20, pretrained=False,
                                d_state=64, num_layers_per_stage=1, in_stages=[2, 3, 4]))
    bb = rb.build_recurrent_backbone(cfg)
    assert isinstance(bb.spatial, ResNetSpatialStages)


def test_puressm_backbone_forward_contract(device):
    _register()
    import models.detection.recurrent_backbone as rb
    torch.manual_seed(0)
    bb = rb.build_recurrent_backbone(_puressm_cfg()).to(device)
    x = torch.randn(2, 1, 20, 256, 320, device=device)
    feats, states = bb(x, None)
    assert feats[2].shape == (2, 1, 128, 32, 40)
    assert len(states) == 4 and states[0].shape == (1, 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest code/event_ssm/tests/test_register_puressm.py -v`
Expected: 4 FAIL — the stock fallback raises `NotImplementedError` (builder doesn't know `PureSSM`).

- [ ] **Step 3: Implement the branch**

In `register.py`, inside `patched(backbone_cfg)` (after the existing `ResNetMamba` block, before `return orig(...)`):
```python
        if backbone_cfg.name == "PureSSM":
            # Stage 12: fully-pure spatial BiMamba injected into the same recurrent skeleton.
            # Temporal path/config identical to ResNetMamba (controlled experiment, spec §3).
            from event_ssm.spatial import BiMambaSpatialStages
            in_stages = backbone_cfg.get("in_stages", None)
            temporal_stages = tuple(in_stages) if in_stages is not None else (2, 3, 4)
            spatial = BiMambaSpatialStages(
                in_channels=backbone_cfg.input_channels,
                depths=tuple(backbone_cfg.get("depths", (2, 2, 8, 2))),
                d_state=backbone_cfg.get("spatial_d_state", 16),
                drop_path_rate=backbone_cfg.get("drop_path_rate", 0.1),
                checkpoint_blocks=backbone_cfg.get("checkpoint_blocks", False),
            )
            return ResNetMambaBackbone(
                in_channels=backbone_cfg.input_channels,
                d_state=backbone_cfg.get("d_state", 64),
                num_layers_per_stage=backbone_cfg.get("num_layers_per_stage", 1),
                temporal_stages=temporal_stages,
                spatial=spatial,
            )
```
And in `register_config_modifier()`'s `patched_modify`, change the dispatch line to:
```python
        if mdl.get("name") == "rnndet" and mdl.backbone.get("name") in ("ResNetMamba", "PureSSM"):
```
and the print to `print(f"[{mdl.backbone.name}] set in_res_hw={tuple(mdl_hw)}, num_classes={num_classes}")`. Update the module docstring's first lines to say it registers "the drop-in ResNetMamba and PureSSM backbones". Do not rename any function (launchers import `register_resnet_mamba`).

- [ ] **Step 4: Run to verify pass**

Run: `$PY -m pytest code/event_ssm/tests/test_register_puressm.py -v` → 4 passed.
Then the neighbors: `$PY -m pytest code/event_ssm/tests/test_register.py code/event_ssm/tests/test_resnet_mamba.py -q` → all pass (regression).

- [ ] **Step 5: Commit**

```bash
git add code/event_ssm/integration/register.py code/event_ssm/tests/test_register_puressm.py
git commit -m "feat(stage12): PureSSM branch in RVT registration (builder dispatch + config modifier)"
```

---

### Task 2: Hydra config pair, symlinks, compose test

**Files:**
- Create: `code/event_ssm/configs/puressm_yolox/default.yaml`, `code/event_ssm/configs/experiment/gen1/puressm.yaml`
- Modify: `code/event_ssm/integration/smoke_harness.py:28,40` (experiment kwarg), `docs/patches/README.md` (append symlink entry)
- Symlinks (gitignored tree): `external/ssms_event_cameras/RVT/config/model/puressm_yolox/default.yaml`, `external/ssms_event_cameras/RVT/config/experiment/gen1/puressm.yaml`
- Test: `code/event_ssm/tests/test_register_puressm.py` (append)

**Interfaces:**
- Consumes: Task 1's dispatch.
- Produces: `compose_smoke_config(..., experiment="puressm")` returning a config with `model.backbone.name == "PureSSM"`; runtime selection `model=rnndet +experiment/gen1=puressm`. Task 3's smoke consumes the kwarg.

- [ ] **Step 1: Append the failing compose test**

```python
def test_compose_selects_puressm_and_sets_hw(device):
    from event_ssm.integration.smoke_harness import compose_smoke_config
    cfg = compose_smoke_config(experiment="puressm")
    assert cfg.model.backbone.name == "PureSSM"
    assert tuple(cfg.model.backbone.in_res_hw) == (256, 320)   # modifier ran (multiple-of-32 pad)
    assert cfg.model.head.num_classes == 2
    assert cfg.model.backbone.spatial_d_state == 16
    assert list(cfg.model.backbone.depths) == [2, 2, 8, 2]
```

Run: `$PY -m pytest code/event_ssm/tests/test_register_puressm.py::test_compose_selects_puressm_and_sets_hw -v`
Expected: FAIL — `compose_smoke_config() got an unexpected keyword argument 'experiment'`.

- [ ] **Step 2: Parameterize the smoke harness**

In `smoke_harness.py`, change the signature and the experiment override line:
```python
def compose_smoke_config(dataset_path=None, max_epochs=50, batch_size=2, extra_overrides=None,
                         experiment="resnet_mamba"):
```
```python
        f"+experiment/gen1={experiment}",      # resnet_mamba (Stage 5) | puressm (Stage 12)
```

- [ ] **Step 3: Create the tracked model config**

`code/event_ssm/configs/puressm_yolox/default.yaml`:
```yaml
# @package _global_
# Canonical (tracked) Hydra model config for the fully-pure PureSSM backbone (Stage 12).
# Symlinked into the (gitignored) RVT config tree at
# external/ssms_event_cameras/RVT/config/model/puressm_yolox/default.yaml.
# Mirrors resnet_mamba_yolox/default.yaml; ONLY the backbone block differs (controlled experiment).
defaults:
  - override /model: rnndet

model:
  backbone:
    name: PureSSM
    compile:
      enable: False
      args:
        mode: reduce-overhead
    input_channels: 20         # stacked_histogram: 2 pol x 10 bins (matches baseline gen1.yaml)
    d_state: 64                # TEMPORAL Mamba-2 blocks (identical to ResNetMamba — never a confound)
    num_layers_per_stage: 1
    in_stages: [2, 3, 4]       # temporal blocks ONLY on FPN-consumed stages (Finding §8)
    depths: [2, 2, 8, 2]       # SPATIAL BiMamba pyramid depths (spec §4.3)
    spatial_d_state: 16        # spatial scans need small state (VMamba evidence, spec §4.2)
    drop_path_rate: 0.1
    checkpoint_blocks: False   # local-16GB training fallback; cloud RTX 5090 32GB leaves False
  fpn:
    name: PAFPN
    compile:
      enable: False
      args:
        mode: reduce-overhead
    depth: 0.67                # round(depth * 3) == num bottleneck blocks
    in_stages: [2, 3, 4]
    depthwise: False
    act: "silu"
  head:
    name: YoloX
    compile:
      enable: False
      args:
        mode: reduce-overhead
    depthwise: False
    act: "silu"
  postprocess:
    confidence_threshold: 0.1
    nms_threshold: 0.45
```

- [ ] **Step 4: Create the tracked experiment config**

`code/event_ssm/configs/experiment/gen1/puressm.yaml` — copy `configs/experiment/gen1/resnet_mamba.yaml` verbatim and change ONLY the header comment and the defaults line:
```yaml
# @package _global_
# Canonical (tracked) Hydra EXPERIMENT config for the PureSSM backbone on Gen1 (Stage 12).
# Byte-identical to experiment/gen1/resnet_mamba.yaml except the model group (recipe comparability).
# Selected at runtime exactly like the baseline:  model=rnndet +experiment/gen1=puressm
# Symlinked into the (gitignored) RVT tree at
# external/ssms_event_cameras/RVT/config/experiment/gen1/puressm.yaml.
defaults:
  - /model/puressm_yolox: default
```
(then the `training:` block onward copied unchanged from `resnet_mamba.yaml` lines 11–46). Verify the parity after writing:
`diff <(tail -n +9 code/event_ssm/configs/experiment/gen1/resnet_mamba.yaml) <(tail -n +9 code/event_ssm/configs/experiment/gen1/puressm.yaml)` → empty output.

- [ ] **Step 5: Create the symlinks (established Stage-4 mechanism; gitignored tree)**

```bash
mkdir -p external/ssms_event_cameras/RVT/config/model/puressm_yolox
ln -s ../../../../../../code/event_ssm/configs/puressm_yolox/default.yaml \
      external/ssms_event_cameras/RVT/config/model/puressm_yolox/default.yaml
ln -s ../../../../../../code/event_ssm/configs/experiment/gen1/puressm.yaml \
      external/ssms_event_cameras/RVT/config/experiment/gen1/puressm.yaml
ls -la external/ssms_event_cameras/RVT/config/model/puressm_yolox/ external/ssms_event_cameras/RVT/config/experiment/gen1/ | grep puressm
```
Expected: both symlinks resolve (compare with the existing `resnet_mamba` links — same 6-level relative prefix).

- [ ] **Step 6: Record the symlinks in docs/patches**

Append to `docs/patches/README.md` (so a re-clone of `external/` can restore them):
```markdown
## Stage 12 (2026-07-11): PureSSM Hydra config symlinks
After any re-clone of external/, re-create:
```bash
mkdir -p external/ssms_event_cameras/RVT/config/model/puressm_yolox
ln -s ../../../../../../code/event_ssm/configs/puressm_yolox/default.yaml external/ssms_event_cameras/RVT/config/model/puressm_yolox/default.yaml
ln -s ../../../../../../code/event_ssm/configs/experiment/gen1/puressm.yaml external/ssms_event_cameras/RVT/config/experiment/gen1/puressm.yaml
```
```

- [ ] **Step 7: Run the compose test + full file**

Run: `$PY -m pytest code/event_ssm/tests/test_register_puressm.py -v` → 5 passed.

- [ ] **Step 8: Commit**

```bash
git add code/event_ssm/configs/puressm_yolox/ code/event_ssm/configs/experiment/gen1/puressm.yaml code/event_ssm/integration/smoke_harness.py code/event_ssm/tests/test_register_puressm.py docs/patches/README.md
git commit -m "feat(stage12): PureSSM Hydra config pair + RVT-tree symlinks + smoke-harness experiment kwarg"
```

---

### Task 3: PureSSM overfit smoke (script by agent; OFFICIAL run by USER)

**Files:**
- Create: `code/event_ssm/proofs/smoke_overfit_puressm.py`

**Interfaces:**
- Consumes: `compose_smoke_config(..., experiment="puressm")` (Task 2), the smoke dataset builder (`event_ssm.integration.make_smoke_dataset.build_smoke_dataset`, already exists, data at `data/gen1_smoke/`).
- Produces: `results/smoke_test/puressm_overfit_loss_curve.png` + a pass/fail gate (≥3× loss reduction, no NaN) — the Stage-12 exit gate.

- [ ] **Step 1: Write the smoke script** — copy `code/event_ssm/proofs/smoke_overfit.py` with exactly these deltas (everything else byte-identical; that file's in-code comments explain the overfit_batches/LR mechanics):
  - docstring: `"""Stage-12 overfit smoke: PureSSM (BiMamba spatial) through the REAL Lightning stack..."""`
  - compose call:
    ```python
    cfg = compose_smoke_config(max_epochs=MAX_EPOCHS, batch_size=2, experiment="puressm",
                               extra_overrides=["training.lr_scheduler.use=False",
                                                "training.learning_rate=1e-3",
                                                "model.backbone.checkpoint_blocks=True"])
    ```
    (checkpointing ON: the smoke runs locally on the 16 GB card; seq 21 × batch 2 fits comfortably at 8.55 GB-class usage.)
  - output: `OUT / "puressm_overfit_loss_curve.png"`, title prefix `"Stage 12 PureSSM overfit"`.
  - keep the ≥3× assertion and NaN guard unchanged.

- [ ] **Step 2 (agent): mini-smoke execution check ONLY**

Run: `$PY code/event_ssm/proofs/smoke_overfit_puressm.py 10`
Expected: completes without crash; prints loss points; the final `assert reduction >= 3.0` may FAIL at 10 epochs — that is acceptable for the execution check ONLY (note the observed reduction). If it crashes (OOM/NaN/shape), STOP and report BLOCKED with the traceback.

- [ ] **Step 3: Commit the script**

```bash
git add code/event_ssm/proofs/smoke_overfit_puressm.py
git commit -m "feat(stage12): PureSSM overfit smoke (Stage-5 harness, experiment=puressm, checkpointed)"
```

- [ ] **Step 4 (USER runs the official gate):** hand the user this command and wait for pasted output:

```bash
cd ~/Desktop/thesis-ssm-event-cameras
/home/ghost/miniforge3/envs/events_signals/bin/python code/event_ssm/proofs/smoke_overfit_puressm.py 150
```
Expected: `reduction >= 3.0`, `PASS`, curve PNG written (~10–25 min; from-scratch BiMamba may converge slower than pretrained ResNet did). **If the 150-epoch run lands <3×:** one retry at 300 epochs is sanctioned; still <3× → BLOCKED, escalate (learning problem, not a tuning knob).

- [ ] **Step 5: Commit the user-run artifact**

```bash
git add results/smoke_test/puressm_overfit_loss_curve.png
git commit -m "feat(stage12): PureSSM overfit smoke PASS — <reduction>x loss reduction on one real Gen1 batch"
```

---

### Task 4: Stage close-out

**Files:**
- Modify: `CLAUDE.md` (Stage-12 status appended to the Stage-11 sentence in Next Immediate Steps item 1), `docs/Stage11_build_notes.md` untouched — create `docs/Stage12_integration_notes.md` (short: what was wired, smoke numbers, curve reference)
- Run: `graphify update .`

- [ ] **Step 1:** Full suite: `$PY -m pytest code/event_ssm/tests/ -q` → everything green (87 pre-existing + 5 new = 92 expected, 1 gpu-deselected).
- [ ] **Step 2:** Write `docs/Stage12_integration_notes.md` (½ page: dispatch mechanism, config pair paths, symlink record pointer, smoke reduction number + PNG path, any deviations).
- [ ] **Step 3:** CLAUDE.md: append to the Stage-11 status sentence: `" **Stage 12 (integration+smoke) COMPLETE (<date>):** PureSSM selectable via model=rnndet +experiment/gen1=puressm; overfit smoke <reduction>x PASS; notes docs/Stage12_integration_notes.md."` Also update the test-count line `87 pass` → `92 pass`.
- [ ] **Step 4:** `graphify update .`
- [ ] **Step 5:** Commit: `docs(stage12): close-out — integration notes, status, suite 92 green`
- [ ] **Step 6:** Final whole-branch review (controller dispatches; superpowers:requesting-code-review template) → fix wave if needed → superpowers:verification-before-completion → merge to main per user instruction (superpowers:finishing-a-development-branch).

---

## Self-Review

1. **Spec coverage:** §3 registration contract → Task 1; config pair + symlinks + selection string → Task 2; roadmap Stage-12 exit gate (overfit ≥3×, tests green) → Tasks 3–4. U4 of the spec fully covered. ✅
2. **Placeholders:** none — the only templated values are the measured `<reduction>`/`<date>` in commit/doc text, fillable only at execution. The Task-3 "copy with deltas" instruction names the exact source file and every delta. ✅
3. **Type consistency:** `compose_smoke_config(experiment=...)` (Task 2) matches Task 3's call; config keys `depths/spatial_d_state/drop_path_rate/checkpoint_blocks` identical in Task 1's builder, Task 1's tests, and Task 2's yaml; `BiMambaSpatialStages` kwargs match the Stage-11 constructor (`d_state` is the spatial one at the constructor boundary, mapped from the `spatial_d_state` config key in the builder — deliberate, documented in the builder comment). ✅
