# Stage 18 — SpikingSSM backbone integration (design)

**Date:** 2026-10-06 · **Status:** approved · **Precedes:** Stage 19 (smoke) → Stage 20a (25k short run)
**Context:** Thesis-C plan §3 (choice C), `docs/notes/Stage17_build_notes.md` ("Next — Stage 18"),
Stage-12 precedent (`docs/plans/2026-07-11-stage12-puressm-integration.md`).

## 1. Goal

Make the Stage-17 `SpikingSSMBlock` trainable/evaluable inside the frozen RVT pipeline, selectable as

```
model=rnndet +experiment/gen1=spikingssm
```

with every Stage-19/20 ablation arm expressible as a CLI override
(e.g. `model.backbone.spiking.output_mode=analog`). No training in this stage.

## 2. Controlled-experiment constraints

* Skeleton = **PureSSM** (BiMamba spatial stays ANN; only the temporal readout spikes — choice C).
* `models/eventssm/`, `models/puressm/`, `temporal/`, `backbone/resnet_mamba.py`: **byte-identical**.
* The only edit to an existing file is an **additive** `"SpikingSSM"` branch in
  `integration/register.py` (builder + config-modifier name tuple). Existing branches unchanged.
* Training recipe byte-identical to `experiment/gen1/puressm.yaml` (enforced by a test).

## 3. Components

### 3.1 `models/spikingssm/backbone.py` — `SpikingSSMBackbone(ResNetMambaBackbone)`

* Constructor: all `ResNetMambaBackbone` args (with `spatial=` injected, as PureSSM) plus
  `spiking_stages: tuple` and `lif_kwargs: dict` (+ `residual: bool`).
* Validation: `set(spiking_stages) ⊆ set(temporal_stages)`, else `ValueError` naming both tuples.
* After `super().__init__`, each stage in `spiking_stages` has its `self.temporal[str(s)]` replaced
  by `SpikingSSMBlock(d_model=stage_dim, d_state=d_state, num_layers=num_layers_per_stage,
  residual=residual, **lif_kwargs)`. Non-spiking temporal stages keep the parent's
  `MambaTemporalBlock` (the de-risking ladder: `(4,)` → `(3,4)` → `(2,3,4)`).
* `forward` overridden: identical loop to the parent, but per stage chooses the state helpers by
  block type:
  * spiking stage: state `(mamba_state_list, mem)`;
    `_spk_state_from_bmajor(state_b, B, hw)` → `(_state_from_bmajor(m, B, hw), mem.reshape(B*hw, C))`,
    `_spk_state_to_bmajor(state, B, hw)` → `(_state_to_bmajor(m, B, hw), mem.reshape(B, hw, C))`;
    `None` passes through.
  * non-spiking stage: the parent's helpers, imported unmodified.
* `spiking_stages=()` ⇒ numerically the PureSSM backbone (same modules, same forward path).
* `spiking_stats()` → `{stage: {rate, beta_mean, beta_min, beta_max, thr_mean}}` (floats) for the monitor.

RVT's `RNNStates.recursive_detach/recursive_reset` already recurse through lists/tuples of tensors,
so the nested state needs no RVT change; `mem` reset-to-zero on sequence boundaries is the correct
LIF reset.

### 3.2 `lif.py` fix (inside the new package)

`last_firing_rate` currently calls `float(...)` every forward → a host-device sync per stage per
step, which would contaminate Stage-22 latency. Store a detached 0-d tensor
(`_last_firing_rate`); `last_firing_rate` becomes a property converting on read. Outputs unchanged;
existing Stage-17 tests must pass untouched.

### 3.3 Dispatch — `integration/register.py` (additive)

* Builder: `backbone_cfg.name == "SpikingSSM"` → build `BiMambaSpatialStages` with exactly the
  PureSSM-branch arguments, then `SpikingSSMBackbone(..., spatial=spatial,
  spiking_stages=tuple(cfg.spiking.spiking_stages), residual=cfg.spiking.residual,
  lif_kwargs={output_mode, beta, threshold, alpha, learn_beta, learn_threshold, reset, detach_reset})`.
  `PURESSM_MONITOR=1` attaches the spatial monitor as for PureSSM; `SPIKING_MONITOR=1` attaches 3.4.
* Modifier: `"SpikingSSM"` added to the handled-name tuple (same in_res_hw / num_classes logic).

### 3.4 `integration/monitors.py` — `attach_spiking_monitor(backbone, every_n=200)` (additive function)

Forward hook on the backbone; reads `backbone.spiking_stats()`. Every `every_n` calls logs per spiking stage: firing rate, mean/min/max
learned β, mean threshold → wandb (`commit=False`) + one printed `[spk-monitor]` line. Warns on
**silence** (rate < 0.01) and **saturation** (rate > 0.90). Same `every_n >= 1` guard and
exception-suppression contract as the spatial monitor. Cadence env: `SPIKING_MONITOR_EVERY`.

### 3.5 Configs (new files) + symlinks

* `configs/spikingssm_yolox/default.yaml` = `puressm_yolox/default.yaml` with `name: SpikingSSM` and:
  ```yaml
  spiking:
    output_mode: spike        # spike | graded | analog (analog = control arm, ≈ PureSSM)
    spiking_stages: [2, 3, 4] # ⊆ in_stages; de-risking ladder [4] → [3,4] → [2,3,4]
    beta: 0.9
    threshold: 1.0
    alpha: 2.0                # arctan surrogate sharpness
    learn_beta: True
    learn_threshold: False
    reset: subtract           # subtract | zero
    detach_reset: True
    residual: False           # escape hatch — must be reported if used
  ```
* `configs/experiment/gen1/spikingssm.yaml` = `puressm.yaml` with `/model/spikingssm_yolox: default`.
* Symlinks into `external/ssms_event_cameras/RVT/config/{model/spikingssm_yolox,experiment/gen1}/`;
  re-create commands appended to `docs/patches/README.md` (Stage 18 section).

## 4. Tests (`code/event_ssm/tests/models/spikingssm/`)

**CPU:**
1. `_spk_state_*` round-trip (incl. `None` passthrough) on random tensors.
2. Subset validation raises `ValueError`.
3. Hydra compose of `model=rnndet +experiment/gen1=spikingssm` succeeds; `backbone.name == "SpikingSSM"`;
   all `spiking.*` keys present.
4. Recipe parity: `spikingssm.yaml` == `puressm.yaml` outside the `defaults` model group;
   model config == PureSSM's outside `name` and `spiking`.
5. Monitor: `every_n=0` raises; silence/saturation warnings fire on a stub backbone.
6. `last_firing_rate` still returns a float matching the previous semantics.

**GPU (`-m gpu`):**
7. Builder dispatch returns `SpikingSSMBackbone` with `SpikingSSMBlock` exactly on `spiking_stages`.
8. Forward+backward on `(L=3, B=2, 20, 256, 320)`; finite loss; grad reaches spatial stem weights.
9. Streaming parity: two clips with carried state == one concatenated clip (per output mode).
10. `spiking_stages=()` output == PureSSM backbone with identical weights (`torch.equal`/allclose).
11. RVT `recursive_detach` + `recursive_reset` on the returned states: no error, mem zeroed.

All GPU tests run in one session together with the 9 pending Stage-17 GPU tests on the idle 5070 Ti
(user runs — terminal policy).

## 5. Out of scope

Stage-19 overfit smoke, any training run, a spiking-specific eval script (existing launchers are reused
via the Hydra override), SOP counting (Stage 22), NIR export (Stage 23).

## 6. Risks

* Python LIF loop over L=21 per stage: acceptable for training throughput; if Stage 19 shows it
  dominates step time, a `torch.compile`/scan rewrite is a Stage-19 follow-up, not a blocker.
* `output_mode=spike` + `residual=False` hands PAFPN binary maps (Stage-17 risk 1). Mitigation is the
  ladder + `analog` diagnostic arm, already exposed by config.
