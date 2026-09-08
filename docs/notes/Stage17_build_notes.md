# Stage 17 — Spiking SSM cell: build notes

**Thesis C · MMAN4953 · UNSW Sydney · Benas Vaiciulis** · built 2026-09-07
**Plan:** `docs/plans/2026-09-07-thesis-c-plan.md` §3 · **Design source:** `docs/research/Spiking_PureSSM_litreview_deepdive.md` §6.1 (choice C), App. A

## What was built

`code/event_ssm/models/spikingssm/` — a **new sibling package**; `models/eventssm/`, `models/puressm/`
and `temporal/` are **untouched** (verified: `git status code/` shows two new directories, nothing modified).

| File | Contents |
|---|---|
| `surrogate.py` | `ATanSpike` — Heaviside forward, arctan surrogate backward. Vendored, not imported from snnTorch. |
| `lif.py` | `LIFReadout` — LIF over the time axis. Pure PyTorch, **CPU-safe**. |
| `spiking_temporal.py` | `SpikingSSMBlock` = `MambaTemporalBlock` (composed, unmodified) + `LIFReadout`. |
| `__init__.py` | Eager LIF/surrogate exports; `SpikingSSMBlock` lazy (PEP 562) since it pulls CUDA-only `mamba_ssm`. |

## Decisions worth recording

1. **The LIF is vendored, not taken from snnTorch.** snnTorch remains the decided library
   (plan §2.4) for cross-validation and the Stage-23 NIR export, but the neuron itself is ~60 lines
   and vendoring it buys three things: **zero install risk** on the Blackwell cu128 stack, a state
   contract we own (needed for the clip carry), and a package that **imports and tests on CPU** — which
   is how all 31 CPU tests were run today without a GPU. Adding an snnTorch equivalence test is a
   cheap future check, not a dependency.

2. **CPU/GPU split is deliberate.** `surrogate.py` and `lif.py` import only torch; only
   `spiking_temporal.py` touches `mamba_ssm`. That is what makes `__init__.py`'s lazy export
   worthwhile and lets the logic be developed away from the workstation.

3. **`MambaTemporalBlock` is composed, never subclassed or edited.** The recurrence, the Stage-9
   `MAMBA_STEP_SCALE` hook and the unified chunk-scan train/eval parity all carry over untouched, so any
   measured delta vs EventSSM/PureSSM is attributable **solely** to the spiking readout. `test_ssm_path_
   matches_the_non_spiking_block` (gpu) asserts this directly: analog readout + beta→0 reproduces
   `MambaTemporalBlock` to 1e-3.

4. **Three output modes, because they are the ablation axis** — `spike` (binary, the neuromorphic
   target), `graded` (`spk * mem_pre`: the Loihi-2 graded-spike / SpikeYOLO I-LIF analogue, and the
   mitigation for the binary information bottleneck), `analog` (membrane, **the control arm** — A/B
   against `spike` measures the exact cost of spiking, which is the thesis question).

5. **`residual=False` by default.** The backbone *replaces* its stage features with the temporal output
   (`backbone/resnet_mamba.py` forward — no residual around the temporal block), so a binary readout
   hands the PAFPN binary feature maps. `residual=True` restores an analog bypass: a Stage-18
   de-risking lever, **not** a default, and it must be reported if used because it weakens the
   spiking claim.

## ⚠️ Non-obvious fix: beta saturation (caught by a test)

`beta` (the leak) is learnable and was originally `sigmoid(logit)`, on the reasoning that a sigmoid
structurally confines it to (0,1) so the recurrence cannot be trained into divergence.
`test_beta_stays_bounded_after_optimiser_steps` (20 SGD steps at lr=1e3, deliberately absurd)
**failed**: in float32 the sigmoid saturates to *exactly* 0.0 and 1.0, silently turning the neuron
into either a memoryless unit (beta=0) or a pure integrator (beta=1).

**Fix:** an epsilon-squeezed sigmoid, `beta = eps + (1-2*eps)*sigmoid(logit)` with `eps=1e-4`. The
affine rescale sits *outside* the sigmoid, so gradient is preserved everywhere (merely scaled by
1-2*eps) rather than clipped to zero as a `clamp` would do. The init logit is inverted through the
same transform so a requested `beta=0.9` still initialises at 0.9.

Threshold gets the equivalent treatment (clamped `>= 1e-3`) for the same reason.

## Test status

**31 CPU tests pass, 9 gpu-marked tests pending the 5070 Ti.** CPU tests were run in the cloud
container against a CPU-only torch build (the local machine has no torch and the workstation was not
reachable); re-run them in the `events_signals` env on the workstation to confirm.

```
pytest code/event_ssm/tests/models/spikingssm/            # 31 CPU tests
pytest code/event_ssm/tests/models/spikingssm/ -m gpu     # 9 GPU tests, needs an idle 5070 Ti
```

`test_spiking_temporal_cpu.py` substitutes a **stateful** stand-in for Mamba-2 (a leaky linear
recurrence, not an identity) and exercises the composition logic on CPU: state threading through
*both* halves of the state, the residual switch, firing-rate passthrough, `fold`/`unfold` re-export,
and gradient flow. It includes a negative control
(`test_carrying_no_state_differs_from_carrying_state`) so the split-equals-full test cannot pass by
silently dropping the state. Stubs are installed per-fixture and torn down, so they never leak into
the gpu tests.

`d_model=32` in those tests: `MambaTemporalBlock` asserts `(d_model*expand) % headdim == 0`, so with
the defaults (expand=2, headdim=64) the smallest legal width is 32. Real stage dims are 64/128/256/512.

## Exit gate — status

| Gate (plan §3, Stage 17) | Status |
|---|---|
| New tests green | ✅ 31/31 CPU |
| Existing 112 CPU + 3 GPU tests untouched and passing | ⚠️ **unverified** — needs the workstation (existing suite imports `mamba_ssm`) |
| No edits under `models/{eventssm,puressm}/` | ✅ verified via `git status` |
| Firing rate in a sane band (not 0 %, not ~100 %) | ⚠️ asserted in `test_output_is_binary_and_sparse` (gpu) — **pending real kernels** |

**⇒ Stage 17 is code-complete and CPU-verified; two gates need one run on the 5070 Ti to close.**

---

## Knobs — what to change to get different results

Every knob below is a constructor argument that already exists; none needs new code. Figure:
`thesis/diagrams/fig_spiking_architecture.png` (source `fig_spiking_architecture.py` — regenerates
itself from the real `LIFReadout`, so it cannot drift from the implementation).

### The full surface

| Knob | Default | What it controls | Expected impact |
|---|---|---|---|
| `output_mode` | `"spike"` | binary / graded / analog readout | **highest** |
| `threshold` | `1.0` | firing rate ⇒ sparsity ⇒ energy | **high** (the energy–accuracy knob) |
| `beta` | `0.9` | membrane leak ⇒ temporal memory length | medium–high |
| `learn_beta` | `True` | per-channel learnable decay | medium (see the multi-timescale test below) |
| `learn_threshold` | `False` | network picks its own operating point | medium |
| `residual` | `False` | analog bypass around the spiking block | high, but weakens the claim |
| `alpha` | `2.0` | surrogate-gradient sharpness | low (robustness check) |
| `reset` | `"subtract"` | keep vs discard the supra-threshold residue | low |
| `detach_reset` | `True` | surrogate noise through the reset path | low (literature is split) |
| `temporal_stages` | `(2,3,4)` | *which* stages get spiked (backbone arg) | high — the de-risking ladder |

### 1. `output_mode` — the biggest single lever

```python
SpikingSSMBlock(d_model=..., output_mode="spike")    # binary {0,1}      — the neuromorphic target
SpikingSSMBlock(d_model=..., output_mode="graded")   # spk * mem_pre     — Loihi-2 graded spikes
SpikingSSMBlock(d_model=..., output_mode="analog")   # membrane          — the CONTROL ARM
```

Running all three is not decoration — it **decomposes the cost of spiking** into *cost of sparsity*
(analog → graded) and *cost of binarisation* (graded → spike). No published SNN detector paper
provides that decomposition, and it is the thesis's central question stated as an experiment.

* `graded` is where the accuracy is most likely to come back: it is the hardware analogue of
  SpikeYOLO's integer-valued (I-LIF) trick, which is what took it to SNN SOTA on Gen1 — and it is
  the mechanism named in §3.3 of the INRC proposal, so a positive result there directly supports
  the Loihi argument.
* `analog` doubles as a **diagnostic**: it should land ≈ PureSSM (46.4). If it does not, the
  integration is wrong, not the science. Run it first at Stage 18.

### 2. `threshold` — the energy–accuracy Pareto

Higher threshold → sparser firing → fewer SOPs → less energy, at the cost of information. Sweeping
it produces the **energy–accuracy Pareto curve**, which is precisely the figure the Stage-22
efficiency pillar and the Loihi rationale both want. Do not report a single operating point.
`learn_threshold=True` lets the network find its own instead of being told.

### 3. `beta` — temporal memory, and a possible headline result

β→0 is memoryless; β→1 is a pure integrator (an IF neuron) with long memory and high firing.
The interesting experiment is **not the value but the learned distribution**: if the per-channel
betas spread out after training, the model has built itself a **multi-timescale filter bank**.
That would tie the spiking model directly into the thesis's rate-robustness story (Stage 9/16) and
is cheap to test — plot the learned `lif.beta` histogram per stage after Stage 20.

### 4. `temporal_stages` — the de-risking ladder

Temporal blocks live on stages 2, 3, 4. Spike **stage 4 only** → **3–4** → **all three**. If full
spiking costs too much accuracy, the partial version is a legitimate hybrid result (cf. Hybrid
Spiking ViT, ICML'25) rather than a failure. Start at the top of the ladder if Stage 20 is tight.

### 5. `residual` — the escape hatch

`residual=True` adds the analog input back around the spiking block. It will recover accuracy and it
**weakens the spiking claim**, so it is off by default and **must be reported if used**. This is the
Stage-20 lever if training stalls, not a free win.

### 6. Low-payoff robustness checks

`alpha` (surrogate sharpness — if results swing on it, training is fragile: that is the finding),
`reset="zero"` vs `"subtract"`, and `detach_reset` (currently `True`; SpikingJelly defaults to
`False`, so one A/B settles it for our setup).

### Beyond the current code — structural moves

1. **Choice A — spike the spatial BiMamba too.** A fully spiking model. Much harder: the spatial
   scan is *bidirectional* and its axis is not time, so "spiking" it is not well defined the way the
   temporal axis is (litreview §8, risk 3).
2. **Choice B / SiLIF — make the neuron's own dynamics BE the SSM.** The most novel option and the
   closest to "one model" rather than "SSM plus neuron"; also the most finicky.
3. **Multi-bit / integer spikes.** Sits between `spike` and `graded`; maps to Loihi 2's graded spikes
   with a bounded bit-width, which is arguably the most hardware-faithful setting of all.

### Recommended experiment order

1. **Output modes** (spike / graded / analog) — the headline decomposition, and `analog` validates the integration.
2. **Threshold sweep** — the energy–accuracy Pareto for the efficiency pillar.
3. **Stage ladder** — de-risking, and a publishable hybrid result if full spiking is too costly.
4. **Learned-beta distribution** — the multi-timescale story.

Everything else is a sanity check, not an experiment.

---

## Novelty positioning — hybrid vs full-spike (decided 2026-09-08)

**Question raised:** should the spiking model be made *fully* spiking, so the work does not resemble
already-published models and pipelines?

**Decision: keep it hybrid (choice C).** Going full-spike would make the work **more** derivative, not less.

### Why

* A full-spike detector on Gen1 is exactly what **SpikeYOLO, SpikeDet/SpikSSD and EMS-YOLO already are.**
  Going full-spike enters their leaderboard, on their axis, with a different backbone — an incremental
  contribution, competing against groups who have iterated for years, on one training run.
* The unoccupied intersection identified in our own lit review (`Spiking_PureSSM_litreview_deepdive.md` §5)
  is a spiking **structured SSM for event detection** — and that is unoccupied *either way*: structured
  spiking SSMs exist only on speech/LM tasks; a spiking Mamba exists only for video grounding; the Gen1 SNN
  detectors all use spiking CNNs or ViTs. **Full-spike buys no novelty and costs a great deal of risk.**
* The bidirectional **spatial** scan is genuinely hard to spike: spikes are causal events in time, and that
  scan's axis is not time (litreview §8, risk 3).

### The reframe — the hybrid IS the idea, not a compromise

The design principle is: **spike the axis that is actually time.**

Prior full-spike detectors must introduce an *artificial* timestep dimension — SpikeYOLO runs `T=5`, i.e.
5× the compute. We ride the clip-time unroll that already exists in the temporal block and pay a **zero**
timestep multiplier. No other detector can make that argument, because no other detector starts from a
temporal SSM. **State this explicitly in the thesis** — it converts a scoping decision into a claim.

### Three additions that would sharpen the novelty (ranked)

1. **Does spiking preserve rate-robustness?** — *highest value, cheapest, schedule it.*
   We own the most rate-robust detector measured (69.7 % retention at true 10×, Stage 16). **Nobody has
   asked whether spiking preserves or destroys that.** It connects the Thesis-B contribution directly to the
   spiking work, and the harness already exists (Stage 9/16 two-regime) — marginal cost is *evaluation only,
   no extra training*. A positive result is a strong argument for neuromorphic flight, where event rates
   vary constantly; a negative result is an equally publishable finding about spiking temporal models.
   **Do not leave this as a good intention — give it a stage and an exit gate.**

2. **The stage ladder as an experiment, not just de-risking.**
   Spike stage 4 only → 3–4 → all three. This measures *where in the hierarchy* spiking hurts (early
   features vs late semantics). It doubles as the Week-4 safety net, so it costs nothing not already
   budgeted.

3. **SSM-specific energy accounting.**
   Standard SNN energy models assume a MAC→accumulate conversion for conv and linear layers; an SSM scan has
   a different operation mix. Getting that accounting right — and stating the model explicitly — is a small
   methodological contribution, and loose energy accounting is the standard reviewer target
   (litreview §8, risk 5).

### The claim to defend

Not *"another SNN detector"*, but:

> what spiking **costs** a state-space detector, measured under controlled conditions nobody else has
> established, decomposed into the cost of *sparsity* and the cost of *binarisation*, with a design
> principle — spike the temporal axis only — that explains why this model avoids the timestep multiplier
> every other spiking detector pays.

That is a stronger thesis than a full-spike model that lands at 39.

## Next — Stage 18

Wire `SpikingSSMBlock` into a backbone as a Hydra-selectable variant. The one piece of real work:
`backbone/resnet_mamba.py`'s `_state_to_bmajor` / `_state_from_bmajor` assume per-layer
`(conv, ssm)` tuples, and the spiking state is `(mamba_state_list, mem)`. A spiking-aware pair of
helpers is needed — `mem` is already shaped `(N, C)` with dim0=N precisely so the same reshape
pattern applies. Do this in the **new** package (a `spikingssm` backbone module), not by editing
`resnet_mamba.py`.
