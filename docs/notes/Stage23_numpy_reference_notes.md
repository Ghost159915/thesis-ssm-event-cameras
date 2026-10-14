# Stage 23 groundwork — NumPy reference of the spiking temporal block

**Written 2026-10-14.** Status: code + NumPy-only tests done on the Mac (30 pass). The LIF-vs-`LIFReadout` CPU test,
the real-kernel GPU test and the checkpoint run have **not been executed yet** (the Mac has no torch; GhostMachine's GPU
is on the graded 400k run). Run order in §4.

## 1. Why

A framework-independent, timestep-by-timestep implementation of one spiking temporal block (Mamba-2 recurrence + LIF
readout), loaded from the trained checkpoints. It serves the neuromorphic-deployment part of the thesis (Future Work /
appendix), not a new contribution:

* it is the streaming form a CPU or neuromorphic core executes (no chunked scan, no kernels, no framework);
* it makes the deployment obstacle measurable. The LIF line is a fixed-decay neuron (β ≈ 0.9 in every channel,
  Stage-22 β report). The Mamba decay `exp(Δt·A)` is recomputed from the input at every step, which a fixed-decay
  neuromorphic neuron cannot do. It is isolated in `ssd_step`, so a fixed-Δt variant can be measured against it;
* it is the oracle a NIR export (Stage 23 proper) would be checked against.

Only the temporal block is ported. Spatial BiMamba, PAFPN and YOLOX are dense ANN layers; spikes feed ≈ 1 % of the
model's MACs.

## 2. What was built (new files only; nothing existing edited)

| File | Role |
|---|---|
| `code/event_ssm/models/spikingssm_np/mamba2.py` | `Mamba2LayerNP` (one step: in_proj → causal conv → SSD → gated RMSNorm → out_proj), `MambaTemporalBlockNP`, `ssd_step` |
| `code/event_ssm/models/spikingssm_np/lif.py` | `LIFReadoutNP` (same β / θ maps, 3 output modes, 2 reset modes) |
| `code/event_ssm/models/spikingssm_np/block.py` | `SpikingSSMBlockNP.from_state_dict` (arm read from `_extra_state`), `state_to_numpy` / `state_to_torch` |
| `code/event_ssm/eval/np_parity.py` | the parity criterion (§3), torch-free |
| `code/event_ssm/scripts/stage23_np_parity.py` | trained checkpoint, real Gen1 features, resynced + free-running comparison |
| `code/event_ssm/tests/models/spikingssm_np/` | `test_np_core.py` (NumPy only), `test_np_lif_parity.py` (torch CPU), `test_np_block_parity_gpu.py` (gpu) |

Upstream semantics reproduced (checked against `mamba_ssm/modules/mamba2.py`, `Mamba2.step`): ngroups = 1,
d_ssm = d_inner, D per head, RMSNormGated eps 1e-5 with `norm_before_gate=False`, dt_limit (0, ∞), `step_scale`
applied after the softplus (as `_scan._scaled_dt`). Any other shape is refused at load time. The package never
imports torch (`test_package_imports_without_torch`); weights are read from torch tensors by duck typing.

## 3. The parity criterion (why not bit-equality, why not a flip fraction)

The threshold amplifies noise: TF32 inside the chunk scan perturbs y by ≈ 1e-3, which flips any spike whose membrane
sits that close to θ. A fraction-of-flips bound (the Stage-18 streaming criterion) would also pass a real bug that
flips only a few spikes. The criterion is therefore exact: both runs start a row from the same membrane, so before the
row's first flip

  |m^a_t − m^b_t| ≤ Σ_k β^k |Δy_{t−k}| ≤ max|Δy| / (1 − β) =: δ,

and a flip requires |m^b_t − θ| ≤ δ. Every first flip must satisfy this, and graded/analog values must agree to δ
before it; afterwards rows legitimately diverge and are only counted.

It has teeth: a readout with a 5 % lower threshold is rejected (unit test). On the script smoke it gave
263 unexplained flips at a flip fraction of only 0.7 %, which a "< 1 % flips" rule would have passed.

## 4. How to run (GhostMachine)

CPU, safe during training (`conda activate events_signals && cd code`):

```bash
CUDA_VISIBLE_DEVICES="" pytest event_ssm/tests/models/spikingssm_np -q
```
Expect 30 NumPy tests + 13 LIF parity tests to pass, the 15 gpu tests deselected. The LIF test demands **bit-identical**
output once the module's own β is substituted (same IEEE ops in the same order); if it fails only there, look at
operation order before anything else.

Idle GPU (minutes):

```bash
pytest event_ssm/tests/models/spikingssm_np -m gpu -q
python event_ssm/scripts/stage23_np_parity.py --ckpt <spike 100k ckpt>
python event_ssm/scripts/stage23_np_parity.py --ckpt <graded 100k ckpt>
```
The script writes `results/stage23/np_parity/<tag>_<rec>_fp32.json` and exits 1 on a parity failure. Default 200
frames (10 s); stage 2 holds a ≈ 170 MB float64 SSM state per copy, so expect a few minutes of CPU per run.
`--precision bf16` reports the launchers' precision for context; fp32 is the verdict.

Mac (NumPy only, the repo `conftest.py` imports torch):

```bash
cd code && PYTHONPATH=$PWD python3 -m pytest --noconftest -c event_ssm/pytest.ini event_ssm/tests/models/spikingssm_np -q
```

## 5. Verification done so far

* SSD step vs the independent dual (masked-attention) form, nonzero initial state: agree to 1e-10.
* Whole layer stepped vs a sequence-level re-derivation (full-clip conv + dual form): 1.8e-15 max error in float64.
  Planted bugs (reversed conv taps, norm before gate, step_scale before the softplus) each give O(1) errors.
* Streaming: any split of a clip (incl. one frame at a time) is bit-identical to the full clip, for all three modes.
* LIF hand-computed traces (β = 0.5, θ = 1, both reset modes), equality fires, β/θ maps, loader refusals.
* Checkpoint script: control flow smoke-tested with stubbed torch (PASS on a clean block, FAIL + exit 1 on a planted
  threshold bug).

## 6. Next (if the calendar allows; after the Stage-21/22 queue)

1. Fixed-Δt measurement: replace `dt` with a per-head constant (e.g. its mean over the stream) and report spike flips /
   y error with the same script — the quantitative version of "Mamba's selective Δt is the Loihi wall".
2. Fixed-point variant of the LIF (integer membrane and decay) — does the dynamics/sparsity/binarisation
   decomposition survive chip arithmetic?
