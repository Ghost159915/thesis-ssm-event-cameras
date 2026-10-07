# Stage 22 — Synaptic-Operation and Energy Accounting for SpikingSSM: Design

**Thesis C · MMAN4953 · UNSW Sydney · Benas Vaiciulis** · 2026-10-07 · status: **draft for review**
**Decision behind it:** option 1 of the 2026-10-07 discussion (exact bound + measured firing rates), chosen by the
user over a hook-based nonzero counter (option 2) and prose only (option 3).
**Context:** `docs/notes/Stage21_22_tooling_notes.md` D5; thesis §5.7 (the readout-ablation table: mAP, firing rate,
SOPs, energy/frame), §6.2.

## 1. Purpose and the question it answers

The readout ablation table of §5.7 reports, per arm, synaptic operations (SOPs) and an estimated energy per frame.
The SNN detection literature estimates these by counting spike-driven operations and pricing them at a
per-operation energy~[Horowitz 2014; EMS-YOLO; SpikeYOLO]. This design produces those two numbers for SpikingSSM and,
first, the **ceiling** on what spiking can change in this hybrid architecture.

That ceiling is the main result. Only the temporal readout spikes (choice C), and its output feeds only the first
layers of the neck. A first-principles count from the layer shapes:

| spiking stage | output (C × h × w) | consuming layer(s) in YOLO-PAFPN | spike-fed MACs |
|---|---|---|---|
| 4 | 512 × 8 × 10 | `lateral_conv0`: 1×1, 512 → 256 | 80 × 512 × 256 = **10,485,760** |
| 3 | 256 × 16 × 20 | `C3_p4.conv1`, `C3_p4.conv2`: 1×1, 512 → 128 each; stage 3 is channels 256–511 of their input (`cat([up(f_out0), x1])`) | 320 × 256 × (128 + 128) = **20,971,520** |
| 2 | 128 × 32 × 40 | `C3_p3.conv1`, `C3_p3.conv2`: 1×1, 256 → 64 each; stage 2 is channels 128–255 (`cat([up(f_out1), x2])`) | 1280 × 128 × (64 + 64) = **20,971,520** |
| **total** | | | **52,428,800 MACs ≈ 0.105 GFLOPs** |

Against PureSSM's 10.12 GFLOPs per frame (Stage 16), spike-driven arithmetic is **≈ 1.0 %** of the network.
Even a model that never fired would change the operation count by at most this much.

## 2. Definitions

Let stage $s \in \mathcal{S}$ (the spiking stages of the arm) have readout output $y_s \in \mathbb{R}^{C_s \times
h_s \times w_s}$ per frame. Let $M_s$ be its **spike-fed MACs**: the multiply–accumulates of the neck convolutions
that read $y_s$, counting only the input channels that come from $y_s$ (§1 table). Let $\rho_s$ be the **nonzero
rate** of $y_s$: the fraction of its entries that are nonzero, averaged over frames.

- **Operations by readout.**
  - **spike:** $y_s \in \{0,1\}$. Each nonzero input triggers accumulates (weight additions) along its fan-out, so
    $\mathrm{SOP}_s = \rho_s M_s$ accumulates (AC) and no multiplies.
  - **graded:** $y_s = s_t\, m_t$ carries a value. Each nonzero input triggers multiply–accumulates, so the cost is
    $\rho_s M_s$ *sparse* MACs. Counting these as accumulates would overstate the saving.
  - **analog:** $y_s$ is dense ($\rho_s \approx 1$), so the cost is $M_s$ MACs, as in PureSSM.
- **Rest of the network.** $\mathrm{MAC}_{\text{rest}} = \tfrac12\,\mathrm{FLOPs}_{\text{total}} - \sum_{s \in \mathcal S} M_s$,
  with $\mathrm{FLOPs}_{\text{total}}$ from the Stage-22 benchmark (profiler-counted plus analytic Mamba FLOPs, the
  existing harness). This part is dense and unaffected by spiking.
- **Energy model.** $E = E_{\text{MAC}}\,(\mathrm{MAC}_{\text{rest}} + \mathrm{MAC}_{\text{readout}}) + E_{\text{AC}}\,
  \mathrm{AC}_{\text{readout}}$, with $E_{\text{MAC}} = 4.6$ pJ and $E_{\text{AC}} = 0.9$ pJ (32-bit floating point,
  45 nm~[Horowitz 2014]). These are the constants of the SNN detection literature, used for comparability.
  The reference is PureSSM with every operation priced as a MAC.
- **Reported per arm:** $M_s$ and $\rho_s$ per stage; $\sum$ SOPs (spike) or sparse MACs (graded); $E$ per frame;
  the relative saving against PureSSM $1 - E/E_{\text{PureSSM}}$; and the **ceiling**, the same saving at $\rho = 0$.

## 3. Components

1. **`code/event_ssm/benchmark/sop.py`**, pure functions that are CPU-testable:
   - `spike_fed_macs(detector, spiking_stages) -> {stage: M_s}`. It is derived from the model's own modules: input
     and output channels and kernel sizes of `fpn.lateral_conv0`, `fpn.C3_p4.conv{1,2}`, `fpn.C3_p3.conv{1,2}`, and
     feature-map sizes from the strides at the fixed 256×320 input. The wiring (which conv reads which stage, and
     which channel slice) is the only hand-written part, and a test checks it against the module shapes.
   - `op_energy(macs_rest, readout_mode, macs_by_stage, rates) -> dict` (MACs, ACs, energy in J, saving, ceiling).
2. **Nonzero-rate measurement**, `measure_nonzero_rates(model, frames) -> {stage: rho_s}`. Forward hooks on the
   spiking temporal blocks record the fraction of nonzero entries of their outputs, in eval mode, with the recurrent
   state carried across the frames of a fixed clip. It needs the CUDA kernels, so the GPU part is run by the user.
   - **Data:** the benchmark's fixed 64-frame clip, so the rates match the frames whose FLOPs and latency are
     measured. As a cross-check, the last `[spk-monitor]` rates are taken from the run's training console log (the
     benchmark itself refuses to run with monitors on). They are averaged differently: training forwards, and the LIF
     spike rate rather than readout nonzeros.
3. **Integration into the benchmark (`stage10_benchmark.py`).** For `spikingssm`, the output JSON gains a `sop`
   block: per-stage $M_s$ and $\rho_s$, the totals, energy, saving and ceiling, plus the constants and their source.
   `stage10_report.py` adds SOP and estimated-energy columns. The measured GPU J/frame stays as it is: it is a
   measurement, the SOP energy is an estimate, and the report labels them differently.

## 4. Error handling and validity

- Fail closed if the neck's channel structure differs from the wiring assumed in §1. For example, if a future config
  changed `in_stages` or the PAFPN, the shape assertions in `spike_fed_macs` raise.
- A rate outside [0, 1], or a missing spiking stage in the measurement, raises.
- The estimate is labelled everywhere as an **operation-count estimate**: it ignores memory access and data movement,
  which dominate energy on real hardware, and sparse-MAC savings need hardware that skips zeros. The thesis states
  this next to the numbers (§2.13 already describes the convention and its limits).

## 5. Testing

- **CPU:**
  - `spike_fed_macs` on a CPU-built model returns exactly 10,485,760 / 20,971,520 / 20,971,520 for stages 4 / 3 / 2
    (the hand-derived §1 values).
  - The wiring test checks the PAFPN input channel counts against the backbone stage dims.
  - `op_energy` is tested against hand-computed literals for each readout mode, including the ρ = 0 ceiling and
    the graded-as-MAC rule.
  - Rate validation is tested on synthetic inputs.
- **GPU** (`-m gpu`, idle GPU only): the rate measurement on a real checkpoint returns one rate per spiking stage
  in [0, 1], consistent in magnitude with the training monitor.

## 6. Out of scope (recorded)

- Counting actual nonzeros inside the neck with hooks (option 2). The ceiling holds regardless, so it cannot change
  the conclusion.
- Any accounting of the state-space recurrence as neuromorphic operations (how S4D runs on Loihi 2, Meyer et al.).
  That would be an estimate of a deployment this thesis does not perform; it is Future Work (§7.2).
- Energy measured on neuromorphic hardware.

## 7. Thesis consequences

§5.7 reports the ceiling first and then the per-arm estimates. §6.2 and the Limitations bullet state that, in this
hybrid, the energy argument for spiking is not an operation-count saving: the readout is a step toward a neuromorphic
temporal core and the instrument for measuring the cost of spiking, not a source of savings by itself.
