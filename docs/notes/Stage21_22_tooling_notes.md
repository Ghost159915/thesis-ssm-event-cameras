# Stages 21–22 — SpikingSSM Evaluation and Benchmark Tooling: Decision Record

**Thesis C · MMAN4953 · UNSW Sydney · Benas Vaiciulis** · 2026-10-07 · **Branch:** `stage19-smoke`
Built while the Stage-19 `[2,3,4]` checks trained, so that Stages 21–22 can run as soon as the Stage-20 checkpoints
exist. **Commits:** `8e78ac7`, `2fff812`, `e9e0ced` and the benchmark-fix commit after it.

## 1. What was built

| Component | Location | Role |
|---|---|---|
| Arm reader | `code/event_ssm/integration/spiking_ckpt.py` | Reads a SpikingSSM checkpoint's ablation arm from its `_extra_state` (`temporal.<s>` → residual; `temporal.<s>.lif` → output mode, reset, alpha, detach_reset, learn flags, and a non-learned beta/threshold). Refuses non-spiking, pre-D14, malformed or per-stage-inconsistent checkpoints. Emits Hydra overrides, the run tag, the full `stage7_eval.py` argv, and a verdict on extra args. CLI: `tag \| overrides \| argv \| check-extra`. |
| Test-set eval | `code/event_ssm/scripts/stage21_spikingssm_test_eval_local.sh` | Sibling of the PureSSM test eval (unchanged), using the same recipe. The arm comes from the checkpoint, so a checkpoint cannot be evaluated as another arm. Log: `results/stage21_test_eval/<tag>[_posthoc]/test_eval_<run>_step<N>_<ts>.txt`. |
| Benchmark entry | `benchmark/bench_models.py`, `scripts/stage10_benchmark.py`, `scripts/stage10_report.py`, `scripts/stage10_run_local.sh` | `--models spikingssm \| all+spikingssm --spikingssm-ckpt PATH [--spikingssm-test-ap AP]`. The arm comes from the checkpoint and is recorded per model. Output defaults to `results/stage22/<tag>[_with_ann]/`. |

Usage, once a Stage-20 checkpoint exists and the GPU is idle (the user runs these):
```bash
bash code/event_ssm/scripts/stage21_spikingssm_test_eval_local.sh /abs/.../epoch=...-val_AP=....ckpt
bash code/event_ssm/scripts/stage10_run_local.sh --models all+spikingssm --spikingssm-ckpt /abs/...ckpt --spikingssm-test-ap 0.4xx
```

## 2. Decisions

- **D1. The arm is read from the checkpoint, never typed.** The checkpoint arm contract (Stage-18 D14) would reject
  a mismatch at load anyway. Deriving the overrides from the checkpoint removes the possibility of the mistake rather
  than only detecting it. The overrides were verified on the three real Stage-19 checkpoints
  (`graded_s4`, `spike_s234`, `spike_s4`); the PureSSM checkpoint is refused.
- **D2. Fixed beta/threshold compare within tolerance.** A fixed beta is stored through its logit, so 0.89999998
  comes back as 0.90000004. The arm contract's 1e-5 relative tolerance (`lif.py` `_FIXED_REL_TOL`) absorbs this.
  The round-trip test (overrides → compose → LIF arm record) checks those two values within that tolerance and
  everything else exactly.
- **D3. Code review ("with fixes") — items taken.**
  - **Important 1:** a spikingssm benchmark would have overwritten the citable `results/stage10/bench_results.json`
    (gitignored, unrecoverable). Spiking runs now default to `results/stage22/`, and any results JSON holding a
    different model set is never overwritten.
  - **Important 2:** the eval launcher accepted any extra argument. In particular, an override of a *learned*
    threshold or beta is silently ignored by the model, so a "sweep" would have measured one operating point under
    the canonical label. Extra args are now allow-listed; learned knobs and recipe keys are refused; post-hoc sweeps
    of non-learned knobs need `SPIKING_ALLOW_ARM_OVERRIDE=1` and are filed under `<tag>_posthoc/`.
  - **Minor items taken:**
    - the arm and its tag are recorded per model and shown in report labels (a residual arm must be visible);
    - the test-AP units are checked;
    - the checkpoint path is stored absolute;
    - stray spiking flags are errors;
    - a flops note says SOPs are not accounted;
    - the benchmark refuses to run with training monitors on, and the run script unsets them;
    - one-line errors replace tracebacks, with a clear message for pre-D14 checkpoints;
    - the checkpoint step appears in the log name;
    - tests were added for fixed beta, residual disagreement, the round trip, the allow-list, and the Stage-9 hook.
  - **Not taken:** the pre-existing `load_weights` dropping of unexpected keys. For spiking checkpoints every arm
    difference still surfaces as a missing key or an extra-state mismatch.
- **D4. Graph replay is not measured for SpikingSSM.** Capturing the LIF time loop as a CUDA graph has not been
  validated (Stage-18 deferred item). The JSON records an honest skip with this reason.
- **D5. SOP / energy accounting (Stage 22, 2c; user chose option 1).** Spec
  `docs/specs/2026-10-07-stage22-sop-energy-design.md` (+ Revision 2), plan `docs/plans/2026-10-07-stage22-sop-energy-plan.md`,
  code `benchmark/sop.py`. The benchmark's `spikingssm` entry gains a `sop` block and the report a `sop_table.md`:
  - **Spike-fed MACs** are derived from the neck's own modules: 10,485,760 (stage 4) + 20,971,520 (stage 3) +
    20,971,520 (stage 2) = 52,428,800 per frame. A test traces the real detector and confirms that these five 1×1
    convolutions are the only arithmetic reading the stage outputs.
  - **Dense MACs** = profiler MAC ops ÷ 2 + the unprofiled SSM kernels (temporal 113,254,400 + spatial 361,799,680
    MACs per frame).
  - **Operation class:** a spike readout without residual is priced as accumulates; everything else as sparse MACs
    (ρ·M). A non-binary "spike" output is refused.
  - **Rates** come from 16 test sequences (32 warm-up + 64 counted frames each), with spread and drift. Each LIF neuron
    costs one MAC per frame.
  - **Ceiling** (the saving at zero activity) ≈ (52.4 M − neurons) / ≈ 5.0 G MACs ≈ 1 % for `[2,3,4]`, ≈ 0.2 % for
    `[4]`.
  - **Conclusion:** the energy case for choice C is not an operation-count saving.
  - **Training-monitor cross-check:** done by hand. The last `[spk-monitor]` rates of the run's console log are quoted
    next to the measured rates.
  - **Rulings during implementation:** the tracer test kept tainted tensors alive (an `id()`-reuse false positive); the
    SOP step sits before `del model`; the train-mode VRAM forward cannot alter spike outputs (no BatchNorm in the
    SpikingSSM backbone).
- **D6. Finding: the stored Stage-10/16 FLOP totals were slightly wrong — FIXED 2026-10-07 (user: "fix the FLOP
  numbers and update chapter 5"; resolution at the end of this item).** `flops.total_gflops` = torch.profiler + an analytic add-on.
  - The analytic Mamba-2 formula (`bench_metrics.mamba2_layer_macs_per_token`) **includes the in/out projections**,
    which are `nn.Linear` layers the profiler already counts. That is 0.832 of the 1.061 analytic GFLOPs for EventSSM
    and PureSSM.
  - The S5 formula (`s5_block_macs_per_token`) double-counts two things, both confirmed by a CPU profile that
    reproduces the stored GPU `counted_gflops` exactly (10.0255):
    - the feed-forward layers: 0.503 GFLOPs (`aten::mm` = 0.505 GFLOPs);
    - the complex B̄u / C̃x products of RVT's vmapped `apply_ssm`: 0.3355 GFLOPs. The profiler counts them as
      `aten::bmm` of shape `[N,P,P]@[N,P,1]`, four stages × 0.0839 G, at real-equivalent FLOPs; the analytic formula
      adds them again at 4 real MACs per complex MAC. MaxViT attention accounts for the other 0.393 G of `aten::bmm`.
    (Corrected after the final code review of 2026-10-07: the first version of this note subtracted only the
    feed-forward and gave 11.39.)
  - PureSSM's **spatial BiMamba conv + scan kernels are counted nowhere**: ≈ 0.724 GFLOPs.
  - **Corrected totals:** EventSSM 13.546 − 0.832 = ≈ 12.71; S5-RVT 11.894 − 0.503 − 0.336 = ≈ 11.06; PureSSM
    10.121 − 0.832 + 0.724 = ≈ 10.01 GFLOPs. mAP/GFLOP becomes 3.64 / 4.32 / 4.64 (was 3.41 / 4.01 / 4.59).
  - **The ranking and every qualitative claim are unchanged:** PureSSM is the lightest and has the best mAP/GFLOP;
    EventSSM does more arithmetic than the baseline yet runs faster.
  - **Fix:**
    - make the Mamba analytic add-on kernel-only and add the spatial term (`benchmark/sop.py` already computes both);
    - make the S5 add-on exclude what the profiler already counts: drop the feed-forward term, and either drop the
      profiled complex `bmm` or count B̄u / C̃x only once, at 4 real MACs.
    The latency/energy JSON values are unaffected.
  - **Resolution (2026-10-07).**
    - **Code:** the add-on is now `bench_metrics.analytic_unprofiled_gflops`: Mamba-2 and BiMamba kernel work only
      (conv + scan + gated norm), and S5 with the feed-forward excluded and the complex products topped up from 1 to 4
      real MACs. `BenchModel.spatial_hparams()` lists the BiMamba blocks. The JSON records `analytic_convention`.
    - **Tests:** the premise "S5 complex products are profiled at one real MAC" is checked on the real baseline. The
      FLOP add-on is checked to equal the SOP kernel count plus the norm terms.
    - **Recount:** `scripts/stage10_flops_recount.py` recounted the citable `results/stage10/bench_results.json` on the
      CPU. Only `flops` changed (verified: every other field is identical, `counted_gflops` untouched). The original is
      at `bench_results.v1.json`, and each model keeps `flops.v1`.
    - **Final values:** EventSSM 12.714, S5-RVT 11.056, PureSSM 10.026 GFLOPs. PureSSM is 0.014 above the ≈ 10.01
      estimated above because its spatial blocks now also get the gated-norm term the temporal blocks always had.
      mAP/GFLOP with the Table-accuracy APs: 3.64 / 4.32 / 4.63.
    - **Thesis:** abstract, the §4 FLOP definition, `tab:efficiency`, §5 prose and the Ch. 7 summary are updated.
      `docs/results/Stage16_results.md` and `Thesis_Progress_Writeup.md` carry the corrected values with a note.

## 3. Tests

Stage-22 SOP work (2026-10-07): 89 CPU tests pass across `test_bench_{sop,metrics,schema,report,models,clip}.py`;
`test_bench_sop.py::test_sop_on_a_real_spike_checkpoint` is gpu-marked (idle GPU).

**Final code review of the SOP work (2026-10-07, independent reviewer):** the reviewer re-derived every count from the
code and confirmed it (spike-fed 52,428,800; kernels 113,254,400 + 361,799,680; dense total 5,001,574,400
MACs/frame; ceiling 1.04 % for `[2,3,4]`, 0.209 % for `[4]`, and 1.16 % even with no kernel term). No critical
findings. One important finding: the D6 S5-RVT total, fixed above. **Deferred minors (not fixed; user's call):**
1. Rates are measured under bf16 autocast only implicitly (the VRAM section leaves it on). Set it explicitly and
   record it.
2. `by_op` is captured only on the profiler-fallback path. Profile unconditionally for spikingssm.
3. Error text containing `|` or a newline breaks a `sop_table.md` row. With error rows only, the footer reads
   "Constants: .".
4. `{"error": str(e)}` drops the exception type.
5. The profiler-convention test lacks the model's own shapes (a 3-D bias-free linear → `mm`, a depthwise conv).
   The reviewer verified both by hand.
6. The rate hook assumes a tuple output; assert it.


187 CPU tests pass across the files touched on 2026-10-07: `test_spiking_ckpt.py`, `test_stage21_eval_script.py`,
`test_bench_{models,schema,report,metrics,clip}.py`, `test_resume_guard.py`, and the Stage-19/20 launcher tests.
GPU-dependent paths (real evaluation, benchmark timing) run only when the GPU is idle.
