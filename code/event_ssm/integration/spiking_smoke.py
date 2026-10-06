"""Stage-19 SpikingSSM overfit smoke: verdict, per-step recorder and figure.

The smoke itself (`proofs/smoke_overfit_spikingssm.py`) needs the GPU. Everything it *decides* lives
here so it is CPU-tested (`tests/models/spikingssm/test_spiking_smoke.py`): a GPU run can then fail
only on the model, never on the bookkeeping. Gates, fixed before the run:

  1. every loss is finite;
  2. loss reduction (mean of the first 3 steps / mean of the last 3) >= 3x on one fixed real Gen1
     batch: the Stage-5/12 overfit gate, unchanged so the three smokes are comparable;
  3. `spike` / `graded` only: the final firing rate of every spiking stage (mean of the last 3
     steps) lies in [0.01, 0.90], the band of `attach_spiking_monitor`. Outside it the arm has hit
     one of the two silent SNN failure modes (no spikes => no surrogate gradient; all spikes => the
     binary output carries nothing). `analog` emits the pre-reset membrane, not the spike, so its
     rate is reported but not gated. A NaN rate fails a gated arm (the monitor lets NaN through,
     Stage-18 notes §5; the smoke must not).
"""
import math
import statistics
import time

import torch
from lightning.pytorch.callbacks import Callback

_MODES = ("spike", "graded", "analog")
WINDOW = 3          # steps averaged for the initial/final loss and the final firing rate
MIN_POINTS = 5      # fewer loss points than this and the reduction is not a measurement
WARMUP_STEPS = 5    # excluded from the step-time median (cuDNN autotune, allocator growth)

# Categorical colour per spiking STAGE (the entity), never per plot order, so a stage keeps its
# colour across arms and ladder rungs. Slots 1-3 of the validated default palette (dataviz skill;
# light-mode CVD and normal-vision checks pass).
_STAGE_COLOUR = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def smoke_verdict(losses, rates, mode, *, expected_stages=None, min_reduction=3.0, silence=0.01,
                  saturation=0.90):
    """Apply the three gates above.

    losses: per-step total loss. rates: {stage: per-step firing rate}. expected_stages: the requested
    spiking stages; if given, the recorded stages must be exactly these (any arm), so a recording
    that silently covers other stages cannot pass. Returns a dict (the Stage-19 notes row; pass it
    through `json_safe` before dumping) with `passed` and the human-readable `failures`.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown output mode {mode!r}; expected one of {_MODES}")
    losses = [float(x) for x in losses]
    failures = []
    if len(losses) < MIN_POINTS:
        failures.append(f"too few loss points ({len(losses)} < {MIN_POINTS})")
    if not all(math.isfinite(x) for x in losses):
        failures.append("non-finite loss encountered")
    initial, final = _mean(losses[:WINDOW]), _mean(losses[-WINDOW:])
    reduction = initial / max(final, 1e-9)
    if not reduction >= min_reduction:                   # written so that a NaN reduction fails
        failures.append(f"loss reduction {reduction:.2f}x < {min_reduction:g}x (no-learning regression)")

    if expected_stages is not None and set(rates) != set(expected_stages):
        failures.append(f"recorded stages {sorted(rates)} != requested stages "
                        f"{sorted(expected_stages)}")

    gated = mode != "analog"
    final_rate, firing = {}, {}
    for stage in sorted(rates):
        r = _mean([float(x) for x in rates[stage][-WINDOW:]])
        if not math.isfinite(r):
            state = "NAN"
        elif r < silence:
            state = "SILENT"
        elif r > saturation:
            state = "SATURATED"
        else:
            state = "OK"
        final_rate[stage], firing[stage] = r, state
        if gated and state != "OK":
            failures.append(f"stage {stage} {state}: final firing rate {r:.4f} "
                            f"outside [{silence:g}, {saturation:g}]")
    if gated and not rates:
        failures.append(f"no firing-rate record (a {mode} arm must report one)")
    return dict(mode=mode, initial_loss=initial, final_loss=final, reduction=reduction,
                final_rate=final_rate, firing=firing, firing_gated=gated,
                failures=failures, passed=not failures)


def median_step_ms(step_s, warmup=WARMUP_STEPS):
    """Median training-step time in ms, skipping the first `warmup` steps when the run is long
    enough to have a post-warm-up sample (otherwise all samples are used)."""
    if not step_s:
        return float("nan")
    return 1000.0 * statistics.median(step_s[warmup:] or step_s)


def output_stem(out_dir, tag, epochs, default_epochs=150):
    """File stem for one smoke run: `spikingssm_<tag>_overfit`, plus `_e<epochs>` off the default
    budget, plus `_run2`, `_run3`, ... when an earlier run already wrote it, so a rerun (e.g. to
    measure run-to-run spread) never overwrites earlier evidence."""
    base = f"spikingssm_{tag}_overfit" + ("" if epochs == default_epochs else f"_e{epochs}")
    stem, n = base, 1
    while (out_dir / f"{stem}.json").exists():
        n += 1
        stem = f"{base}_run{n}"
    return stem


def verify_arm(backbone, mode, stages):
    """Fail closed on a mislabelled arm: the BUILT model, not the config, decides what a run measured.
    Raises (never `assert`, which `python -O` strips) unless the backbone spikes on exactly `stages`,
    every one of them with `output_mode == mode`."""
    built = tuple(backbone.spiking_stages)
    if sorted(built) != sorted(stages):
        raise RuntimeError(f"mislabelled arm: built spiking_stages {list(built)} != requested "
                           f"{sorted(stages)}")
    for s in stages:
        got = backbone.temporal[str(s)].lif.output_mode
        if got != mode:
            raise RuntimeError(f"mislabelled arm: stage {s} built as {got!r}, requested {mode!r}")


def json_safe(obj):
    """Recursively map non-finite floats (NaN/inf) to None so the summary is strict JSON (`jq`-safe);
    a failing run is exactly when NaN appears."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def find_spiking_backbone(module):
    """The first submodule exposing `spiking_stats()` (SpikingSSMBackbone), wherever RVT nests it."""
    for m in module.modules():
        if callable(getattr(m, "spiking_stats", None)):
            return m
    raise ValueError("no spiking backbone (a module with spiking_stats()) in the model; "
                     "was it built with +experiment/gen1=spikingssm?")


def _sync():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


class SpikingSmokeRecorder(Callback):
    """Per training step: total loss, step time and, per spiking stage, firing rate and learned beta
    (mean / max).

    Step time is synchronised and spans Lightning's on_train_batch_start -> on_train_batch_end:
    forward, backward, gradient clipping and the optimiser step, plus RVT's host-side
    post-processing in training_step and, every `--monitor-every` steps, the monitors' host syncs
    and prints (the median absorbs those). Data loading and the LR-scheduler step are outside it.

    Reading `spiking_stats()` costs a host sync every step: fine for a smoke, which is why the
    production monitor (`attach_spiking_monitor`) reads it only at its cadence. The read happens
    after the step timer stops, so it does not inflate the step time.
    """

    def __init__(self):
        self.losses, self.step_s = [], []
        self.rates, self.beta_mean, self.beta_max = {}, {}, {}
        self._t0 = None
        self._backbone = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        _sync()
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not (isinstance(outputs, dict) and "loss" in outputs):
            return
        _sync()
        dt = time.perf_counter() - self._t0
        self.losses.append(float(outputs["loss"].detach()))
        self.step_s.append(dt)
        if self._backbone is None:
            self._backbone = find_spiking_backbone(pl_module)
        for stage, st in self._backbone.spiking_stats().items():
            self.rates.setdefault(stage, []).append(float(st["rate"]))
            self.beta_mean.setdefault(stage, []).append(float(st["beta_mean"]))
            self.beta_max.setdefault(stage, []).append(float(st["beta_max"]))


def plot_smoke(rec, verdict, mode, stages, path, *, silence=0.01, saturation=0.90,
               beta_cap=None, footer=""):
    """Three stacked panels on one step axis: loss (log), firing rate per stage (band marked),
    learned beta per stage (mean solid, max dashed). One y-scale per panel, by design."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, (ax_l, ax_r, ax_b) = plt.subplots(3, 1, figsize=(7.5, 8.2), sharex=True,
                                           gridspec_kw=dict(height_ratios=[1.2, 1, 1]))
    steps = range(len(rec.losses))
    tag = "PASS" if verdict["passed"] else "FAIL"
    stage_str = ",".join(str(s) for s in stages)

    ax_l.plot(steps, rec.losses, color=_INK, lw=1.5)
    ax_l.set_yscale("log")
    plain = FuncFormatter(lambda y, _: f"{y:g}")           # 6, not 6x10^0
    ax_l.yaxis.set_major_formatter(plain)
    ax_l.yaxis.set_minor_formatter(plain)
    ax_l.set_ylabel("total loss (log)")
    ax_l.set_title(f"Stage 19 SpikingSSM overfit: {mode}, spiking_stages=[{stage_str}]: {tag}\n"
                   f"loss {verdict['initial_loss']:.2f} → {verdict['final_loss']:.2f} "
                   f"({verdict['reduction']:.1f}×, gate ≥ 3×)",
                   fontsize=10, color=_INK)

    for s in sorted(rec.rates):
        ax_r.plot(range(len(rec.rates[s])), rec.rates[s], color=_STAGE_COLOUR.get(s, _MUTED),
                  lw=2, label=f"stage {s} ({verdict['firing'].get(s, '?')})")
    for y, name in ((silence, f"SILENT < {silence:g}"), (saturation, f"SATURATED > {saturation:g}")):
        ax_r.axhline(y, color=_MUTED, lw=1, ls="--")
        ax_r.annotate(name, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(4, 3),
                      textcoords="offset points", fontsize=8, color=_MUTED)
    ax_r.set_yscale("symlog", linthresh=1e-3)            # a rate of exactly 0 stays on the plot
    ax_r.set_ylim(0, 1)
    gate_note = "gated" if verdict["firing_gated"] else "reported, not gated (analog)"
    ax_r.set_ylabel(f"firing rate\n({gate_note})")
    if rec.rates:
        ax_r.legend(fontsize=8, frameon=False, loc="lower right")   # below 1e-3: rarely data

    for s in sorted(rec.beta_mean):
        c = _STAGE_COLOUR.get(s, _MUTED)
        ax_b.plot(range(len(rec.beta_mean[s])), rec.beta_mean[s], color=c, lw=2,
                  label=f"stage {s} β mean")
        ax_b.plot(range(len(rec.beta_max[s])), rec.beta_max[s], color=c, lw=1.5, ls="--",
                  label=f"stage {s} β max")
    if beta_cap is not None:
        ax_b.axhline(beta_cap, color=_MUTED, lw=1, ls=":")
        ax_b.annotate(f"β cap {beta_cap:g} (integrator limit)", xy=(0, beta_cap),
                      xycoords=("axes fraction", "data"), xytext=(4, -10),
                      textcoords="offset points", fontsize=8, color=_MUTED)
    ax_b.set_ylabel("learned β (leak)")
    ax_b.set_xlabel("train step (1 per epoch, same batch)")
    if rec.beta_mean:
        ax_b.legend(fontsize=8, frameon=False, loc="best", ncol=2)   # beta may sit anywhere

    for ax in (ax_l, ax_r, ax_b):
        ax.grid(True, color=_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(which="both", colors=_MUTED, labelsize=8)   # minor log labels too
    if footer:
        fig.text(0.01, 0.005, footer, fontsize=8, color=_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02 if footer else 0, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
