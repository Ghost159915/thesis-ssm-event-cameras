"""Record the official test evaluation's input, tagged by test recording (Stage 21, item 1, Part A).

`event_ssm.eval.bootstrap` (Part B) resamples test RECORDINGS, so it needs, for every element the RVT
`PropheseeEvaluator` scores, the recording it came from. RVT never tracks that, but its streaming test loader makes it
recoverable exactly:

  * each test recording is one whole stream, assigned to a fixed (dataloader worker, batch slot) pair
    (`ShardedStreamingDataPipe`); its first sample carries IS_FIRST_SAMPLE = True, padding samples carry False and
    no labels;
  * in `Module._val_test_step_impl`, the evaluator receives one element per (time step, batch slot with labels), in
    the order the step asks each time step's `SparselyBatchedObjectLabels.get_valid_labels_and_batch_indices()`.

So a counter per (worker, slot), bumped on IS_FIRST_SAMPLE, plus the slot list of each step names the recording of
every element. The labels and predictions saved are the evaluator's own buffer at `evaluate_buffer` time, i.e.
exactly what was scored; the evaluator's construction and evaluation arguments and the official metrics go into the
capture's meta, so `bootstrap.RvtScorer` rebuilds the same evaluator.

Nothing in `external/` is edited: methods are wrapped at run time, like `scripts/perclass_patch.py`. The original
method always runs and its result is returned unchanged, so the metrics cannot change. Any inconsistency (an element no
step accounted for, a slot that never saw a first sample) disables the capture with a loud message — no file is
written — and never breaks the evaluation. Only a TEST evaluation is written.

Switch: BOOTSTRAP_CAPTURE=/abs/path/capture.pkl (unset = nothing is imported or patched: byte-identical eval).
Hooked in scripts/stage7_eval.py (EventSSM, PureSSM, SpikingSSM launchers) and scripts/stage8_baseline_eval.py
(S5-RVT). Usage: docs/runbooks/Stage21_bootstrap_ghostmachine.md.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

from event_ssm.eval.bootstrap import save_capture

ENV = "BOOTSTRAP_CAPTURE"
_TAG = "[bootstrap-capture]"


class CaptureError(RuntimeError):
    """The recorded element order does not match what the evaluator received."""


class Recorder:
    """Recording id per (worker, batch slot), and the labelled slots of the step in progress."""

    def __init__(self, path):
        self.path = Path(path)
        self.error = None
        self._rec_of_slot = {}
        self._n_rec = 0
        self._step = None                # {"worker": id, "slots": [...]} while a val/test step runs

    def begin_step(self, worker_id, first_flags):
        for slot, first in enumerate(first_flags):
            if bool(first):
                self._rec_of_slot[(worker_id, slot)] = self._n_rec
                self._n_rec += 1
        self._step = {"worker": worker_id, "slots": []}

    def note_valid_indices(self, indices):
        if self._step is not None:
            self._step["slots"].extend(int(i) for i in indices)

    def end_step(self):
        self._step = None

    def take_seq_ids(self, n):
        """Recording ids of the n elements the current step is handing to the evaluator."""
        if self._step is None:
            raise CaptureError("the evaluator received labels outside a val/test step")
        slots = self._step["slots"]
        if len(slots) != n:
            raise CaptureError(f"the evaluator received {n} elements, the step gathered {len(slots)} labelled slots")
        ids = [self._rec_of_slot.get((self._step["worker"], s)) for s in slots]
        if any(i is None for i in ids):
            raise CaptureError("an element came from a batch slot that never saw a first sample")
        self._step["slots"] = []
        return ids

    def fail(self, msg):
        if self.error is None:
            self.error = msg
            print(f"{_TAG} DISABLED: {msg} -- the evaluation continues, no capture will be written", file=sys.stderr)


def _flags(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


def _bound_args(fn, self, args, kwargs):
    b = inspect.signature(fn).bind(self, *args, **kwargs)
    b.apply_defaults()
    d = dict(b.arguments)
    d.pop("self", None)
    return d


def _ckpt_from_argv(argv):
    for a in argv:
        if a.startswith("checkpoint="):
            return a.split("=", 1)[1].strip("'\"")
    return None


def _write(rec, ev, metrics, evaluate_kwargs):
    if metrics is None:
        return
    mode = getattr(ev, "_bootstrap_mode", None)
    if mode != "TEST":
        print(f"{_TAG} {mode} evaluation: not captured (test only)", file=sys.stderr)
        return
    if rec.error:
        print(f"{_TAG} NOT written: {rec.error}", file=sys.stderr)
        return
    labels, predictions = list(ev._buffer[ev.LABELS]), list(ev._buffer[ev.PREDICTIONS])
    seq_ids = list(ev._bootstrap_seq_ids)
    if not (len(seq_ids) == len(labels) == len(predictions)):
        print(f"{_TAG} NOT written: {len(seq_ids)} recording ids, {len(labels)} labels, {len(predictions)} "
              f"predictions", file=sys.stderr)
        return
    meta = {"evaluator_kwargs": dict(ev._bootstrap_init), "evaluate_kwargs": dict(evaluate_kwargs),
            "official_metrics": {k: float(v) for k, v in metrics.items()}, "mode": mode,
            "n_recordings": len(set(seq_ids)), "argv": list(sys.argv), "ckpt": _ckpt_from_argv(sys.argv)}
    tmp = rec.path.with_name(rec.path.name + ".tmp")
    save_capture(tmp, seq_ids, labels, predictions, meta)
    os.replace(tmp, rec.path)                                     # never a half-written capture
    print(f"{_TAG} wrote {rec.path}: {len(labels)} elements from {meta['n_recordings']} recordings "
          f"(AP {meta['official_metrics'].get('AP', float('nan')):.4f})", file=sys.stderr)


def install(rec, module_cls, labels_cls, evaluator_cls, first_key):
    """Wrap the step, the label lookup and the evaluator. Idempotent per class."""
    if evaluator_cls.__dict__.get("_bootstrap_capture_installed"):
        return

    step_orig = module_cls._val_test_step_impl

    def step(self, batch, mode):
        if rec.error is None:
            try:
                rec.begin_step(self.get_worker_id_from_batch(batch), _flags(self.get_data_from_batch(batch)[first_key]))
                ev = self.mode_2_psee_evaluator.get(mode)
                if ev is not None:
                    ev._bootstrap_mode = getattr(mode, "name", str(mode))
            except Exception as e:  # noqa: BLE001 -- the capture must never break the evaluation
                rec.fail(f"step bookkeeping failed ({type(e).__name__}: {e})")
        try:
            return step_orig(self, batch=batch, mode=mode)
        finally:
            rec.end_step()

    valid_orig = labels_cls.get_valid_labels_and_batch_indices

    def get_valid_labels_and_batch_indices(self):
        out = valid_orig(self)
        if rec.error is None:
            rec.note_valid_indices(out[1])
        return out

    init_orig, reset_orig = evaluator_cls.__init__, evaluator_cls._reset_buffer
    add_orig, eval_orig = evaluator_cls.add_labels, evaluator_cls.evaluate_buffer

    def __init__(self, *args, **kwargs):
        init_orig(self, *args, **kwargs)
        self._bootstrap_init = _bound_args(init_orig, self, args, kwargs)

    def _reset_buffer(self):
        reset_orig(self)
        self._bootstrap_seq_ids = []

    def add_labels(self, labels):
        add_orig(self, labels)                                    # the evaluator's own checks run first
        if rec.error is None:
            try:
                self._bootstrap_seq_ids.extend(rec.take_seq_ids(len(labels)))
            except Exception as e:  # noqa: BLE001
                rec.fail(str(e))

    def evaluate_buffer(self, *args, **kwargs):
        metrics = eval_orig(self, *args, **kwargs)
        try:
            _write(rec, self, metrics, _bound_args(eval_orig, self, args, kwargs))
        except Exception as e:  # noqa: BLE001
            print(f"{_TAG} NOT written ({type(e).__name__}: {e})", file=sys.stderr)
        return metrics

    module_cls._val_test_step_impl = step
    labels_cls.get_valid_labels_and_batch_indices = get_valid_labels_and_batch_indices
    evaluator_cls.__init__ = __init__
    evaluator_cls._reset_buffer = _reset_buffer
    evaluator_cls.add_labels = add_labels
    evaluator_cls.evaluate_buffer = evaluate_buffer
    evaluator_cls._bootstrap_capture_installed = True


def apply() -> bool:
    """Install the capture if BOOTSTRAP_CAPTURE is set (RVT must be importable). Returns whether it did."""
    path = os.environ.get(ENV)
    if not path:
        return False
    if not Path(path).is_absolute():
        raise ValueError(f"{ENV} must be an absolute path (the launchers cd into RVT/), got {path!r}")
    from data.genx_utils.labels import SparselyBatchedObjectLabels
    from data.utils.types import DataType
    from modules.detection import Module
    from utils.evaluation.prophesee.evaluator import PropheseeEvaluator

    install(Recorder(path), Module, SparselyBatchedObjectLabels, PropheseeEvaluator, DataType.IS_FIRST_SAMPLE)
    print(f"{_TAG} recording the test evaluation's input -> {path}", file=sys.stderr)
    return True
