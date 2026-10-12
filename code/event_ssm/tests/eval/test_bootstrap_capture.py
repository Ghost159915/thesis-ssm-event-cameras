"""Part A of the Stage-21 bootstrap: recording the evaluator's input, tagged by test recording (CPU).

The RVT classes are stood in for by minimal fakes that keep the parts the capture relies on: a val/test step that
asks each time step's sparse labels for (labels, batch slots) and hands the gathered elements to the evaluator in that
order, and an evaluator buffer. Every test installs the patches on FRESH classes, so nothing leaks between tests. One
test uses the real RVT PropheseeEvaluator (a subclass, so the real class stays untouched) and checks that re-scoring
the capture with the bootstrap's RvtScorer reproduces the official metrics exactly — the identity check in miniature.
"""
import enum

import numpy as np
import pytest

from event_ssm.eval import bootstrap, bootstrap_capture as bc

FIRST = "is_first"                       # stands in for DataType.IS_FIRST_SAMPLE


class Mode(enum.Enum):                   # as RVT's modules.utils.detection.Mode
    TRAIN = 1
    VAL = 2
    TEST = 3


def make_classes(evaluator_base=None):
    """Fresh fake RVT classes (install() patches classes in place)."""

    class Labels:                        # SparselyBatchedObjectLabels: one per time step, one entry per batch slot
        def __init__(self, per_slot):
            self.per_slot = per_slot

        def get_valid_labels_and_batch_indices(self):
            idx = [i for i, x in enumerate(self.per_slot) if x is not None]
            return [self.per_slot[i] for i in idx], idx

    if evaluator_base is None:
        class Evaluator:                 # PropheseeEvaluator's buffer semantics
            LABELS, PREDICTIONS = "lables", "predictions"

            def __init__(self, dataset, downsample_by_2):
                self.dataset, self.downsample_by_2 = dataset, downsample_by_2
                self._reset_buffer()

            def _reset_buffer(self):
                self._buffer = {self.LABELS: [], self.PREDICTIONS: []}

            def add_labels(self, labels):
                self._buffer[self.LABELS].extend(labels)

            def add_predictions(self, predictions):
                self._buffer[self.PREDICTIONS].extend(predictions)

            def evaluate_buffer(self, img_height, img_width):
                return {"AP": 0.5, "n": float(len(self._buffer[self.LABELS]))}
    else:
        class Evaluator(evaluator_base):
            pass

    class Module:                        # modules.detection.Module, val/test step reduced to the element order
        def __init__(self, evaluator, mode, pred_fn=lambda lab: lab.copy()):
            self.mode_2_psee_evaluator = {mode: evaluator}
            self.pred_fn = pred_fn

        def get_data_from_batch(self, batch):
            return batch["data"]

        def get_worker_id_from_batch(self, batch):
            return batch["worker_id"]

        def _val_test_step_impl(self, batch, mode):
            obj_labels = []
            for sparse in batch["data"]["labels"]:
                current, _ = sparse.get_valid_labels_and_batch_indices()
                if len(current) > 0:
                    obj_labels.extend(current)
            if not obj_labels:
                return None
            ev = self.mode_2_psee_evaluator[mode]
            ev.add_labels(obj_labels)
            ev.add_predictions([self.pred_fn(lab) for lab in obj_labels])
            return {}

    return Module, Labels, Evaluator


def run(module, batches, mode=Mode.TEST):
    for b in batches:
        module._val_test_step_impl(batch=b, mode=mode)
    return module.mode_2_psee_evaluator[mode].evaluate_buffer(img_height=240, img_width=304)


def installed(tmp_path, evaluator_base=None):
    Module, Labels, Evaluator = make_classes(evaluator_base)
    rec = bc.Recorder(tmp_path / "cap.pkl")
    bc.install(rec, Module, Labels, Evaluator, FIRST)
    return rec, Module, Labels, Evaluator


def mark(i):
    return np.array([i], dtype=np.int64)


# two workers, two batch slots each; recordings start where a slot's first flag is set; worker 1's last batch is padding
STREAM = [
    (0, [True, True], [[mark(0), None], [None, mark(1)]]),       # w0: rec 0 in slot 0, rec 1 in slot 1
    (1, [True, True], [[None, None], [None, mark(2)]]),          # w1: rec 2 (no labels yet), rec 3 in slot 1
    (0, [False, True], [[mark(3), mark(4)]]),                    # w0: slot 0 still rec 0, slot 1 starts rec 4
    (1, [False, False], [[None, None]]),                         # w1: padding, no labels
]
EXPECTED_SEQ = [0, 1, 3, 0, 4]


def _rebatch(Labels):
    return [{"worker_id": w, "data": {FIRST: np.array(f), "labels": [Labels(s) for s in steps]}} for w, f, steps in STREAM]


def test_apply_is_a_noop_without_the_env_var(monkeypatch):
    monkeypatch.delenv(bc.ENV, raising=False)
    assert bc.apply() is False


def test_apply_refuses_a_relative_path(monkeypatch):
    monkeypatch.setenv(bc.ENV, "captures/arm.pkl")
    with pytest.raises(ValueError, match="absolute"):
        bc.apply()


def test_elements_are_tagged_with_their_recording(tmp_path):
    rec, Module, Labels, Evaluator = installed(tmp_path)
    ev = Evaluator(dataset="gen1", downsample_by_2=False)
    metrics = run(Module(ev, Mode.TEST), _rebatch(Labels))
    assert metrics == {"AP": 0.5, "n": 5.0}                       # the evaluation itself is untouched

    cap = bootstrap.load_capture(tmp_path / "cap.pkl", "arm")
    assert cap.seq_ids.tolist() == EXPECTED_SEQ
    assert [int(x[0]) for x in cap.labels] == [0, 1, 2, 3, 4]
    assert all(a is not b and np.array_equal(a, b) for a, b in zip(cap.labels, cap.predictions))
    assert cap.meta["evaluator_kwargs"] == {"dataset": "gen1", "downsample_by_2": False}
    assert cap.meta["evaluate_kwargs"] == {"img_height": 240, "img_width": 304}
    assert cap.meta["official_metrics"] == {"AP": 0.5, "n": 5.0}
    assert cap.meta["mode"] == "TEST" and cap.meta["n_recordings"] == 4


def test_validation_evaluation_is_not_captured(tmp_path, capsys):
    rec, Module, Labels, Evaluator = installed(tmp_path)
    run(Module(Evaluator(dataset="gen1", downsample_by_2=False), Mode.VAL), _rebatch(Labels), mode=Mode.VAL)
    assert not (tmp_path / "cap.pkl").exists()
    assert "test only" in capsys.readouterr().err


def test_an_inconsistent_step_disables_the_capture_but_not_the_eval(tmp_path, capsys):
    rec, Module, Labels, Evaluator = installed(tmp_path)
    ev = Evaluator(dataset="gen1", downsample_by_2=False)
    module = Module(ev, Mode.TEST)
    module._val_test_step_impl(batch=_rebatch(Labels)[0], mode=Mode.TEST)
    ev.add_labels([mark(9)])                                       # an element no step accounted for
    ev.add_predictions([mark(9)])
    assert ev.evaluate_buffer(img_height=240, img_width=304) == {"AP": 0.5, "n": 3.0}
    assert not (tmp_path / "cap.pkl").exists()
    assert "NOT written" in capsys.readouterr().err


def test_install_twice_does_not_double_count(tmp_path):
    rec, Module, Labels, Evaluator = installed(tmp_path)
    bc.install(rec, Module, Labels, Evaluator, FIRST)
    run(Module(Evaluator(dataset="gen1", downsample_by_2=False), Mode.TEST), _rebatch(Labels))
    assert bootstrap.load_capture(tmp_path / "cap.pkl", "arm").seq_ids.tolist() == EXPECTED_SEQ


def _boxes(t, specs, dtype):
    """Prophesee boxes at time t: specs = [(x, y, w, h, class_id, confidence)]."""
    a = np.zeros(len(specs), dtype=dtype)
    for i, (x, y, w, h, c, conf) in enumerate(specs):
        a[i] = (t, x, y, w, h, c, 0, conf)
    return a


def test_rescoring_the_capture_reproduces_the_real_rvt_metrics(tmp_path):
    from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
    from utils.evaluation.prophesee.io.box_loading import BBOX_DTYPE

    rng = np.random.default_rng(0)

    def pred_fn(lab):                    # jittered true boxes with varied confidence, plus one false positive
        p = lab.copy()
        p["x"] += rng.normal(0, 4, len(p)).astype(np.float32)
        p["y"] += rng.normal(0, 4, len(p)).astype(np.float32)
        p["class_confidence"] = rng.uniform(0.3, 1.0, len(p)).astype(np.float32)
        fp = _boxes(int(lab["t"][0]), [(200, 150, 50, 40, 1, 0.4)], BBOX_DTYPE)
        return np.concatenate([p, fp])

    t = lambda k: 1_000_000 + 50_000 * k                          # past the evaluator's 0.5 s skip window
    lab = lambda k, specs: _boxes(t(k), specs, BBOX_DTYPE)
    stream = [
        (0, [True, True], [[lab(0, [(10, 20, 60, 40, 0, 1)]), None], [None, lab(1, [(100, 80, 30, 50, 1, 1)])]]),
        (1, [True, True], [[lab(2, [(50, 50, 80, 60, 0, 1), (150, 60, 40, 80, 1, 1)]), None]]),
        (0, [False, True], [[lab(3, [(12, 22, 60, 40, 0, 1)]), lab(4, [(30, 100, 90, 45, 0, 1)])]]),
    ]
    rec, Module, Labels, Evaluator = installed(tmp_path, evaluator_base=PropheseeEvaluator)
    ev = Evaluator(dataset="gen1", downsample_by_2=False)
    batches = [{"worker_id": w, "data": {FIRST: np.array(f), "labels": [Labels(s) for s in steps]}}
               for w, f, steps in stream]
    official = run(Module(ev, Mode.TEST, pred_fn=pred_fn), batches)

    cap = bootstrap.load_capture(tmp_path / "cap.pkl", "arm")
    assert cap.seq_ids.tolist() == [0, 1, 2, 0, 4]
    rescored = bootstrap.RvtScorer(cap.meta)(cap.labels, cap.predictions)
    assert rescored == {k: float(v) for k, v in official.items()}
    assert 0.0 < rescored["AP"] < 1.0                               # a non-trivial score, not an empty-input 0
