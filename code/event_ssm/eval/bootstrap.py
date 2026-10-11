"""Paired, sequence-level bootstrap confidence intervals for Gen1 test AP (Stage 21, item 1 of
docs/plans/2026-10-11-post-training-spiking-analysis.md).

WHAT IT ESTIMATES. Test-set sampling uncertainty: how much an arm's test AP, and the gap between two
arms, depends on which of the 470 Gen1 test recordings happen to be in the test set. It does NOT measure
training-seed variance (one seed per arm); the thesis must say so.

HOW. The official evaluation is captured once per arm (Part A, `bootstrap_capture.py`, run inside the
normal test eval with BOOTSTRAP_CAPTURE=/abs/path.pkl): the exact per-element label and prediction arrays the RVT
`PropheseeEvaluator` receives, each tagged with the test recording it came from. Here, on CPU:

  1. Recordings are resampled with replacement, B times (one (B, n_seq) index table, fixed by `seed`).
  2. Replicate b scores EVERY arm on the SAME resampled recordings (paired design): the shared
     difficulty of a recording cancels in a gap, so a gap's interval is far narrower than the two
     arms' intervals combined.
  3. The point estimate is the score on the full, unresampled capture, which must equal the official
     test AP (the identity check, `--identity --expect`); the 95 % interval is the percentile interval of
     the B replicate scores.

Scoring is pluggable (`score_fn(labels, predictions) -> {metric: float}`). The default `RvtScorer`
rebuilds the same RVT evaluator with the arguments the capture recorded, so the bootstrap uses the
official filters and COCO settings; the unit tests inject a toy scorer (no torch, no RVT needed).

A gap "a:b" is reported as b - a: what moving from arm a to arm b changes (negative = a cost), e.g.
`--pair graded:binary` is the cost of binarisation.

Usage (workstation, after the captures exist):
  python -m event_ssm.eval.bootstrap \\
      --capture puressm=results/stage21_test_eval/puressm_100k/bootstrap_capture.pkl \\
      --capture analog=... --capture graded=... --capture binary=... \\
      --pair puressm:analog --pair analog:graded --pair graded:binary \\
      --n-boot 1000 --workers 8 --out results/stage21_bootstrap/ladder_100k
  python -m event_ssm.eval.bootstrap --capture puressm=... --identity --expect puressm=0.4643
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

CAPTURE_FORMAT = "event_ssm.bootstrap_capture/v1"

ScoreFn = Callable[[List[np.ndarray], List[np.ndarray]], Dict[str, float]]


class CaptureMismatch(ValueError):
    """Two arms' captures do not describe the same test elements, so pairing them would be invalid."""


@dataclass(frozen=True)
class Capture:
    """One arm's captured evaluator input: element i = (labels[i], predictions[i]) of recording seq_ids[i]."""
    name: str
    seq_ids: np.ndarray            # (n,) int, the test recording each element belongs to
    labels: List[np.ndarray]       # n arrays, exactly as handed to PropheseeEvaluator.add_labels
    predictions: List[np.ndarray]  # n arrays, exactly as handed to PropheseeEvaluator.add_predictions
    meta: dict                     # evaluator construction/evaluation args + provenance (ckpt, tag)


# ── capture I/O ────────────────────────────────────────────────────────────────────────────────────

def save_capture(path, seq_ids, labels, predictions, meta) -> None:
    """Write a capture file (used by Part A). Lists are stored as given; nothing is converted."""
    if not (len(seq_ids) == len(labels) == len(predictions)):
        raise ValueError(f"lengths differ: {len(seq_ids)} seq_ids, {len(labels)} labels, "
                         f"{len(predictions)} predictions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"format": CAPTURE_FORMAT, "meta": dict(meta),
                     "seq_ids": np.asarray(seq_ids, dtype=np.int64),
                     "labels": list(labels), "predictions": list(predictions)}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_capture(path, name: str) -> Capture:
    """Read and validate a capture file. Only load captures this project wrote (pickle)."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or d.get("format") != CAPTURE_FORMAT:
        raise ValueError(f"{path}: not a {CAPTURE_FORMAT} capture")
    seq_ids = np.asarray(d["seq_ids"], dtype=np.int64)
    if not (len(seq_ids) == len(d["labels"]) == len(d["predictions"])):
        raise ValueError(f"{path}: element lists have different lengths")
    return Capture(name, seq_ids, list(d["labels"]), list(d["predictions"]), dict(d["meta"]))


# ── resampling ─────────────────────────────────────────────────────────────────────────────────────

def sequence_groups(seq_ids: np.ndarray) -> List[np.ndarray]:
    """Element indices per recording, recordings in order of first appearance, element order kept."""
    seq_ids = np.asarray(seq_ids)
    _, first = np.unique(seq_ids, return_index=True)
    order = seq_ids[np.sort(first)]
    return [np.flatnonzero(seq_ids == s) for s in order]


def check_aligned(caps: Sequence[Capture]) -> None:
    """Pairing is valid only if every arm saw the same elements of the same recordings with the same labels."""
    ref = caps[0]
    for c in caps[1:]:
        if c.seq_ids.shape != ref.seq_ids.shape or not np.array_equal(c.seq_ids, ref.seq_ids):
            raise CaptureMismatch(f"{c.name} vs {ref.name}: different recording/element layout")
        for i, (a, b) in enumerate(zip(ref.labels, c.labels)):
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                raise CaptureMismatch(f"{c.name} vs {ref.name}: ground-truth labels differ at element {i}")


def draw_samples(n_seq: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_seq) recording indices drawn with replacement; one table shared by every arm."""
    return np.random.default_rng(seed).integers(0, n_seq, size=(n_boot, n_seq))


def gather(cap: Capture, groups: List[np.ndarray], seq_sample: np.ndarray):
    """The resampled evaluator input: every element of every drawn recording, duplicates kept."""
    idx = np.concatenate([groups[s] for s in seq_sample]) if len(seq_sample) else np.empty(0, int)
    return [cap.labels[i] for i in idx], [cap.predictions[i] for i in idx]


def percentile_ci(values, level: float = 0.95) -> Tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.asarray(values, dtype=float), [alpha, 1.0 - alpha])
    return float(lo), float(hi)


# ── the bootstrap ──────────────────────────────────────────────────────────────────────────────────

_W: dict = {}   # per-worker state (set once by the pool initializer, so captures are not re-pickled per task)


def _init_worker(caps, groups, samples, score_fn):
    _W.update(caps=caps, groups=groups, samples=samples, score_fn=score_fn)


def _replicate(b: int) -> Dict[str, Dict[str, float]]:
    seq_sample = _W["samples"][b]
    return {c.name: _W["score_fn"](*gather(c, _W["groups"], seq_sample)) for c in _W["caps"]}


def run_bootstrap(caps: Sequence[Capture], score_fn: ScoreFn, n_boot: int = 1000, seed: int = 0,
                  pairs: Sequence[Tuple[str, str]] = (), level: float = 0.95, workers: int = 1,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    """Score every arm on the full capture (point) and on n_boot shared resamples (interval)."""
    caps = list(caps)
    names = [c.name for c in caps]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arm names: {names}")
    for a, b in pairs:
        if a not in names or b not in names:
            raise ValueError(f"pair {a}:{b} names an arm with no capture (have {names})")
    check_aligned(caps)

    groups = sequence_groups(caps[0].seq_ids)
    samples = draw_samples(len(groups), n_boot, seed)
    point = {c.name: score_fn(c.labels, c.predictions) for c in caps}

    reps: List[Dict[str, Dict[str, float]]] = []
    if workers <= 1:
        _init_worker(caps, groups, samples, score_fn)
        for b in range(n_boot):
            reps.append(_replicate(b))
            if progress:
                progress(b + 1, n_boot)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(caps, groups, samples, score_fn)) as ex:
            for b, r in enumerate(ex.map(_replicate, range(n_boot), chunksize=max(1, n_boot // (8 * workers)))):
                reps.append(r)      # ex.map keeps replicate order, so results do not depend on `workers`
                if progress:
                    progress(b + 1, n_boot)

    metrics = [m for m in point[names[0]] if all(m in point[n] for n in names)]

    def summarise(pt: float, vals: np.ndarray) -> dict:
        lo, hi = percentile_ci(vals, level)
        return {"point": float(pt), "ci_lo": lo, "ci_hi": hi, "se": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    arms = {n: {m: summarise(point[n][m], np.array([r[n][m] for r in reps])) for m in metrics} for n in names}
    gaps = {}
    for a, b in pairs:
        gaps[f"{a}:{b}"] = {}
        for m in metrics:
            diff = np.array([r[b][m] - r[a][m] for r in reps])
            s = summarise(point[b][m] - point[a][m], diff)
            s["frac_reps_ge_0"] = float(np.mean(diff >= 0))   # share of resamples where b scores at least a
            gaps[f"{a}:{b}"][m] = s
    return {"config": {"n_boot": n_boot, "seed": seed, "level": level, "n_sequences": len(groups),
                       "n_elements": int(len(caps[0].seq_ids)), "arms": names,
                       "pairs": [f"{a}:{b}" for a, b in pairs], "metrics": metrics},
            "arms": arms, "gaps": gaps}


# ── scorers ────────────────────────────────────────────────────────────────────────────────────────

class RvtScorer:
    """Re-score with the official RVT evaluator, rebuilt from the arguments the capture recorded.

    The capture's meta must hold `evaluator_kwargs` (PropheseeEvaluator construction) and
    `evaluate_kwargs` (evaluate_buffer call). Needs RVT on sys.path (workstation). Correctness is
    established by the identity check: the unresampled score must equal the official test AP."""

    def __init__(self, meta: dict):
        self.evaluator_kwargs = dict(meta["evaluator_kwargs"])
        self.evaluate_kwargs = dict(meta["evaluate_kwargs"])

    def __call__(self, labels, predictions) -> Dict[str, float]:
        from utils.evaluation.prophesee.evaluator import PropheseeEvaluator   # RVT, imported lazily
        ev = PropheseeEvaluator(**self.evaluator_kwargs)
        ev.add_labels(labels)
        ev.add_predictions(predictions)
        out = ev.evaluate_buffer(**self.evaluate_kwargs)
        return {k: float(v) for k, v in out.items() if isinstance(v, (int, float, np.floating, np.integer))}


def load_scorer(spec: str, meta: dict) -> ScoreFn:
    """'rvt' -> RvtScorer(meta); 'package.module:attr' -> that callable (tests, other evaluators)."""
    if spec == "rvt":
        return RvtScorer(meta)
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


# ── report ─────────────────────────────────────────────────────────────────────────────────────────

def to_markdown(result: dict, scale: float = 100.0) -> str:
    """Point [lo, hi] per arm and per gap, in AP points (x100) by default."""
    cfg = result["config"]
    f = lambda s: f"{s['point'] * scale:.2f} [{s['ci_lo'] * scale:.2f}, {s['ci_hi'] * scale:.2f}]"
    head = "| | " + " | ".join(cfg["metrics"]) + " |\n|---|" + "---|" * len(cfg["metrics"]) + "\n"
    out = [f"Paired sequence-level bootstrap: B = {cfg['n_boot']}, seed {cfg['seed']}, "
           f"{cfg['n_sequences']} test recordings, {int(cfg['level'] * 100)} % percentile intervals. "
           "Test-set sampling uncertainty only (one training seed per arm).\n",
           "**Arms**\n\n" + head + "".join(f"| {n} | " + " | ".join(f(result['arms'][n][m]) for m in cfg["metrics"]) + " |\n"
                                       for n in cfg["arms"])]
    if result["gaps"]:
        out.append("\n**Gaps (b − a)**\n\n" + head + "".join(
            f"| {g} | " + " | ".join(f(result['gaps'][g][m]) for m in cfg["metrics"]) + " |\n" for g in cfg["pairs"]))
    return "\n".join(out)


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────

def _kv(s: str) -> Tuple[str, str]:
    k, sep, v = s.partition("=")
    if not sep or not k or not v:
        raise argparse.ArgumentTypeError(f"expected name=value, got {s!r}")
    return k, v


def _pair(s: str) -> Tuple[str, str]:
    a, sep, b = s.partition(":")
    if not sep or not a or not b:
        raise argparse.ArgumentTypeError(f"expected a:b, got {s!r}")
    return a, b


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--capture", type=_kv, action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--pair", type=_pair, action="append", default=[], metavar="A:B", help="report B - A")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--metrics", nargs="*", default=None, help="keep only these metrics (default: all)")
    p.add_argument("--scorer", default="rvt", help="'rvt' or 'package.module:callable'")
    p.add_argument("--out", type=Path, default=None, help="directory for bootstrap.json + bootstrap.md")
    p.add_argument("--identity", action="store_true", help="score the full captures only, no resampling")
    p.add_argument("--expect", type=_kv, action="append", default=[], metavar="NAME=AP",
                   help="with --identity: fail unless the full-capture AP matches (abs tol 5e-5)")
    a = p.parse_args(argv)

    caps = [load_capture(path, name) for name, path in a.capture]
    score_fn = load_scorer(a.scorer, caps[0].meta)
    if a.metrics:
        full_fn, keep = score_fn, list(a.metrics)
        score_fn = _Subset(full_fn, keep)

    if a.identity:
        ok = True
        expect = {k: float(v) for k, v in a.expect}
        for c in caps:
            ap = score_fn(c.labels, c.predictions)
            line = f"[identity] {c.name}: " + ", ".join(f"{k}={v:.4f}" for k, v in ap.items())
            if c.name in expect:
                match = abs(ap.get("AP", float("nan")) - expect[c.name]) <= 5e-5
                ok &= match
                line += f"  expected AP={expect[c.name]:.4f} -> {'MATCH' if match else 'MISMATCH'}"
            print(line)
        return 0 if ok else 1

    t0 = time.time()

    def progress(done, total):
        if done == 1 or done % max(1, total // 20) == 0 or done == total:
            el = time.time() - t0
            print(f"[bootstrap] {done}/{total} replicates, {el:.0f} s elapsed, "
                  f"~{el / done * (total - done):.0f} s left", file=sys.stderr)

    result = run_bootstrap(caps, score_fn, a.n_boot, a.seed, a.pair, a.level, a.workers, progress)
    result["config"]["captures"] = {name: str(path) for name, path in a.capture}
    result["config"]["seconds"] = round(time.time() - t0, 1)
    md = to_markdown(result)
    print(md)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "bootstrap.json").write_text(json.dumps(result, indent=2))
        (a.out / "bootstrap.md").write_text(md)
        print(f"[bootstrap] wrote {a.out}/bootstrap.json and bootstrap.md", file=sys.stderr)
    return 0


class _Subset:
    """Picklable wrapper keeping only the requested metrics (so --workers > 1 still works)."""
    def __init__(self, fn, keep):
        self.fn, self.keep = fn, keep

    def __call__(self, labels, predictions):
        out = self.fn(labels, predictions)
        return {k: out[k] for k in self.keep if k in out}


if __name__ == "__main__":
    sys.exit(main())
