"""Additive per-class (car / pedestrian) AP for the RVT/Prophesee COCO eval -- WITHOUT editing the
vendored eval code (preserves the controlled-comparison guarantee; the 6 aggregate metrics are
unchanged, so the overall AP stays directly comparable to the baseline).

How: the RVT evaluator (`utils/evaluation/prophesee/metrics/coco_eval.py::_coco_eval`) builds a COCOeval
over both categories and then keeps only the *category-averaged* `coco_eval.stats` (AP, AP_50, ...). The
per-class values already exist in `coco_eval.eval['precision']` (axis K = category) -- we just read them
out. We monkeypatch `COCOeval.summarize` to also print per-category AP / AP_50 *to stderr* (the RVT code
redirects summarize's STDOUT to devnull, so stdout prints would be swallowed; stderr is not redirected).

Apply AFTER `setup_paths()` (RVT must be importable). Idempotent.
"""
import sys


def apply():
    # Patch the exact COCOeval class the RVT evaluator imported (python pycocotools OR detectron2 cpp).
    from utils.evaluation.prophesee.metrics import coco_eval as ce
    COCOeval = ce.COCOeval
    if getattr(COCOeval.summarize, "_perclass_patched", False):
        return
    _orig_summarize = COCOeval.summarize

    def _summarize_with_perclass(self):
        _orig_summarize(self)                       # unchanged: fills self.stats (the 6 aggregate metrics)
        try:
            precision = self.eval.get("precision")  # shape [T(iou=10), R(recall=101), K(cat), A(area=4), M(maxDet=3)]
            if precision is None:
                return
            cat_ids = list(self.params.catIds)      # K axis order matches this (e.g. [1, 2] -> car, ped)
            id2name = {c["id"]: c["name"] for c in self.cocoGt.loadCats(cat_ids)}

            def _ap(arr):
                arr = arr[arr > -1]
                return float(arr.mean()) if arr.size else float("nan")

            print("[per-class] ===== per-category AP (COCO) =====", file=sys.stderr)
            print(f"[per-class] {'class':>11}  {'AP@[.50:.95]':>12}  {'AP_50':>7}", file=sys.stderr)
            for k, cid in enumerate(cat_ids):
                ap = _ap(precision[:, :, k, 0, -1])      # area=all (0), maxDet=100 (-1), all IoU
                ap50 = _ap(precision[0, :, k, 0, -1])    # IoU=0.50 (index 0)
                name = id2name.get(cid, f"cat{cid}")
                print(f"[per-class] {name:>11}  {ap:>12.4f}  {ap50:>7.4f}", file=sys.stderr)
            print("[per-class] ===================================", file=sys.stderr)
        except Exception as e:  # never let the readout break the (correct, unchanged) aggregate eval
            print(f"[per-class] extraction skipped ({type(e).__name__}: {e})", file=sys.stderr)

    _summarize_with_perclass._perclass_patched = True
    COCOeval.summarize = _summarize_with_perclass
    print("[per-class] patched COCOeval.summarize -> per-class AP will print to stderr", file=sys.stderr)
