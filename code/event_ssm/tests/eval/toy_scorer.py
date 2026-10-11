"""A deterministic stand-in for the RVT evaluator, for testing the bootstrap mechanics without torch or RVT.

score = recall: the fraction of ground-truth boxes (over all elements) that have an identical row among
that element's predictions. Perfect predictions -> 1.0, no predictions -> 0.0; every ground-truth box
weighs the same, like the pooled COCO evaluation. Module-level so spawned workers can import it."""
import numpy as np

BOX_DTYPE = np.dtype([("t", "<u8"), ("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4"), ("class_id", "<u4")])


def boxes(rows):
    return np.array([tuple(r) for r in rows], dtype=BOX_DTYPE)


def score(labels, predictions):
    hit = total = 0
    for gt, dt in zip(labels, predictions):
        dt_rows = {tuple(r) for r in dt.tolist()}
        total += len(gt)
        hit += sum(tuple(r) in dt_rows for r in gt.tolist())
    recall = hit / total if total else 0.0
    return {"AP": recall, "AP_L": recall}
