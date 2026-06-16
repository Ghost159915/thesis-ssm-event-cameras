"""Build a fixed-seed random 10% Gen1 TRAIN subset for Stage-6 short training (ISSUE-10).

Sample 10% of train RECORDINGS uniformly at random (fixed seed; log the chosen list for
reproducibility). Windows within each recording stay contiguous (whole-recording symlinks) -- we
subsample recordings, never split them. val/test point at the FULL real splits so val-mAP stays
comparable to a full run. Builds a symlink subtree the RVT data module reads. Data dir is gitignored.

ISSUE-10: do NOT take the first 10% (ordering correlates with capture session/conditions); sample
uniformly at random with a fixed seed and log the recording list.

Usage: python -m event_ssm.integration.make_train_subset --gen1-root data/gen1_raw/gen1
"""
import argparse, pathlib, random, shutil
from event_ssm.integration.gen1_paths import LEAF_REPR, LEAF_LABELS   # shared (also re-exported here)

REPO = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "data/gen1_raw/gen1"
DEFAULT_DEST = REPO / "data/gen1_subset10"


def select_recordings(train_dir, frac: float = 0.10, seed: int = 1234):
    """Deterministic fixed-seed random sample of `frac` of the recording dir NAMES under train_dir.
    Leaf-agnostic (lists subdirs only) so it is unit-testable without real data; leaf validation
    happens in build_subset. Returns a sorted list of names (reproducible for a given seed)."""
    recs = sorted(p.name for p in pathlib.Path(train_dir).iterdir() if p.is_dir())
    k = max(1, round(len(recs) * frac))
    return sorted(random.Random(seed).sample(recs, k))


def _link_split(src_split: pathlib.Path, dest_split: pathlib.Path, names):
    """Symlink each named recording from src_split into dest_split, asserting the data leaf exists."""
    dest_split.mkdir(parents=True, exist_ok=True)
    for name in names:
        seq = src_split / name
        assert (seq / LEAF_REPR).exists() and (seq / LEAF_LABELS).exists(), f"invalid recording: {seq}"
        (dest_split / name).symlink_to(seq.resolve(), target_is_directory=True)


def build_subset(gen1_root, dest_root, frac: float = 0.10, seed: int = 1234):
    """Build data/gen1_subset10 = {train: random `frac` of recordings, val/test: full real splits}."""
    gen1_root, dest_root = pathlib.Path(gen1_root), pathlib.Path(dest_root)
    for split in ("train", "val", "test"):
        assert (gen1_root / split).is_dir(), f"missing split: {gen1_root / split}"
    chosen = select_recordings(gen1_root / "train", frac, seed)
    if dest_root.exists():
        shutil.rmtree(dest_root)                                    # rebuild clean (unlinks symlinks only)
    _link_split(gen1_root / "train", dest_root / "train", chosen)   # 10% subset
    for split in ("val", "test"):                                   # FULL real splits (comparable val-mAP)
        names = sorted(p.name for p in (gen1_root / split).iterdir() if p.is_dir())
        _link_split(gen1_root / split, dest_root / split, names)
    (dest_root / "train_subset_recordings.txt").write_text("\n".join(chosen) + "\n")
    n_train = sum(1 for p in (gen1_root / "train").iterdir() if p.is_dir())
    print(f"train subset @ {dest_root}: {len(chosen)} / {n_train} train recordings (frac={frac}, seed={seed})")
    print(f"  val/test = full real splits; chosen recording list -> {dest_root}/train_subset_recordings.txt")
    return chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen1-root", default=str(DEFAULT_ROOT), help="Gen1 root containing train/val/test")
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    build_subset(a.gen1_root, a.dest, a.frac, a.seed)
