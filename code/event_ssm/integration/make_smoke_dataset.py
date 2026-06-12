"""Build a tiny Gen1 smoke dataset by symlinking K real sequences into train/val/test.
Only the Gen1 `test/` split is present locally; overfitting test data is fine for a smoke
(goal = prove the machinery learns, not generalisation). Data dir is gitignored.

Usage: python -m event_ssm.integration.make_smoke_dataset   (run from code/)
"""
import argparse, pathlib, shutil

REPO = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO / "data/gen1_raw/gen1/test"
DEFAULT_DEST = REPO / "data/gen1_smoke"
LEAF_REPR = "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5"
LEAF_LABELS = "labels_v2/labels.npz"


def build_smoke_dataset(src: pathlib.Path = DEFAULT_SRC,
                        dest: pathlib.Path = DEFAULT_DEST, k: int = 2) -> pathlib.Path:
    src, dest = pathlib.Path(src), pathlib.Path(dest)
    assert src.is_dir(), f"source split not found: {src}"
    seqs = sorted(p for p in src.iterdir() if p.is_dir()
                  and (p / LEAF_REPR).exists() and (p / LEAF_LABELS).exists())
    assert len(seqs) >= k, f"need >= {k} valid sequences in {src}, found {len(seqs)}"
    chosen = seqs[:k]
    if dest.exists():
        shutil.rmtree(dest)
    for split in ("train", "val", "test"):
        sp = dest / split
        sp.mkdir(parents=True, exist_ok=True)
        for seq in chosen:
            link = sp / seq.name
            link.symlink_to(seq.resolve(), target_is_directory=True)
            assert (link / LEAF_REPR).exists(), f"broken symlink leaf: {link}"
    print(f"smoke dataset @ {dest}  (k={k} seqs x 3 splits)")
    for split in ("train", "val", "test"):
        names = [p.name for p in sorted((dest / split).iterdir())]
        print(f"  {split}: {names}")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--k", type=int, default=2)
    a = ap.parse_args()
    build_smoke_dataset(pathlib.Path(a.src), pathlib.Path(a.dest), a.k)
