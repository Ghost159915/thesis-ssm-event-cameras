"""Stage-6 Task 10: fixed-seed 10%-recording train-subset builder (ISSUE-10). CPU-only (no CUDA)."""
from pathlib import Path
from event_ssm.integration.make_train_subset import select_recordings, build_subset, LEAF_REPR, LEAF_LABELS


def test_deterministic_ten_percent(tmp_path):
    """select_recordings is reproducible, samples ~frac, and is seed-sensitive (not first-N)."""
    src = tmp_path / "train"; src.mkdir()
    for i in range(50):
        (src / f"rec_{i:03d}").mkdir()
    a = select_recordings(src, frac=0.10, seed=1234)
    b = select_recordings(src, frac=0.10, seed=1234)
    assert a == b                                          # deterministic for a given seed
    assert len(a) == 5                                     # 10% of 50
    assert select_recordings(src, frac=0.10, seed=1) != a  # seed matters (random, not first-N)
    assert a != [f"rec_{i:03d}" for i in range(5)]         # explicitly NOT the first 10% (ISSUE-10)


def _mk_recording(split_dir: Path, name: str):
    """Create a fake recording dir with the two required data leaves (repr .h5 + labels .npz)."""
    rec = split_dir / name
    (rec / LEAF_REPR).parent.mkdir(parents=True, exist_ok=True)
    (rec / LEAF_REPR).touch()
    (rec / LEAF_LABELS).parent.mkdir(parents=True, exist_ok=True)
    (rec / LEAF_LABELS).touch()


def test_build_subset_train_subset_with_full_val_test(tmp_path):
    """build_subset: train = random `frac` subset; val/test = FULL real splits; list logged; idempotent."""
    root = tmp_path / "gen1"
    for split, n in (("train", 20), ("val", 4), ("test", 3)):
        (root / split).mkdir(parents=True)
        for i in range(n):
            _mk_recording(root / split, f"{split}_{i:03d}")
    dest = tmp_path / "subset"

    chosen = build_subset(root, dest, frac=0.10, seed=7)
    assert len(chosen) == 2                                            # 10% of 20
    assert sorted(p.name for p in (dest / "train").iterdir()) == chosen
    assert len(list((dest / "val").iterdir())) == 4                    # val = full
    assert len(list((dest / "test").iterdir())) == 3                   # test = full
    for r in chosen:                                                   # symlinks resolve to real leaves
        assert (dest / "train" / r / LEAF_LABELS).exists()
    assert (dest / "train_subset_recordings.txt").read_text().split() == chosen

    # idempotent: rebuilding with the same seed reproduces the same selection on a clean tree
    chosen2 = build_subset(root, dest, frac=0.10, seed=7)
    assert chosen2 == chosen
    assert sorted(p.name for p in (dest / "train").iterdir()) == chosen
