"""Real Gen1 frames for the benchmark (spec §4): 64 consecutive dt=50 test frames, zero-padded
240x304 -> 256x320, float32 raw counts — matching the RVT eval feed (uint8 histograms are cast to
float by the module input path; verified during Task-4 discovery at RVT/modules/detection.py:158).
Cached for reproducibility."""
import pathlib

import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "data/gen1_raw/gen1/test"
DEFAULT_CACHE = REPO / "results/stage10/bench_clip.pt"
EVR = "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5"
KEY = "data"   # confirmed against the real file in Task-4 discovery


def load_bench_clip(n_frames: int = 64, root=None, cache=None, skip: int = 100) -> torch.Tensor:
    import h5py, hdf5plugin  # noqa: F401  (hdf5plugin registers the blosc filter)
    root = pathlib.Path(root) if root else DEFAULT_ROOT
    cache = pathlib.Path(cache) if cache else DEFAULT_CACHE
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        if blob["clip"].shape[0] == n_frames:
            return blob["clip"]
    seq = sorted(p for p in root.iterdir() if p.is_dir())[0]     # deterministic first sequence
    h5 = seq / EVR
    with h5py.File(h5, "r") as f:
        key = KEY if KEY in f else list(f.keys())[0]
        assert f[key].shape[0] >= skip + n_frames, f"sequence too short: {f[key].shape}"
        arr = f[key][skip:skip + n_frames]                        # (n, 20, 240, 304) uint8
    t = torch.from_numpy(arr).float()
    clip = torch.zeros(n_frames, t.shape[1], 256, 320)
    clip[:, :, :240, :304] = t
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"clip": clip, "src": str(h5)}, cache)
    return clip
