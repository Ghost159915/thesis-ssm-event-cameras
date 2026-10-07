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
PAD_H, PAD_W = 256, 320  # Gen1 stacked-histogram frames (20, 240, 304) zero-padded network input


def pad_to_network_shape(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Top-left zero-pad `src`'s trailing (H, W) into `dst`'s corresponding top-left region, in
    place, and return `dst`. `dst` must already be zero everywhere else (e.g. fresh from
    `torch.zeros`, or a reused buffer whose margin was never written); every leading dim must
    already match between `src` and `dst`, only the trailing (H, W) differ (`dst` >= `src` there).

    One shared implementation of the Gen1 240x304 -> 256x320 pad step (DRY) -- used by
    `load_bench_clip` below (whole-clip batch) and
    `scripts/stage16_dump_predictions.py:iter_rec_frames` (per-frame reused buffer), which
    previously duplicated this same top-left assignment.
    """
    h, w = src.shape[-2], src.shape[-1]
    dst[..., :h, :w] = src
    return dst


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
    clip = torch.zeros(n_frames, t.shape[1], PAD_H, PAD_W)
    pad_to_network_shape(clip, t)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"clip": clip, "src": str(h5)}, cache)
    return clip


def _n_frames(seq: pathlib.Path) -> int:
    import h5py, hdf5plugin  # noqa: F401
    with h5py.File(seq / EVR, "r") as f:
        return f[KEY if KEY in f else list(f.keys())[0]].shape[0]


def iter_rate_clips(k: int = 16, n_frames: int = 96, root=None, skip: int = 100):
    """Stage 22 firing-rate data: k clips of n_frames consecutive dt=50 test frames from sequences evenly spaced
    over the sorted test set (deterministic), zero-padded like the benchmark clip. A pick shorter than
    skip + n_frames moves to the next long-enough sequence. Yields (sequence name, clip) one at a time: sixteen
    float clips at once would need ~10 GB. Selection happens before the first yield, so a shortage raises early."""
    import h5py, hdf5plugin  # noqa: F401
    root = pathlib.Path(root) if root else DEFAULT_ROOT
    seqs = sorted(p for p in root.iterdir() if p.is_dir())
    picks, used = [], set()
    for j in range(k):
        i = (j * len(seqs)) // k
        while i < len(seqs) and (i in used or _n_frames(seqs[i]) < skip + n_frames):
            i += 1
        if i == len(seqs):
            raise ValueError(f"not enough test sequences long enough for {k} clips of {skip}+{n_frames} frames")
        used.add(i)
        picks.append(seqs[i])

    def _gen():
        for seq in picks:
            with h5py.File(seq / EVR, "r") as f:
                arr = f[KEY if KEY in f else list(f.keys())[0]][skip:skip + n_frames]
            t = torch.from_numpy(arr).float()
            yield seq.name, pad_to_network_shape(torch.zeros(n_frames, t.shape[1], PAD_H, PAD_W), t)
    return _gen()
