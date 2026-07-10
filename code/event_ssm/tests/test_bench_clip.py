import h5py, hdf5plugin
import numpy as np
import torch
from event_ssm.benchmark.bench_clip import load_bench_clip


def _make_fake_tree(tmp_path, T=16):
    seq = tmp_path / "seq_000" / "event_representations_v2" / "stacked_histogram_dt=50_nbins=10"
    seq.mkdir(parents=True)
    with h5py.File(seq / "event_representations.h5", "w") as f:
        f.create_dataset("data", data=np.random.randint(0, 10, (T, 20, 240, 304), dtype=np.uint8))
    return tmp_path


def test_load_bench_clip_shape_pad_cache(tmp_path):
    root = _make_fake_tree(tmp_path)
    cache = tmp_path / "clip.pt"
    clip = load_bench_clip(n_frames=8, root=root, cache=cache, skip=2)
    assert clip.shape == (8, 20, 256, 320) and clip.dtype == torch.float32
    assert clip[:, :, 240:, :].abs().sum() == 0 and clip[:, :, :, 304:].abs().sum() == 0  # zero pad
    assert cache.exists()
    clip2 = load_bench_clip(n_frames=8, root=root, cache=cache)   # cache hit path
    assert torch.equal(clip, clip2)
