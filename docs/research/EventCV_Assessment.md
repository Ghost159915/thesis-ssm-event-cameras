# EventCV — Assessment for this Thesis (2026-07-11)

Supervisor-suggested tool (LinkedIn post, 2026-07). Researched 2026-07-11.

## What it is

Open-source "OpenCV for event cameras": Python library over a **Rust core** (`eventcv-core`).
- PyPI `eventcv` **v1.0.2** (released 2026-07-10 — days old), author **Adam Hines, EventLAB/QUT** (Fischer/Milford neuromorphic group).
- Site: <https://eventcv.net> · Docs: <https://eventcv.readthedocs.io> · Rust core: <https://docs.rs/eventcv-core>

## Environment safety (verified from PyPI metadata)

- **Only runtime dependency: numpy** (h5py optional, tests only). Prebuilt **cp39-abi3** wheels incl. Linux x86_64.
- **No torch/opencv/CUDA anywhere in its dependency tree** ⇒ `pip install eventcv` into `events_signals` **cannot** disturb the torch 2.11/cu128 stack. (The `--no-deps` house rule targets torch-dependent packages; this is not one.)
- Caveat: v1.0 is days old — treat as a convenience layer, never a load-bearing dependency.

## Capabilities relevant to us

- Loads **Prophesee `.dat`** (our Gen1 raw format), ROS `.bag`, HDF5, AEDAT 2.0, npz/txt/csv.
- 11+ representations: voxel grids, time surfaces, count images, polarity reps, optical flow, connected components — returned as numpy arrays (PyTorch-ready).
- Chainable spatial/temporal/polarity transforms; fast Rust parsing; cross-platform.
- API shape: `ecv.load("seq.dat")` / `ecv.open(...)` streaming → `stream.voxel()` etc.

## Where it aids the thesis (ranked)

1. **Stage-16 visual unit (in scope, spec U6):** GT-vs-pred overlay videos and large-car qualitative figures for EventSSM vs PureSSM — EventCV reads raw `.dat` and renders event frames/time surfaces; we overlay boxes. Simultaneously produces the qualitative AP_L evidence Stage 8 lacked and clears the two deferred video TODOs (smooth overlay, GT-vs-pred).
2. **Thesis figures:** background/methodology illustrations (event stream, voxel grid, time surface) from our own data; citing a community tool = good reproducibility practice.
3. **Future work:** Gen3.1 stereo camera (`.bag`/ROS ✓), UAV-dataset ingestion (assorted formats).
4. **Cross-validation only:** sanity-check event counts from the Stage-9 dat→h5 converter.

## Hard line — where it must NOT be used

**Never for training/eval tensor generation.** EventSSM and S5-RVT were trained and evaluated on the frozen RVT stacked-histogram tensors (`stacked_histogram_dt=50_nbins=10`). Regenerating those with a different library (different binning semantics, however slight) silently breaks the fair-comparison spine of the whole thesis. EventCV lives strictly downstream: visualization, figures, new-data ingestion.

## Install plan (when Stage 16 starts)

```bash
conda run -n events_signals pip install eventcv
# smoke: load one Gen1 .dat, render a count image, compare event count vs our converter
```
