#!/usr/bin/env python
"""Stage-9 saturation check.

Question: does the stacked-histogram `count_cutoff` clip the dense (large-dt) rates and
confound the temporal-generalisation curve? At a fixed 50 ms grid stride, a 200 ms window
integrates ~4x the events of a 50 ms window into the SAME 10 temporal bins, so per-(polarity,
bin,pixel) counts are far more likely to hit the clip ceiling. If the dense rates are heavily
clipped, AP changes across rates partly reflect a representation artifact, not the model's
temporal response.

For each dt in {200,100,50,25,12} ms this samples the rebuilt test representations and reports:
  occupancy   = fraction of ALL cells that are nonzero (active)
  sat_active  = fraction of ACTIVE cells pinned at the clip ceiling   <- the key confound metric
  sat_all     = fraction of ALL cells at the ceiling
  mean_count  = mean cell value over all cells (proxy for total event mass / frame)
  p99 / max   = 99th-percentile and max cell value observed (max should equal count_cutoff)

Read-only. Uses the events_signals env (numpy + h5py + hdf5plugin for the blosc codec).
"""
import argparse
import glob
import os

import hdf5plugin  # noqa: F401  registers the blosc codec so h5py can read the representations
import h5py
import numpy as np

DEFAULT_PREPROC = "/home/ghost/Desktop/thesis-ssm-event-cameras/data/gen1_stage9/preproc/test"
RATES = [200, 100, 50, 25, 12]


def rate_stats(preproc: str, dt: int, n_seq: int, frames_per_seq: int):
    pattern = os.path.join(
        preproc, "*", f"event_representations_v2/stacked_histogram_dt={dt}_nbins=10/event_representations.h5"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    files = files[:n_seq]  # deterministic sample (first N sequences alphabetically)

    tot_cells = nonzero = sat = mass = 0
    p99_acc, maxval, nframes = [], 0, 0
    for f in files:
        with h5py.File(f, "r") as h:
            d = h["data"]              # (N, 20, H, W) uint8, values 0..count_cutoff
            n = d.shape[0]
            idxs = np.unique(np.linspace(0, n - 1, min(frames_per_seq, n)).astype(int))
            for i in idxs:
                a = np.asarray(d[i], dtype=np.int64)
                tot_cells += a.size
                nonzero += int((a > 0).sum())
                mass += int(a.sum())
                maxval = max(maxval, int(a.max()))
                nframes += 1
    # second pass for ceiling stats once we know the ceiling (= observed max, the clip value)
    ceil = maxval
    for f in files:
        with h5py.File(f, "r") as h:
            d = h["data"]
            n = d.shape[0]
            idxs = np.unique(np.linspace(0, n - 1, min(frames_per_seq, n)).astype(int))
            for i in idxs:
                a = np.asarray(d[i], dtype=np.int64)
                sat += int((a == ceil).sum())
                nz = a[a > 0]
                if nz.size:
                    p99_acc.append(np.percentile(nz, 99))
    return dict(
        frames=nframes,
        occupancy=nonzero / tot_cells,
        sat_active=(sat / nonzero) if nonzero else 0.0,
        sat_all=sat / tot_cells,
        mean_count=mass / tot_cells,
        p99_nonzero=float(np.mean(p99_acc)) if p99_acc else 0.0,
        ceil=ceil,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc", default=DEFAULT_PREPROC, help="dir containing per-sequence test dirs")
    ap.add_argument("--n-seq", type=int, default=40, help="sequences sampled per rate")
    ap.add_argument("--frames-per-seq", type=int, default=8, help="evenly-spaced frames sampled per sequence")
    args = ap.parse_args()

    print(f"sampling {args.n_seq} seqs x {args.frames_per_seq} frames per rate from {args.preproc}\n")
    hdr = ("dt(ms)", "mult", "occupancy", "sat_active", "sat_all", "mean_count", "p99_nz", "ceil")
    print("{:>6} {:>5} {:>10} {:>11} {:>9} {:>11} {:>7} {:>5}".format(*hdr))
    print("-" * 72)
    mult = {200: "0.25x", 100: "0.5x", 50: "1x", 25: "2x", 12: "4x"}
    base_mass = None
    rows = []
    for dt in RATES:
        s = rate_stats(args.preproc, dt, args.n_seq, args.frames_per_seq)
        if s is None:
            print(f"{dt:>6} {mult.get(dt,''):>5}   <not rendered>")
            continue
        rows.append((dt, s))
        print("{:>6} {:>5} {:>10.4f} {:>11.3f} {:>9.4f} {:>11.4f} {:>7.1f} {:>5}".format(
            dt, mult.get(dt, ""), s["occupancy"], s["sat_active"], s["sat_all"],
            s["mean_count"], s["p99_nonzero"], s["ceil"]))

    # interpretation: does total event mass scale ~linearly with window, or does clipping eat it?
    print("\nmass scaling vs 50 ms (linear would be 4.0 / 2.0 / 1.0 / 0.5 / 0.25):")
    m50 = next((s["mean_count"] for dt, s in rows if dt == 50), None)
    if m50:
        for dt, s in rows:
            print(f"  dt={dt:>3} : mean_count ratio to 50ms = {s['mean_count']/m50:5.2f}  (window ratio = {dt/50:4.2f})")
    print("\nread: high sat_active at dt=200/100 (vs ~0 at dt=50) => dense rates are clipped,")
    print("      and mean_count growing sub-linearly with the window confirms lost signal.")


if __name__ == "__main__":
    main()
