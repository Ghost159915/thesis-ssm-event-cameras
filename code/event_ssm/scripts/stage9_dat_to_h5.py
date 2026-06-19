#!/usr/bin/env python
"""Stage 9 - convert raw Prophesee Gen1 ``.dat`` event files to the RVT ``.dat.h5`` format.

Why this exists
---------------
The Stage-9 temporal-generalisation study re-renders the Gen1 *test* split at several
event-accumulation windows (dt = 100 / 50 / 25 ms, see the Stage-9 doc).  Re-rendering
needs the **raw events**, which the Prophesee original download ships as packed binary
``<name>_td.dat`` files.  The RVT pre-processor however
(``external/.../RVT/scripts/genx/preprocess_dataset.py``, class ``H5Reader``) reads each
recording's events from an HDF5 file named ``<name>_td.dat.h5`` containing a group
``events`` with the 1-D datasets ``x, y, p, t`` (``t`` in microseconds) plus the scalar
``height`` / ``width``.  This script bridges the two: ``.dat`` -> ``.dat.h5``.

It deliberately re-uses the repo's own ``PSEELoader`` to decode the ``.dat`` bit-packing
(x = bits 0-13, y = bits 14-27, p = bit 28) so the decoding is identical to the baseline.

Output schema (consumed unmodified by ``H5Reader``)::

    /events/x   uint16   (N,)   pixel column   [0, 303]
    /events/y   uint16   (N,)   pixel row       [0, 239]
    /events/p   uint8    (N,)   polarity        {0, 1}
    /events/t   int64    (N,)   timestamp (us), non-decreasing-ish (H5Reader re-sorts)
    /events/height  int64 scalar (240)
    /events/width   int64 scalar (304)

Compression: blosc-lz4 + byte-shuffle via ``hdf5plugin`` (fast; monotonic timestamps
compress extremely well).  The pre-processor already ``import hdf5plugin`` so the files
read back transparently.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the blosc filter for read+write)
import numpy as np

# --- make the repo's Prophesee .dat reader importable ------------------------------
# .../code/event_ssm/scripts/stage9_dat_to_h5.py -> parents[3] == repo root
REPO = Path(__file__).resolve().parents[3]
RVT = REPO / "external" / "ssms_event_cameras" / "RVT"
sys.path.insert(0, str(RVT))
# NOTE: we use the lower-level ``load_td_data`` rather than ``PSEELoader``: under NumPy
# 2.x (NEP-50) the loader's ``(_end - _start) // _ev_size`` overflows because ``_ev_size``
# is a numpy uint8. ``load_td_data`` decodes the identical bit-packing (x=bits0-13,
# y=bits14-27, p=bit28) without that code path, and we do not touch the vendored file.
from utils.evaluation.prophesee.io import dat_events_tools as dat  # noqa: E402

GEN1_H, GEN1_W = 240, 304
CHUNK = (1_000_000,)  # events per HDF5 chunk -> efficient windowed slicing later
_BLOSC = hdf5plugin.Blosc(cname="lz4", clevel=1, shuffle=hdf5plugin.Blosc.SHUFFLE)


def convert_one(dat_path: Path, out_path: Path, height: int = GEN1_H, width: int = GEN1_W) -> int:
    """Convert a single ``.dat`` to ``.dat.h5``. Returns the number of events written."""
    ev = dat.load_td_data(str(dat_path))  # reads all events -> structured array t, x, y, p
    n = len(ev)
    if n <= 0:
        raise ValueError(f"{dat_path.name}: 0 events")

    x = ev["x"].astype(np.uint16)
    y = ev["y"].astype(np.uint16)
    p = ev["p"].astype(np.uint8)
    t = ev["t"].astype(np.int64)

    # sanity (warn, do not abort - a handful of off-grid events get clipped downstream)
    if x.max() >= width or y.max() >= height:
        print(f"  [warn] {dat_path.name}: coords out of {width}x{height} "
              f"(x.max={x.max()}, y.max={y.max()})", file=sys.stderr)
    n_nonmono = int((np.diff(t) < 0).sum())
    if n_nonmono:
        print(f"  [warn] {dat_path.name}: {n_nonmono} non-monotonic timestamps "
              f"(H5Reader re-sorts these)", file=sys.stderr)

    tmp = out_path.with_suffix(out_path.suffix + ".inprogress")
    with h5py.File(str(tmp), "w") as f:
        g = f.create_group("events")
        chunks = CHUNK if n >= CHUNK[0] else (n,)
        for name, arr in (("x", x), ("y", y), ("p", p), ("t", t)):
            g.create_dataset(name, data=arr, chunks=chunks, **_BLOSC)
        g.create_dataset("height", data=np.int64(height))
        g.create_dataset("width", data=np.int64(width))
    tmp.rename(out_path)  # atomic: a partial crash never leaves a "complete" .dat.h5
    return n


def verify_one(out_path: Path, expected_n: int) -> None:
    """Read the file back the same way the RVT ``H5Reader`` does (group ``events`` ->
    ``t`` loaded whole + an indexed ``x`` slice) so we know the pre-processor will accept it.
    Kept dependency-light (plain h5py) to avoid importing torch/numba just to verify."""
    with h5py.File(str(out_path), "r") as f:
        g = f["events"]
        assert int(g["height"][()]) == GEN1_H and int(g["width"][()]) == GEN1_W
        t = np.asarray(g["t"])  # full read -> proves blosc decompresses (hdf5plugin)
        assert len(t) == expected_n, f"{out_path.name}: read back {len(t)} != {expected_n}"
        k = min(1000, expected_n)
        assert g["x"][:k].max() < GEN1_W and g["y"][:k].max() < GEN1_H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="dir containing *_td.dat files (e.g. .../test)")
    ap.add_argument("--delete-dat", action="store_true", help="delete each .dat after a verified .dat.h5 (saves disk)")
    ap.add_argument("--no-verify", action="store_true", help="skip the H5Reader read-back check")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files (0 = all; for testing)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    assert in_dir.is_dir(), f"not a dir: {in_dir}"
    dats = sorted(in_dir.glob("*_td.dat"))
    if args.limit:
        dats = dats[: args.limit]
    assert dats, f"no *_td.dat files in {in_dir}"

    total = len(dats)
    done = skipped = 0
    ev_total = 0
    t0 = time.time()
    print(f"[stage9] converting {total} .dat -> .dat.h5 in {in_dir}", flush=True)
    for i, dat in enumerate(dats, 1):
        out = dat.with_suffix(".dat.h5")  # 17-..._td.dat -> 17-..._td.dat.h5
        if out.exists():
            skipped += 1
            print(f"[{i}/{total}] skip (exists) {out.name}", flush=True)
            if args.delete_dat and dat.exists():
                dat.unlink()
            continue
        ts = time.time()
        n = convert_one(dat, out)
        if not args.no_verify:
            verify_one(out, n)
        if args.delete_dat:
            dat.unlink()
        done += 1
        ev_total += n
        dt = time.time() - ts
        elapsed = time.time() - t0
        eta = elapsed / i * (total - i)
        print(f"[{i}/{total}] {out.name}  {n/1e6:.1f}M ev  "
              f"{out.stat().st_size/1e6:.0f}MB  {dt:.1f}s  "
              f"| done={done} skip={skipped}  ETA {eta/60:.0f}min", flush=True)

    print(f"[stage9] finished: {done} converted, {skipped} skipped, "
          f"{ev_total/1e9:.2f}B events, {(time.time()-t0)/60:.1f} min total", flush=True)


if __name__ == "__main__":
    main()
