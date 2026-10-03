"""Split each benchmark case into Python overhead vs. kernel time.

Not part of `pixi run bench`: it prints a breakdown, not a table, so the two
never disagree.  Run it under the same flock:

    flock /tmp/mojo-bench.lock python bench/profile_split.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import xarray as xr  # noqa: E402

import mojo_xarray as mx  # noqa: E402
from mojo_xarray import _ops  # noqa: E402
from mojo_xarray._lib import lib  # noqa: E402


def timeit(fn, reps=7, warmup=2):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def main() -> int:
    rng = np.random.default_rng(0)
    rows = 64

    def make(shape, nan_frac=0.05):
        a = rng.normal(size=shape)
        a[rng.random(shape) < nan_frac] = np.nan
        return np.ascontiguousarray(a, dtype=np.float64)

    cases = []

    col = make((2_000_000,))
    gcol = mx.DataArray(col, dims="t")
    for method in ("sum", "mean", "min", "max", "std", "var", "prod"):
        cases.append((
            f"2M {method}",
            lambda m=method: getattr(gcol, m)(),
            lambda m=method: _ops.reduce_axis(col, f"mx_nan{m}", None,
                                              ddof=0 if m in ("std", "var") else None),
        ))

    blk = make((rows, 1024))
    gblk = mx.DataArray(blk, dims=("r", "c"))
    for method in ("sum", "mean", "min", "max", "std", "var", "median"):
        cases.append((
            f"64x1024 ax1 {method}",
            lambda m=method: getattr(gblk, m)(dim="c"),
            lambda m=method: _ops.reduce_axis(blk, f"mx_nan{m}", 1,
                                              ddof=0 if m in ("std", "var") else None),
        ))

    roll = make((rows, 4096))
    groll = mx.DataArray(roll, dims=("r", "t"))
    for method in ("sum", "mean", "min", "max", "std", "var", "median", "count"):
        w = 64
        if method == "median":
            mroll = make((16, 2048))
            gmr = mx.DataArray(mroll, dims=("r", "t"))
            cases.append(("roll 16x2048 median", lambda: gmr.rolling(t=32, min_periods=1).median(),
                          lambda: _ops.move(mroll, "mx_move_median", 32, 1, False, n_rows=16)))
            continue
        cases.append((
            f"roll w={w} {method}",
            lambda m=method, w=w: getattr(groll.rolling(t=w, min_periods=1), m)(),
            lambda m=method, w=w: _ops.move(roll, f"mx_move_{m}", w, 1, False, n_rows=rows),
        ))

    gds = mx.Dataset({f"v{i}": (("r", "t"), blk) for i in range(4)},
                     coords={"r": np.arange(rows), "t": np.arange(blk.shape[1])})
    for method in ("sum", "mean", "min", "max", "std", "var", "median", "argmin"):
        cases.append((
            f"DS 4var {method}",
            lambda m=method: getattr(gds, m)(dim="t"),
            lambda m=method: [_ops.reduce_axis(blk, f"mx_nan{m}", 1,
                                               ddof=0 if m in ("std", "var") else None)
                              for _ in range(4)],
        ))
    cases.append((
        "DS 4var roll sum",
        lambda: gds.rolling(t=64, min_periods=1).sum(),
        lambda: [_ops.move(blk, "mx_move_sum", 64, 1, False, n_rows=rows) for _ in range(4)],
    ))

    gwin = mx.DataArray(blk, dims=("r", "t"))
    cases.append(("roll construct", lambda: gwin.rolling(t=64).construct("w"),
                  lambda: np.lib.stride_tricks.sliding_window_view(
                      np.pad(blk, ((0, 0), (63, 0))), 64, axis=-1)))

    cases.append(("coarsen 2 mean", lambda: gblk.coarsen(c=2).mean(), None))
    cases.append(("coarsen 64 std", lambda: gblk.coarsen(c=64).std(), None))

    small = make((128, 1024))
    for q in (0.1, 0.5, 0.9):
        cases.append((f"nanquantile q={q}",
                      lambda q=q: mx.nanquantile(small, q, axis=1),
                      lambda q=q: _ops.reduce_axis(small, "mx_nanquantile", 1, q=q)))

    mcol = make((500_000,))
    gmcol = mx.DataArray(mcol, dims="t")
    cases.append(("500K median", lambda: gmcol.median(),
                  lambda: _ops.reduce_axis(mcol, "mx_nanmedian", None)))

    fcol = make((2_000_000,), nan_frac=0.05)
    fcol[:1_000_000] = np.nan
    cases.append(("nanfirst", lambda: mx.nputils.nanfirst(fcol, 0), None))
    cases.append(("nanlast", lambda: mx.nputils.nanlast(fcol, 0), None))

    header = f"| {'case':<22} | {'full':>9} | {'kernel':>9} | {'py':>9} | {'py%':>5} |"
    print(header)
    print("|" + "-" * 24 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 7 + "|")
    for name, full, kern in cases:
        t_full, _ = timeit(full)
        if kern is None:
            print(f"| {name:<22} | {t_full * 1e3:7.3f} ms | {'-':>9} | {'-':>9} | {'-':>5} |")
            continue
        t_k, _ = timeit(kern)
        py = t_full - t_k
        pct = 100.0 * py / t_full if t_full else 0.0
        print(f"| {name:<22} | {t_full * 1e3:7.3f} ms | {t_k * 1e3:7.3f} ms | "
              f"{py * 1e3:7.3f} ms | {pct:4.1f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
