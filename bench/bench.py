"""mojo-xarray vs upstream xarray on the same data.

Run it through `pixi run bench` -- the pixi task takes a machine-wide flock, so
a concurrent factory job cannot distort the numbers.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time

import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import xarray as xr  # noqa: E402
from xarray.core import nputils as UP_NPUTILS  # noqa: E402

import mojo_xarray as mx  # noqa: E402


def timeit(fn, reps=7, warmup=2):
    """Best of `reps` after `warmup`, which is the standard robust estimator on a
    shared machine: the minimum is the run that was least disturbed."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def load_average() -> float:
    """1-minute load average, printed with the numbers so a table taken on a busy
    host can be recognised as one."""
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except (OSError, ValueError):
        return float("nan")


def machine():
    model = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": model or platform.processor(),
        "cores": os.cpu_count(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "xarray": xr.__version__,
        "load": load_average(),
    }


def check(got, ref, name):
    """Value parity, Dataset-aware: every data variable and coord is compared."""
    if isinstance(got, xr.Dataset) or isinstance(ref, xr.Dataset):
        if sorted(got.data_vars) != sorted(ref.data_vars):
            return f"VARS {sorted(got.data_vars)} vs {sorted(ref.data_vars)}"
        for key in got.data_vars:
            why = check(got[key], ref[key], f"{name}.{key}")
            if why:
                return why
        return ""
    g = np.asarray(getattr(got, "values", got), dtype=np.float64)
    r = np.asarray(getattr(ref, "values", ref), dtype=np.float64)
    if g.shape != r.shape:
        return f"SHAPE MISMATCH {g.shape} vs {r.shape}"
    if not np.allclose(g, r, rtol=1e-10, atol=1e-12, equal_nan=True):
        return "VALUE MISMATCH"
    return ""


def main() -> int:
    # upstream's own reductions warn on all-NaN and low-dof slices; the cases
    # below produce plenty of both and the warnings are not what is measured
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--rows", type=int, default=64)
    args = ap.parse_args()

    info = machine()
    print(f"mojo-xarray vs xarray {info['xarray']}")
    print(f"machine: {info['cpu']} ({info['cores']} cores), kernel {info['kernel']}")
    print(f"python {info['python']}, numpy {info['numpy']}")
    print(f"load average during the run: {info['load']:.1f} on {info['cores']} "
          f"cores; the host is shared, so this bounds how clean the numbers are")
    print()

    rng = np.random.default_rng(0)
    rows = args.rows

    def make(shape, nan_frac=0.05):
        a = rng.normal(size=shape)
        a[rng.random(shape) < nan_frac] = np.nan
        return np.ascontiguousarray(a, dtype=np.float64)

    cases = []

    # --- whole-array reductions over a tall column -------------------------
    col = make((2_000_000,))
    gcol, rcol = mx.DataArray(col, dims="t"), xr.DataArray(col, dims="t")
    for method in ("sum", "mean", "min", "max", "std", "var", "prod"):
        cases.append((
            f"reduce 2M f64  {method}",
            lambda m=method: getattr(gcol, m)(),
            lambda m=method: getattr(rcol, m)(),
        ))
    # the median sorts, so it gets a smaller column on both sides
    mcol = make((500_000,))
    gmcol, rmcol = mx.DataArray(mcol, dims="t"), xr.DataArray(mcol, dims="t")
    cases.append((
        "reduce 500K f64 median",
        lambda: gmcol.median(),
        lambda: rmcol.median(),
    ))

    # --- nanfirst / nanlast -------------------------------------------------
    # `nputils` computes both as an argmax over the notnull mask plus a gather,
    # so the input is a long run of leading NaNs: that is what makes the scan
    # the work rather than the first element.
    fcol = make((2_000_000,), nan_frac=0.05)
    fcol[:1_000_000] = np.nan
    for op in ("nanfirst", "nanlast"):
        cases.append((
            f"nputils 2M f64 {op}",
            lambda op=op: getattr(mx.nputils, op)(fcol, 0),
            lambda op=op: getattr(UP_NPUTILS, op)(fcol, 0),
        ))

    # --- reductions along an axis of a 2-D block ---------------------------
    blk = make((rows, 1024))
    gblk, rblk = (mx.DataArray(blk, dims=("r", "c")),
                  xr.DataArray(blk, dims=("r", "c")))
    for method in ("sum", "mean", "min", "max", "std", "var", "median"):
        cases.append((
            f"reduce {rows}x1024 axis=1 {method}",
            lambda m=method: getattr(gblk, m)(dim="c"),
            lambda m=method: getattr(rblk, m)(dim="c"),
        ))

    # --- 1-D rolling along the last axis -----------------------------------
    roll = make((rows, 4096))
    groll, rroll = (mx.DataArray(roll, dims=("r", "t")),
                    xr.DataArray(roll, dims=("r", "t")))
    for method in ("sum", "mean", "min", "max", "std", "var", "median", "count"):
        cases.append((
            f"rolling w=64  {method}",
            lambda m=method: getattr(groll.rolling(t=64, min_periods=1), m)(),
            lambda m=method: getattr(rroll.rolling(t=64, min_periods=1), m)(),
        ))
    for w in (8, 256, 512):
        cases.append((
            f"rolling w={w:<5d} mean",
            lambda w=w: groll.rolling(t=w, min_periods=1).mean(),
            lambda w=w: rroll.rolling(t=w, min_periods=1).mean(),
        ))
    # the rolling median sorts each window, so it gets its own smaller input
    rmed = make((16, 2048))
    grmed = mx.DataArray(rmed, dims=("r", "t"))
    rrmed = xr.DataArray(rmed.copy(), dims=("r", "t"))
    cases.append((
        "rolling 16x2048 w=32 median",
        lambda: grmed.rolling(t=32, min_periods=1).median(),
        lambda: rrmed.rolling(t=32, min_periods=1).median(),
    ))

    # --- rolling along a non-last axis (the moveaxis path) -----------------
    cases.append((
        f"rolling axis=0 w=64 mean",
        lambda: gblk.rolling(r=64, min_periods=1).mean(),
        lambda: rblk.rolling(r=64, min_periods=1).mean(),
    ))

    # --- construct (pad + sliding_window_view) -----------------------------
    win = make((rows, 1024))
    gwin, rwin = mx.DataArray(win, dims=("r", "t")), xr.DataArray(win, dims=("r", "t"))
    cases.append((
        "rolling construct w=64",
        lambda: gwin.rolling(t=64).construct("w"),
        lambda: rwin.rolling(t=64).construct("w"),
    ))

    # --- coarsen -----------------------------------------------------------
    cases.append((
        f"coarsen 2 mean",
        lambda: gblk.coarsen(c=2).mean(),
        lambda: rblk.coarsen(c=2).mean(),
    ))
    cases.append((
        f"coarsen 64 std",
        lambda: gblk.coarsen(c=64).std(),
        lambda: rblk.coarsen(c=64).std(),
    ))

    # --- quantile: xarray's own quantile() is a Dataset-level ufunc with no
    # --- direct ndarray counterpart, so this compares the kernel against the
    # --- NumPy function xarray itself delegates to.
    small = make((128, 1024))
    for q in (0.1, 0.5, 0.9):
        cases.append((
            f"nanquantile q={q} axis=1 (vs np)",
            lambda q=q: mx.nanquantile(small, q, axis=1),
            lambda q=q: np.nanquantile(small, q, axis=1),
        ))

    # --- Dataset: every variable is reduced, so this is the per-call overhead
    # --- times the number of variables plus the reductions themselves.
    gds = mx.Dataset({f"v{i}": (("r", "t"), blk) for i in range(4)},
                     coords={"r": np.arange(rows), "t": np.arange(blk.shape[1])})
    rds = xr.Dataset({f"v{i}": (("r", "t"), blk.copy()) for i in range(4)},
                     coords={"r": np.arange(rows), "t": np.arange(blk.shape[1])})
    for method in ("sum", "mean", "min", "max", "std", "var", "median", "argmin"):
        cases.append((
            f"Dataset 4x{rows}x1024 {method}",
            lambda m=method: getattr(gds, m)(dim="t"),
            lambda m=method: getattr(rds, m)(dim="t"),
        ))
    cases.append((
        f"Dataset 4x{rows}x1024 rolling w=64 sum",
        lambda: gds.rolling(t=64, min_periods=1).sum(),
        lambda: rds.rolling(t=64, min_periods=1).sum(),
    ))

    header = (f"| {'case':<34} | {'mojo':>10} | {'xarray':>10} | {'speedup':>8} | ok |")
    print(header)
    print("|" + "-" * 36 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 10
          + "|-----|")
    for name, mine, theirs in cases:
        t_mine, out_mine = timeit(mine, reps=args.reps)
        t_ref, out_ref = timeit(theirs, reps=args.reps)
        status = check(out_mine, out_ref, name) or "yes"
        speed = t_ref / t_mine if t_mine else float("inf")
        print(f"| {name:<34} | {t_mine * 1e3:8.2f} ms | {t_ref * 1e3:8.2f} ms "
              f"| {speed:7.2f}x | {status} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
