"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")

# `MOJO_XARRAY_LIB` points at an already-built `capi.so`. Two cases need it:
# an ordinary (non-editable) `pip install`, where the package is copied into
# site-packages and `ROOT` no longer names the checkout that holds `src/` and
# `build/`; and any server that must never pay a `mojo build` inside a request
# and would rather ship the kernel next to the deploy. Set it and nothing is
# ever compiled at import or call time.
LIB = os.environ.get("MOJO_XARRAY_LIB") or os.path.join(ROOT, "build", "capi.so")

I = ctypes.c_int64
F = ctypes.c_double
P = ctypes.c_void_p

_REDUCE = ([I, I, I, I, I, I], None)
_REDUCE_DDOF = ([I, I, I, I, I, I, I], None)
_MOVE = ([I, I, I, I, I, I, I, I], None)
_MOVE_DDOF = ([I, I, I, I, I, I, I, I, I], None)

# name -> (argtypes, restype)
_SIGNATURES = {
    "mx_nanmin": _REDUCE,
    "mx_nanmax": _REDUCE,
    "mx_nansum": _REDUCE,
    "mx_nanmean": _REDUCE,
    "mx_nanmedian": _REDUCE,
    "mx_nanprod": _REDUCE,
    "mx_nanargmin": _REDUCE,
    "mx_nanargmax": _REDUCE,
    "mx_nan_count": _REDUCE,
    "mx_nanvar": _REDUCE_DDOF,
    "mx_nanstd": _REDUCE_DDOF,
    "mx_nanquantile": ([I, I, I, I, I, I, F], None),
    "mx_nanfirst_index": ([I, I, I, I, I], None),
    "mx_maybe_null_out": ([I, I, I, I, I], None),
    "mx_nancumsum": ([I, I, I, I, I], None),
    "mx_nancumprod": ([I, I, I, I, I], None),
    "mx_move_sum": _MOVE,
    "mx_move_mean": _MOVE,
    "mx_move_min": _MOVE,
    "mx_move_max": _MOVE,
    "mx_move_median": _MOVE,
    "mx_move_argmin": _MOVE,
    "mx_move_argmax": _MOVE,
    "mx_move_sum_of_squares": _MOVE,
    "mx_move_count": ([I, I, I, I, I, I, I], None),
    "mx_move_var": _MOVE_DDOF,
    "mx_move_std": _MOVE_DDOF,
    "mx_pad": ([I, I, I, P, P, P, P, P, I, F], None),
    "mx_par_workers": ([], I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_XARRAY_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_XARRAY_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `build/capi.so` if it is missing or stale."""
    if os.environ.get("MOJO_XARRAY_LIB") and os.path.exists(LIB) and not force:
        # An explicitly supplied library is taken as given: the sources it was
        # built from need not be present at all, so there is nothing to
        # compare mtimes against.
        return LIB
    if not os.path.isdir(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            f"no Mojo sources at {SRC} and no library at {LIB}; this looks like "
            "a copied install - point MOJO_XARRAY_LIB at a built capi.so"
        )
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def main() -> int:
    """`python -m mojo_xarray._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
