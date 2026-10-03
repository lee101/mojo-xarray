import os

import numpy as np
import pytest
import xarray as xr

import mojo_xarray as mx
from mojo_xarray import _lib

xarray_version = tuple(int(p) for p in xr.__version__.split(".")[:3] if p.isdigit())

# Every test in this suite asserts something about the *compiled* library, so a
# missing `capi.so` has to stop the run rather than let a pure-Python fallback
# pass it.  The load happens once, here, at collection time.
_loaded = _lib.lib()
KERNELS = tuple(sorted(name for name in _lib._SIGNATURES))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def assert_same(got, ref, rtol=1e-12, atol=0.0):
    """Numerical parity with upstream, NaN-for-NaN."""
    g = np.asarray(got.values if hasattr(got, "values") else got, dtype=np.float64)
    r = np.asarray(ref.values if hasattr(ref, "values") else ref, dtype=np.float64)
    if g.shape != r.shape:
        raise AssertionError(f"shape {g.shape} != {r.shape}")
    np.testing.assert_allclose(g, r, rtol=rtol, atol=atol, equal_nan=True)


def array(data, dims=("x",), **kwargs):
    """A `mojo_xarray.DataArray` and its plain `xarray` twin."""
    data = np.ascontiguousarray(data, dtype=np.float64)
    ref = xr.DataArray(data.copy(), dims=dims, **kwargs)
    return mx.DataArray(data.copy(), dims=dims, **kwargs), ref


def library_path() -> str:
    """The shared library this run is exercising."""
    return os.path.realpath(_lib.LIB)
