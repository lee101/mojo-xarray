"""Parity of `mojo_xarray.rolling` with `xarray.computation.rolling`.

Every case is a `mojo_xarray.DataArray` next to the identical plain
`xarray.DataArray`, reduced the same way.
"""

import numpy as np
import pytest
import xarray as xr

import mojo_xarray as mx
from mojo_xarray.rolling import DataArrayCoarsen, DataArrayRolling, DatasetRolling

from .conftest import assert_same

REDUCE_METHODS = ["sum", "mean", "min", "max", "std", "var", "median",
                  "argmin", "argmax"]


def _pair(data, dims=("x",), **kwargs):
    data = np.ascontiguousarray(data, dtype=np.float64)
    return (mx.DataArray(data.copy(), dims=dims, **kwargs),
            xr.DataArray(data.copy(), dims=dims, **kwargs))


def _data(rng, n, nan_frac=0.2):
    a = rng.normal(size=n)
    a[rng.random(n) < nan_frac] = np.nan
    return a


# --- the public surface ----------------------------------------------------


def test_rolling_classes_mirror_upstream():
    from xarray.computation import rolling as up
    assert isinstance(mx.DataArrayRolling, type)
    for name in ("sum", "mean", "min", "max", "std", "var", "median", "prod",
                 "argmin", "argmax", "count", "construct", "reduce", "__iter__",
                 "__len__", "ndim"):
        assert hasattr(DataArrayRolling, name), name
        assert hasattr(up.DataArrayRolling, name), name
    for name in ("sum", "mean", "min", "max", "std", "var", "median", "prod",
                 "count", "construct", "reduce", "all", "any"):
        assert hasattr(DataArrayCoarsen, name), name
        assert hasattr(up.DataArrayCoarsen, name), name


def test_rolling_rejects_a_non_positive_window():
    da = mx.DataArray(np.arange(5.0), dims="x")
    with pytest.raises(ValueError, match="window must be > 0"):
        da.rolling(x=0)
    with pytest.raises(ValueError, match="min_periods must be greater"):
        da.rolling(x=2, min_periods=0)


def test_rolling_rejects_a_missing_dim():
    da = mx.DataArray(np.arange(5.0), dims="x")
    with pytest.raises(KeyError, match="not found"):
        da.rolling(y=2)


def test_repr_and_len_match_upstream():
    g, r = _pair(np.arange(10.0))
    assert repr(g.rolling(x=3)) == repr(r.rolling(x=3))
    assert repr(g.rolling(x=3, center=True)) == repr(r.rolling(x=3, center=True))
    assert len(g.rolling(x=3)) == len(r.rolling(x=3))
    assert g.rolling(x=3).ndim == r.rolling(x=3).ndim


def test_iter_yields_the_same_windows(rng):
    g, r = _pair(_data(rng, 12))
    got = [w.values for _, w in g.rolling(x=4, min_periods=1)]
    ref = [w.values for _, w in r.rolling(x=4, min_periods=1)]
    assert len(got) == len(ref)
    for a, b in zip(got, ref, strict=True):
        np.testing.assert_allclose(a, b, equal_nan=True)


# --- the reductions --------------------------------------------------------


@pytest.mark.parametrize("method", REDUCE_METHODS)
@pytest.mark.parametrize("window", [1, 2, 3, 5, 8, 20])
@pytest.mark.parametrize("center", [False, True])
def test_rolling_reduction_matches_upstream(rng, method, window, center):
    a = _data(rng, 24)
    g, r = _pair(a)
    got = getattr(g.rolling(x=window, center=center), method)()
    ref = getattr(r.rolling(x=window, center=center), method)()
    assert_same(got, ref, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("min_periods", [1, 2, 3, 7])
@pytest.mark.parametrize("method", ["sum", "mean", "min", "max", "std", "var",
                                    "median", "count"])
def test_rolling_min_periods_matches_upstream(rng, method, min_periods):
    a = _data(rng, 20, nan_frac=0.35)
    g, r = _pair(a)
    got = getattr(g.rolling(x=4, min_periods=min_periods), method)()
    ref = getattr(r.rolling(x=4, min_periods=min_periods), method)()
    assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_rolling_min_periods_zero_is_rejected_upstream_too(rng):
    a = _data(rng, 12, nan_frac=0.4)
    g, r = _pair(a)
    for obj in (g, r):
        with pytest.raises(ValueError, match="min_periods must be greater"):
            obj.rolling(x=3, min_periods=0)


@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_std_var_ddof_matches_upstream(rng, ddof):
    a = _data(rng, 16)
    g, r = _pair(a)
    for method in ("std", "var"):
        got = getattr(g.rolling(x=4, min_periods=1), method)(ddof=ddof)
        ref = getattr(r.rolling(x=4, min_periods=1), method)(ddof=ddof)
        assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_rolling_along_a_non_last_axis_matches_upstream(rng):
    a = _data(rng, 12).reshape(3, 4)
    g, r = _pair(a, dims=("a", "b"))
    for method in REDUCE_METHODS:
        got = getattr(g.rolling(a=2, min_periods=1), method)()
        ref = getattr(r.rolling(a=2, min_periods=1), method)()
        assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_rolling_keeps_coords_name_and_attrs(rng):
    a = _data(rng, 10)
    coords = {"x": np.arange(10) * 2}
    g, r = _pair(a, dims=("x",), coords=coords, name="v", attrs={"units": "m"})
    for obj in (g.rolling(x=3).sum(), r.rolling(x=3).sum()):
        assert obj.name == "v"
        assert obj.attrs == {"units": "m"}
        np.testing.assert_array_equal(obj.coords["x"].values, coords["x"])
    assert_same(g.rolling(x=3).sum(), r.rolling(x=3).sum())


def test_rolling_keep_attrs_false_drops_attrs(rng):
    a = _data(rng, 10)
    g, r = _pair(a, dims=("x",), attrs={"units": "m"})
    assert g.rolling(x=3).sum(keep_attrs=False).attrs == {}
    assert r.rolling(x=3).sum(keep_attrs=False).attrs == {}


def test_rolling_prod_matches_upstream(rng):
    a = _data(rng, 10, nan_frac=0.1)
    g, r = _pair(a)
    assert_same(g.rolling(x=3, min_periods=1).prod(), r.rolling(x=3, min_periods=1).prod())


def test_rolling_reduce_with_a_python_callable_matches_upstream(rng):
    a = _data(rng, 10, nan_frac=0.1)
    g, r = _pair(a)
    for func in (np.nansum, np.nanmean, np.nanmax):
        got = g.rolling(x=3, min_periods=1).reduce(func)
        ref = r.rolling(x=3, min_periods=1).reduce(func)
        assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_rolling_construct_matches_upstream(rng):
    a = _data(rng, 12)
    for center in (False, True):
        g, r = _pair(a)
        got = g.rolling(x=4, center=center).construct("w")
        ref = r.rolling(x=4, center=center).construct("w")
        assert got.dims == ref.dims
        assert_same(got, ref)


def test_rolling_construct_with_stride_and_fill_value(rng):
    a = _data(rng, 12)
    g, r = _pair(a)
    got = g.rolling(x=3).construct("w", stride=2)
    ref = r.rolling(x=3).construct("w", stride=2)
    assert_same(got, ref)
    got = g.rolling(x=3, min_periods=1).reduce(np.nansum, fillna=0.0)
    ref = r.rolling(x=3, min_periods=1).reduce(np.nansum, fillna=0.0)
    assert_same(got, ref)


def test_rolling_construct_nan_pads_the_leading_window(rng):
    a = np.arange(8.0)
    g, r = _pair(a)
    got = g.rolling(x=3).construct("w").values
    ref = r.rolling(x=3).construct("w").values
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_allclose(got[~np.isnan(got)], ref[~np.isnan(ref)])


def test_nd_rolling_matches_upstream(rng):
    a = _data(rng, 24).reshape(4, 6)
    g, r = _pair(a, dims=("a", "b"))
    for method in ["sum", "mean", "max", "median", "count"]:
        got = getattr(g.rolling(a=2, b=3), method)()
        ref = getattr(r.rolling(a=2, b=3), method)()
        assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_rolling_on_an_all_nan_array_matches_upstream():
    a = np.full(10, np.nan)
    g, r = _pair(a)
    for method in REDUCE_METHODS + ["count"]:
        for window, min_periods in [(3, 3), (3, 1), (2, 1)]:
            got = getattr(g.rolling(x=window, min_periods=min_periods), method)()
            ref = getattr(r.rolling(x=window, min_periods=min_periods), method)()
            assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_rolling_on_integer_data_falls_back(rng):
    a = rng.integers(0, 20, size=12)
    g = mx.DataArray(a, dims="x")
    r = xr.DataArray(a.copy(), dims="x")
    assert_same(g.rolling(x=3).sum(), r.rolling(x=3).sum())
    assert_same(g.rolling(x=3).mean(), r.rolling(x=3).mean())


def test_rolling_with_skipna_false_falls_back(rng):
    a = _data(rng, 10)
    g, r = _pair(a)
    assert_same(g.rolling(x=3, min_periods=1).min(skipna=False),
                r.rolling(x=3, min_periods=1).min(skipna=False))


def test_rolling_dim_kwarg_warns_like_upstream(rng):
    a = _data(rng, 10)
    g, r = _pair(a)
    with pytest.warns(FutureWarning, match="has no effect"):
        got = g.rolling(x=3).sum(dim="x")
    with pytest.warns(FutureWarning, match="has no effect"):
        ref = r.rolling(x=3).sum(dim="x")
    assert_same(got, ref)


# --- Dataset ---------------------------------------------------------------


def test_dataset_rolling_matches_upstream(rng):
    a = _data(rng, 12)
    b = _data(rng, 12)
    ds_g = mx.Dataset({"a": ("x", a), "b": ("x", b)}, coords={"x": np.arange(12)})
    ds_r = xr.Dataset({"a": ("x", a.copy()), "b": ("x", b.copy())},
                      coords={"x": np.arange(12)})
    assert isinstance(ds_g.rolling(x=3), DatasetRolling)
    for method in ["sum", "mean", "max", "count"]:
        got = getattr(ds_g.rolling(x=3), method)()
        ref = getattr(ds_r.rolling(x=3), method)()
        for k in ("a", "b"):
            assert_same(got[k], ref[k], rtol=1e-11, atol=1e-13)


def test_dataset_rolling_construct_matches_upstream(rng):
    a = _data(rng, 8)
    ds_g = mx.Dataset({"a": ("x", a)})
    ds_r = xr.Dataset({"a": ("x", a.copy())})
    got = ds_g.rolling(x=3).construct("w")
    ref = ds_r.rolling(x=3).construct("w")
    assert got["a"].dims == ref["a"].dims
    assert_same(got["a"], ref["a"])


# --- Coarsen ---------------------------------------------------------------


def _pair2d(a, **kwargs):
    return (mx.DataArray(a.copy(), dims=("a", "b"), **kwargs),
            xr.DataArray(a.copy(), dims=("a", "b"), **kwargs))


COARSEN_METHODS = ["sum", "mean", "min", "max", "std", "var", "median", "count"]


@pytest.mark.parametrize("method", COARSEN_METHODS)
@pytest.mark.parametrize("boundary,side", [("exact", "left"), ("exact", "right"),
                                            ("trim", "left"), ("trim", "right"),
                                            ("pad", "left"), ("pad", "right")])
@pytest.mark.parametrize("window", [2, 4])
def test_coarsen_matches_upstream(rng, method, boundary, side, window):  # noqa: D103
    a = _data(rng, 24).reshape(4, 6)
    if window == 4 and boundary == "exact":
        return  # 6 is not a multiple of 4; covered by the raise test below
    g, r = _pair2d(a)
    got = getattr(g.coarsen(b=window, boundary=boundary, side=side), method)()
    ref = getattr(r.coarsen(b=window, boundary=boundary, side=side), method)()
    assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_coarsen_exact_boundary_raises_like_upstream(rng):
    a = _data(rng, 24).reshape(4, 6)
    g, r = _pair2d(a)
    with pytest.raises(ValueError, match="Could not coarsen"):
        g.coarsen(b=4).sum()
    with pytest.raises(ValueError, match="Could not coarsen"):
        r.coarsen(b=4).sum()


def test_coarsen_construct_matches_upstream(rng):
    a = _data(rng, 24).reshape(4, 6)
    g, r = _pair2d(a)
    got = g.coarsen(b=2).construct(b=("block", "size"))
    ref = r.coarsen(b=2).construct(b=("block", "size"))
    assert got.dims == ref.dims
    assert_same(got, ref)


def test_coarsen_reduce_with_a_callable_matches_upstream(rng):
    a = _data(rng, 24).reshape(4, 6)
    g, r = _pair2d(a)
    for func in (np.nansum, np.nanmean, np.nanmax):
        assert_same(g.coarsen(b=2).reduce(func), r.coarsen(b=2).reduce(func),
                    rtol=1e-11, atol=1e-13)


def test_coarsen_preserves_and_reduces_coords(rng):
    a = _data(rng, 24).reshape(4, 6)
    coords = {"a": ("a", np.arange(4.0) * 2), "b": ("b", np.arange(6.0))}
    g, r = _pair2d(a, coords=coords)
    got = g.coarsen(b=2).mean()
    ref = r.coarsen(b=2).mean()
    np.testing.assert_allclose(got.coords["b"].values, ref.coords["b"].values)
    np.testing.assert_allclose(got.coords["a"].values, ref.coords["a"].values)
    assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_dataset_coarsen_matches_upstream(rng):
    a = _data(rng, 12)
    ds_g = mx.Dataset({"a": ("x", a)})
    ds_r = xr.Dataset({"a": ("x", a.copy())})
    got = ds_g.coarsen(x=3).mean()
    ref = ds_r.coarsen(x=3).mean()
    assert_same(got["a"], ref["a"], rtol=1e-11, atol=1e-13)


def test_parallel_rolling_median_matches_upstream(rng):
    """Above the parallel threshold the rolling median stripes across threads
    and sorts into a per-stripe scratch; this is the guard for that buffer."""
    a = _data(rng, (16, 20_000))
    g, r = _pair(a, dims=("row", "t"))
    for w in (8, 64, 257):
        got = g.rolling(t=w, min_periods=1).median()
        ref = r.rolling(t=w, min_periods=1).median()
        assert_same(got, ref, rtol=0)


def test_parallel_rolling_reductions_match_upstream(rng):
    a = _data(rng, (8, 40_000))
    g, r = _pair(a, dims=("row", "t"))
    for method in ("sum", "mean", "min", "max", "std", "var", "count"):
        got = getattr(g.rolling(t=32, min_periods=1), method)()
        ref = getattr(r.rolling(t=32, min_periods=1), method)()
        assert_same(got, ref, rtol=1e-10, atol=1e-12)


def test_parallel_rolling_count_matches_upstream(rng):
    a = _data(rng, (4, 60_000), nan_frac=0.5)
    g, r = _pair(a, dims=("row", "t"))
    assert_same(g.rolling(t=64, min_periods=8).count(),
                r.rolling(t=64, min_periods=8).count(), rtol=1e-10, atol=1e-12)


# -- `construct` outside the float64 fast path -------------------------------
#
# `_rolling_window` pads through the Mojo kernel only for a C-contiguous
# float64 NumPy array.  Everything else has to reach upstream's own
# `Variable.rolling_window`, or the result silently changes dtype and a lazy
# array is silently computed.

@pytest.mark.parametrize("dtype", ["float32", "int32", "int64", "bool"])
def test_construct_preserves_upstream_dtype(dtype):
    a = (np.arange(24) % 3).astype(dtype).reshape(4, 6)
    got = mx.DataArray(a, dims=("a", "b")).rolling(b=3, center=True).construct("w")
    ref = xr.DataArray(a, dims=("a", "b")).rolling(b=3, center=True).construct("w")
    assert got.dtype == ref.dtype, f"{got.dtype} != {ref.dtype}"
    np.testing.assert_allclose(
        np.asarray(got.values, dtype=np.float64),
        np.asarray(ref.values, dtype=np.float64),
        equal_nan=True,
    )


def test_construct_keeps_dask_lazy():
    dask = pytest.importorskip("dask.array")
    a = dask.arange(64, chunks=8, dtype="f8")
    got = mx.DataArray(a, dims="x").rolling(x=4, min_periods=1).construct("w")
    ref = xr.DataArray(a, dims="x").rolling(x=4, min_periods=1).construct("w")
    assert type(got.variable.data) is type(ref.variable.data), "dask was computed"
    assert_same(got, ref)


def test_construct_uses_the_mojo_pad_on_the_fast_path(monkeypatch):
    """The gate must not have cost the fast path its kernel."""
    from mojo_xarray import _ops
    seen = []
    real = _ops.pad
    monkeypatch.setattr(_ops, "pad", lambda *a, **k: (seen.append(a), real(*a, **k))[1])
    mx.DataArray(np.arange(24.0), dims="x").rolling(x=3).construct("w")
    assert len(seen) == 1
