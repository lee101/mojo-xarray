"""`mojo_xarray.DataArray` / `Dataset` against the plain xarray classes.

These check the drop-in surface: the same object, the same method names, the
same signatures, the same answers, and the same fallback to upstream whenever
the Mojo kernels do not apply.
"""

import inspect
import warnings

import numpy as np
import pytest
import xarray as xr

import mojo_xarray as mx

from .conftest import assert_same

# `quantile` is not overridden: see the README's coverage section.
METHODS = ["sum", "mean", "min", "max", "std", "var", "median", "prod",
           "argmin", "argmax"]


def _pair(data, dims=("x",), **kwargs):
    data = np.ascontiguousarray(data, dtype=np.float64)
    return (mx.DataArray(data.copy(), dims=dims, **kwargs),
            xr.DataArray(data.copy(), dims=dims, **kwargs))


def _data(rng, n, nan_frac=0.2):
    a = rng.normal(size=n)
    a[rng.random(n) < nan_frac] = np.nan
    return a


def test_dataarray_is_an_xarray_dataarray():
    da = mx.DataArray(np.arange(4.0), dims="x", name="v", attrs={"u": "m"})
    assert isinstance(da, xr.DataArray)
    assert da.name == "v" and da.attrs == {"u": "m"}
    assert da.dims == ("x",)


def test_dataset_is_an_xarray_dataset():
    ds = mx.Dataset({"a": ("x", np.arange(4.0))})
    assert isinstance(ds, xr.Dataset)


def assert_same_dataset(got, ref, rtol=1e-11, atol=1e-13):
    """`assert_same` for Datasets: same variables, same coords, same values."""
    assert sorted(got.data_vars) == sorted(ref.data_vars)
    assert sorted(got.coords) == sorted(ref.coords)
    assert got.dims == ref.dims
    assert got.attrs == ref.attrs
    for key in got.data_vars:
        assert got[key].dims == ref[key].dims, key
        assert got[key].dtype == ref[key].dtype, key
        assert_same(got[key], ref[key], rtol=rtol, atol=atol)
    for key in got.coords:
        assert_same(got[key], ref[key], rtol=rtol, atol=atol)


def dataset_values(ds):
    """Every data variable as a dict of float64 arrays."""
    return {k: np.asarray(ds[k].values, dtype=np.float64) for k in ds.data_vars}


@pytest.mark.parametrize("method", METHODS + ["quantile"])
def test_method_signatures_match_upstream(method):
    mine = inspect.signature(getattr(mx.DataArray, method))
    theirs = inspect.signature(getattr(xr.DataArray, method))
    assert list(mine.parameters) == list(theirs.parameters), method
    for name in theirs.parameters:
        assert mine.parameters[name].default == theirs.parameters[name].default, name


@pytest.mark.parametrize("method", ["rolling", "coarsen"])
def test_accessor_signatures_match_upstream(method):
    mine = inspect.signature(getattr(mx.DataArray, method))
    theirs = inspect.signature(getattr(xr.DataArray, method))
    assert list(mine.parameters) == list(theirs.parameters), method
    for name in theirs.parameters:
        if name == "return_annotation":
            continue
        assert mine.parameters[name].default == theirs.parameters[name].default, name


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_reduction_matches_upstream(rng, method, axis):
    a = _data(rng, 24)
    g, r = _pair(a)
    got = getattr(g, method)(axis=axis) if axis is not None else getattr(g, method)()
    ref = getattr(r, method)(axis=axis) if axis is not None else getattr(r, method)()
    assert_same(got, ref, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("method", METHODS)
def test_reduction_over_a_named_dim_matches_upstream(rng, method):
    a = _data(rng, 12).reshape(3, 4)
    g, r = _pair(a, dims=("a", "b"))
    assert_same(getattr(g, method)(dim="b"), getattr(r, method)(dim="b"),
                rtol=1e-11, atol=1e-13)
    got = getattr(g, method)(dim=("a", "b"))
    ref = getattr(r, method)(dim=("a", "b"))
    # argmin/argmax answer a sequence of dims with a dict of DataArrays
    if isinstance(ref, dict):
        assert set(got) == set(ref)
        for k in ref:
            assert_same(got[k], ref[k])
    else:
        assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_argmin_dim_sequence_returns_a_dict_like_upstream(rng):
    a = _data(rng, 12).reshape(3, 4)
    g, r = _pair(a, dims=("a", "b"))
    got, ref = g.argmin(dim=("a", "b")), r.argmin(dim=("a", "b"))
    assert isinstance(got, dict) and isinstance(ref, dict)
    assert set(got) == set(ref) == {"a", "b"}
    assert_same(got["a"], ref["a"])
    assert_same(got["b"], ref["b"])
    # the result must be usable with isel, which is its purpose
    assert_same(g.isel(got), r.isel(ref))


def test_min_count_on_sum_matches_upstream(rng):
    a = _data(rng, 12, nan_frac=0.4)
    g, r = _pair(a)
    for min_count in (0, 1, 3, 6):
        assert_same(g.sum(min_count=min_count), r.sum(min_count=min_count),
                    rtol=1e-11, atol=1e-13)


def test_ddof_on_std_var_matches_upstream(rng):
    a = _data(rng, 12)
    g, r = _pair(a)
    for ddof in (0, 1, 2):
        assert_same(g.std(ddof=ddof), r.std(ddof=ddof), rtol=1e-11, atol=1e-13)
        assert_same(g.var(ddof=ddof), r.var(ddof=ddof), rtol=1e-11, atol=1e-13)


def test_quantile_is_inherited_from_upstream(rng):
    a = _data(rng, 12)
    g, r = _pair(a)
    for q in (0.1, 0.5, 0.9):
        assert_same(g.quantile(q), r.quantile(q), rtol=1e-11, atol=1e-13)
    assert_same(g.quantile([0.25, 0.75]), r.quantile([0.25, 0.75]),
                rtol=1e-11, atol=1e-13)


def test_reduction_keeps_coords_name_and_attrs(rng):
    a = _data(rng, 20).reshape(2, 10)
    coords = {"y": ("y", [0, 1]),
              "lbl": ("x", [f"t{i}" for i in range(10)])}
    g, r = _pair(a, dims=("y", "x"), coords=coords, name="v",
                 attrs={"units": "m"})
    got = g.sum(dim="x")
    ref = r.sum(dim="x")
    assert got.name == ref.name == "v"
    assert got.attrs == ref.attrs == {"units": "m"}
    np.testing.assert_array_equal(got.coords["y"].values, ref.coords["y"].values)
    assert_same(got, ref)


def test_keep_attrs_false_drops_attrs(rng):
    a = _data(rng, 8)
    g, r = _pair(a, attrs={"units": "m"})
    assert g.sum(keep_attrs=False).attrs == r.sum(keep_attrs=False).attrs == {}
    assert g.sum(keep_attrs=True).attrs == r.sum(keep_attrs=True).attrs


def test_skipna_false_falls_back_to_upstream(rng):
    a = _data(rng, 8)
    g, r = _pair(a)
    for method in ["sum", "mean", "min", "max", "prod"]:
        got = getattr(g, method)(skipna=False)
        ref = getattr(r, method)(skipna=False)
        assert_same(got, ref)


def test_integer_data_falls_back_to_upstream(rng):
    a = rng.integers(-30, 30, size=(3, 4))
    g = mx.DataArray(a, dims=("a", "b"))
    r = xr.DataArray(a.copy(), dims=("a", "b"))
    for method in METHODS:
        got = getattr(g, method)(dim="b")
        ref = getattr(r, method)(dim="b")
        assert np.asarray(got.values).dtype == np.asarray(ref.values).dtype, method
        assert_same(got, ref)


def test_float32_data_falls_back_to_upstream(rng):
    a = _data(rng, (3, 4)).astype(np.float32)
    g = mx.DataArray(a, dims=("a", "b"))
    r = xr.DataArray(a.copy(), dims=("a", "b"))
    for method in METHODS:
        got = getattr(g, method)(dim="b")
        ref = getattr(r, method)(dim="b")
        assert got.dtype == ref.dtype, method
        assert_same(got, ref, rtol=1e-5, atol=1e-6)


def test_non_contiguous_input_matches_upstream(rng):
    a = _data(rng, (4, 6))[:, ::2]
    assert not a.flags["C_CONTIGUOUS"]
    g = mx.DataArray(a, dims=("a", "b"))
    r = xr.DataArray(a.copy(), dims=("a", "b"))
    assert_same(g.mean(dim="b"), r.mean(dim="b"), rtol=1e-11, atol=1e-13)
    assert_same(g.rolling(b=2, min_periods=1).sum(), r.rolling(b=2, min_periods=1).sum(),
                rtol=1e-11, atol=1e-13)


def test_nan_argmin_raises_like_upstream(rng):
    a = _data(rng, (2, 4))
    a[0, :] = np.nan
    g = mx.DataArray(a, dims=("r", "c"))
    r = xr.DataArray(a.copy(), dims=("r", "c"))
    with pytest.raises(ValueError, match="All-NaN slice encountered"):
        g.argmin(dim="c")
    with pytest.raises(ValueError, match="All-NaN slice encountered"):
        r.argmin(dim="c")


def test_rolling_dispatch_uses_the_kernels(rng):
    """The fast path must actually be the Mojo one, not a silent fallback."""
    from mojo_xarray import rolling as mx_rolling

    calls = []
    real = mx_rolling._ops.move

    def spy(*args, **kwargs):
        calls.append(args[1])
        return real(*args, **kwargs)

    mx_rolling._ops.move = spy
    try:
        a = _data(rng, 20)
        g, _ = _pair(a)
        for method, kernel in [("sum", "mx_move_sum"), ("mean", "mx_move_mean"),
                               ("min", "mx_move_min"), ("max", "mx_move_max"),
                               ("median", "mx_move_median"), ("std", "mx_move_std"),
                               ("var", "mx_move_var"), ("count", "mx_move_count")]:
            calls.clear()
            getattr(g.rolling(x=3), method)()
            assert calls, method
            assert calls[0] == kernel, method
    finally:
        mx_rolling._ops.move = real


def test_reductions_use_the_kernels(rng):
    from mojo_xarray import _ops

    real = _ops.reduce_axis
    names = []

    def spy(a, name, *args, **kwargs):
        names.append(name)
        return real(a, name, *args, **kwargs)

    _ops.reduce_axis = spy
    try:
        a = _data(rng, 8)
        g, _ = _pair(a)
        for method, kernel in [("sum", "mx_nansum"), ("mean", "mx_nanmean"),
                               ("min", "mx_nanmin"), ("max", "mx_nanmax"),
                               ("median", "mx_nanmedian"), ("std", "mx_nanstd"),
                               ("var", "mx_nanvar"), ("argmin", "mx_nanargmin"),
                               ("argmax", "mx_nanargmax")]:
            names.clear()
            getattr(g, method)()
            assert kernel in names, method
    finally:
        _ops.reduce_axis = real


def test_as_xarray_escape_hatch(rng):
    a = _data(rng, 8)
    g, _ = _pair(a, dims=("x",), name="v", attrs={"u": "m"})
    plain = mx.as_xarray(g)
    assert type(plain) is xr.DataArray
    assert plain.name == "v" and plain.attrs == {"u": "m"}
    assert_same(plain.sum(dim="x"), xr.DataArray(a, dims="x").sum(dim="x"),
                rtol=1e-11, atol=1e-13)


def test_toplevel_names_are_exported():
    for name in ["nanmin", "nanmax", "nansum", "nanmean", "nanmedian", "nanvar",
                 "nanstd", "nanprod", "nancumsum", "nancumprod", "nanargmin",
                 "nanargmax", "nanquantile", "DataArray", "Dataset", "Rolling",
                 "DataArrayRolling", "DatasetRolling", "Coarsen",
                 "DataArrayCoarsen", "DatasetCoarsen", "nanops", "nputils",
                 "rolling"]:
        assert hasattr(mx, name), name
        assert name in mx.__all__, name


def test_readme_usage_example_runs(rng):
    """The snippet in the README, verbatim."""
    da = mx.DataArray(
        np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0]),
        dims="time",
        coords={"time": np.arange(7)},
        name="temperature",
        attrs={"units": "K"},
    )
    out = da.rolling(time=3, min_periods=1, center=True).mean()
    assert out.dims == ("time",)
    assert out.name == "temperature"
    assert out.attrs == {"units": "K"}
    assert_same(out, da.rolling(time=3, min_periods=1, center=True).mean())
    np.testing.assert_allclose(
        out.values, [1.5, 1.5, 3.0, 4.5, 5.0, 6.0, 6.5], rtol=1e-12
    )


# --- Dataset ----------------------------------------------------------------
#
# `Dataset.__getitem__` builds a plain `xarray.DataArray`, so the DataArray
# overrides above never fire from `Dataset.mean` and friends: upstream's
# `Dataset.reduce` calls `Variable.reduce` with `duck_array_ops.<name>`.  These
# check the `Dataset` overrides in `api.py` that substitute the other half.


def _dataset_pair(rng):
    a = _data(rng, (4, 9), nan_frac=0.2)
    b = _data(rng, 9, nan_frac=0.1)
    mine = {"v": (("x", "y"), a), "w": ("y", b)}
    g = mx.Dataset(dict(mine), coords={"y": np.arange(9)}, attrs={"u": "m"})
    r = xr.Dataset({k: (d, v.copy()) for (k, (d, v)) in mine.items()},
                   coords={"y": np.arange(9)}, attrs={"u": "m"})
    return g, r


@pytest.mark.parametrize("method", METHODS + ["quantile", "count"])
def test_dataset_method_signatures_match_upstream(method):
    mine = inspect.signature(getattr(mx.Dataset, method))
    theirs = inspect.signature(getattr(xr.Dataset, method))
    assert list(mine.parameters) == list(theirs.parameters), method
    for name in theirs.parameters:
        assert mine.parameters[name].default == theirs.parameters[name].default, name


@pytest.mark.parametrize("method", sorted({
    "sum", "mean", "min", "max", "std", "var", "median", "prod",
    "argmin", "argmax", "count", "quantile",
} | {"quantile", "count"}))
def test_dataset_method_signatures_match_upstream(method):
    mine = inspect.signature(getattr(mx.Dataset, method))
    theirs = inspect.signature(getattr(xr.Dataset, method))
    assert list(mine.parameters) == list(theirs.parameters), method
    for name in theirs.parameters:
        assert mine.parameters[name].default == theirs.parameters[name].default, (
            f"{method}.{name}: {mine.parameters[name].default!r} != "
            f"{theirs.parameters[name].default!r}"
        )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dim", ["y", ["x", "y"], None])
def test_dataset_reduction_matches_upstream(rng, method, dim):
    g, r = _dataset_pair(rng)
    if method in ("argmin", "argmax") and isinstance(dim, list):
        # upstream's own branch: a Dataset cannot hold the dict a sequence of
        # dims produces, so it raises rather than reducing
        with pytest.raises(ValueError):
            getattr(g, method)(dim=dim)
        with pytest.raises(ValueError):
            getattr(r, method)(dim=dim)
        return
    with warnings.catch_warnings():
        # `Dataset.argmin()` with no dim warns on both sides
        warnings.simplefilter("ignore", FutureWarning)
        assert_same_dataset(getattr(g, method)(dim=dim), getattr(r, method)(dim=dim))


def test_dataset_sum_min_count_matches_upstream(rng):
    g, r = _dataset_pair(rng)
    for min_count in (0, 1, 5, 12):
        assert_same_dataset(g.sum(dim="y", min_count=min_count),
                            r.sum(dim="y", min_count=min_count))
    # unlike `DataArray.sum`, upstream's `Dataset.sum` has no negative-min_count
    # guard -- `_maybe_null_out` only ever sees a positive one -- so neither does
    # the override
    assert_same_dataset(g.sum(dim="y", min_count=-1),
                        r.sum(dim="y", min_count=-1))


def test_dataset_prod_min_count_matches_upstream(rng):
    """`DataArray.prod` routes a `min_count` to upstream because
    `duck_array_ops.prod` has no `available_min_count`; `Dataset.prod` does not,
    so the override keeps it."""
    g, r = _dataset_pair(rng)
    for min_count in (0, 1, 5, 12):
        assert_same_dataset(g.prod(dim="y", min_count=min_count),
                            r.prod(dim="y", min_count=min_count))


def test_dataset_std_var_ddof_matches_upstream(rng):
    g, r = _dataset_pair(rng)
    for ddof in (0, 1, 2):
        assert_same_dataset(g.std(dim="y", ddof=ddof), r.std(dim="y", ddof=ddof))
        assert_same_dataset(g.var(dim="y", ddof=ddof), r.var(dim="y", ddof=ddof))


def test_dataset_skipna_false_matches_upstream(rng):
    """`_create_nan_agg_method` sends `skipna=False` to NumPy, not to the
    kernel, and so must the port."""
    g, r = _dataset_pair(rng)
    for method in METHODS:
        assert_same_dataset(getattr(g, method)(dim="y", skipna=False),
                            getattr(r, method)(dim="y", skipna=False), rtol=0)


def test_dataset_integer_variable_keeps_its_dtype(rng):
    a = rng.integers(-30, 30, size=(3, 5))
    g = mx.Dataset({"v": (("x", "y"), a)})
    r = xr.Dataset({"v": (("x", "y"), a.copy())})
    for method in METHODS:
        got, ref = getattr(g, method)(dim="y"), getattr(r, method)(dim="y")
        assert got["v"].dtype == ref["v"].dtype, method
        assert_same_dataset(got, ref)


def test_dataset_argmin_dim_sequence_still_raises(rng):
    """Upstream's dict-of-dims branch raises before any reduction runs, and
    the override keeps that."""
    g, r = _dataset_pair(rng)
    with pytest.raises(ValueError):
        g.argmin(dim=("x", "y"))
    with pytest.raises(ValueError):
        r.argmin(dim=("x", "y"))
    with pytest.raises(ValueError):
        g.argmin(dim=...)
    with pytest.raises(ValueError):
        r.argmin(dim=...)


def test_dataset_keep_attrs_and_numeric_only(rng):
    g, r = _dataset_pair(rng)
    got, ref = g.sum(dim="y", keep_attrs=False), r.sum(dim="y", keep_attrs=False)
    assert got.attrs == ref.attrs == {}
    got, ref = g.sum(dim="y", keep_attrs=True), r.sum(dim="y", keep_attrs=True)
    assert got.attrs == ref.attrs == {"u": "m"}
    # `numeric_only=True` drops a non-numeric variable on both sides
    g2 = g.assign(label=("y", list("abcdefghi")))
    r2 = r.assign(label=("y", list("abcdefghi")))
    assert set(g2.mean(dim="y").data_vars) == set(r2.mean(dim="y").data_vars)


def test_dataset_reductions_use_the_kernels(rng):
    """The fast path must actually be the Mojo one, not a silent fallback."""
    from mojo_xarray import _ops

    real = _ops.reduce_axis
    names = []

    def spy(a, name, *args, **kwargs):
        names.append(name)
        return real(a, name, *args, **kwargs)

    _ops.reduce_axis = spy
    try:
        g, _ = _dataset_pair(rng)
        for method, kernel in [("sum", "mx_nansum"), ("mean", "mx_nanmean"),
                               ("min", "mx_nanmin"), ("max", "mx_nanmax"),
                               ("median", "mx_nanmedian"), ("std", "mx_nanstd"),
                               ("var", "mx_nanvar"), ("prod", "mx_nanprod"),
                               ("argmin", "mx_nanargmin"),
                               ("argmax", "mx_nanargmax")]:
            names.clear()
            getattr(g, method)(dim="y")
            assert names and names[0] == kernel, method
    finally:
        _ops.reduce_axis = real


def test_dataset_rolling_and_coarsen_use_the_kernels(rng):
    """`DatasetRolling` builds its own `DataArrayRolling` per variable, so those
    do reach the move kernels; `DatasetCoarsen` reduces each variable through
    the same `_coarsen_reduce` the DataArray form uses."""
    from mojo_xarray import _ops
    from mojo_xarray import rolling as mx_rolling

    real_move, real_reduce = _ops.move, _ops.reduce_axis
    moved, reduced = [], []

    def move_spy(*args, **kwargs):
        moved.append(args[1])
        return real_move(*args, **kwargs)

    def reduce_spy(a, name, *args, **kwargs):
        reduced.append(name)
        return real_reduce(a, name, *args, **kwargs)

    _ops.move, _ops.reduce_axis = move_spy, reduce_spy
    mx_rolling._ops.move = move_spy
    try:
        g, _ = _dataset_pair(rng)
        moved.clear()
        g.rolling(y=4, min_periods=1).sum()
        assert moved and set(moved) == {"mx_move_sum"}, moved
        reduced.clear()
        g.coarsen(y=3).mean()
        assert reduced and set(reduced) == {"mx_nanmean"}, reduced
    finally:
        _ops.move, _ops.reduce_axis = real_move, real_reduce
        mx_rolling._ops.move = real_move
