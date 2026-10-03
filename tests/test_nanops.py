"""Parity of `mojo_xarray.nanops` / `nputils` with `xarray.computation.nanops`.

The reference is upstream's own `nanops` module, called on the same array.
"""

import numpy as np
import pytest
import xarray as xr
from xarray.computation import nanops as UPSTREAM
from xarray.core import nputils as UP_NPUTILS

import mojo_xarray as mx
from mojo_xarray import _ops, nanops, nputils

from .conftest import assert_same


NAN_FUNCS = ["nansum", "nanmean", "nanmin", "nanmax", "nanmedian", "nanprod",
             "nanvar", "nanstd", "nanargmin", "nanargmax"]

SHAPES = [(1,), (7,), (33,), (2, 5), (5, 2), (3, 4, 5), (1, 1, 9)]


def _data(rng, shape, nan_frac=0.2):
    a = rng.normal(size=shape)
    a[rng.random(shape) < nan_frac] = np.nan
    return a


def test_module_mirrors_upstream_names():
    for name in NAN_FUNCS + ["nancumsum", "nancumprod"]:
        assert hasattr(nanops, name), name
        assert hasattr(UPSTREAM, name), name
    # nputils has no nansum upstream or here: nansum belongs to nanops
    for name in [n for n in NAN_FUNCS if n != "nansum"] + [
        "nancumsum", "nancumprod", "nanquantile"
    ]:
        assert hasattr(nputils, name), name
        assert hasattr(UP_NPUTILS, name), name


@pytest.mark.parametrize("name", NAN_FUNCS)
@pytest.mark.parametrize("shape", SHAPES)
def test_full_reduction_matches_upstream(rng, name, shape):
    a = _data(rng, shape)
    got = getattr(mx, name)(a)
    ref = getattr(UPSTREAM, name)(a)
    assert_same(got, ref)


@pytest.mark.parametrize("name", ["nansum", "nanmean", "nanmin", "nanmax",
                                   "nanmedian", "nanprod", "nanvar", "nanstd"])
@pytest.mark.parametrize("shape,axis", [((2, 5), 0), ((2, 5), 1), ((2, 5), -1),
                                         ((3, 4, 5), 0), ((3, 4, 5), 1),
                                         ((3, 4, 5), 2), ((3, 4, 5), -2),
                                         ((2, 5), (0, 1))])
def test_axis_reduction_matches_upstream(rng, name, shape, axis):
    a = _data(rng, shape)
    got = getattr(mx, name)(a, axis=axis)
    ref = getattr(UPSTREAM, name)(a, axis=axis)
    assert_same(got, ref)


@pytest.mark.parametrize("ddof", [0, 1, 2, 5])
@pytest.mark.parametrize("name", ["nanvar", "nanstd"])
def test_ddof_matches_upstream(rng, name, ddof):
    a = _data(rng, (6, 9))
    for axis in (None, 0, 1):
        assert_same(getattr(mx, name)(a, axis=axis, ddof=ddof),
                    getattr(UPSTREAM, name)(a, axis=axis, ddof=ddof))


def test_all_nan_slices_match_upstream(rng):
    a = _data(rng, (3, 4))
    a[1, :] = np.nan
    for name in ["nansum", "nanmean", "nanmin", "nanmax", "nanmedian", "nanprod",
                 "nanvar", "nanstd"]:
        for axis in (None, 0, 1):
            assert_same(getattr(mx, name)(a, axis=axis),
                        getattr(UPSTREAM, name)(a, axis=axis))


def test_nanprod_all_nan_is_one(rng):
    a = np.full((2, 3), np.nan)
    assert_same(mx.nanprod(a, axis=1), UPSTREAM.nanprod(a, axis=1))
    assert mx.nanprod(a, axis=1)[0] == 1.0


def test_nanargmin_raises_like_upstream(rng):
    a = _data(rng, (3, 4))
    a[0, :] = np.nan
    with pytest.raises(ValueError, match="All-NaN slice encountered"):
        mx.nanargmin(a, axis=1)
    with pytest.raises(ValueError, match="All-NaN slice encountered"):
        UPSTREAM.nanargmin(a, axis=1)


def test_nanargmin_argmax_dtype_is_intp(rng):
    a = _data(rng, (4, 6))
    assert mx.nanargmin(a, axis=1).dtype == np.intp
    assert mx.nanargmax(a, axis=1).dtype == np.intp


@pytest.mark.parametrize("min_count", [0, 1, 3, 7])
def test_nansum_min_count_matches_upstream(rng, min_count):
    for shape, axis in [((5, 4), None), ((5, 4), 0), ((5, 4), 1), ((3, 4, 5), 2)]:
        a = _data(rng, shape, nan_frac=0.4)
        assert_same(mx.nansum(a, axis=axis, min_count=min_count),
                    UPSTREAM.nansum(a, axis=axis, min_count=min_count))


@pytest.mark.parametrize("min_count", [0, 1, 3])
def test_nanprod_min_count_matches_upstream(rng, min_count):
    a = _data(rng, (5, 4), nan_frac=0.4)
    assert_same(mx.nanprod(a, axis=1, min_count=min_count),
                UPSTREAM.nanprod(a, axis=1, min_count=min_count))


def test_maybe_null_out_matches_upstream(rng):
    a = _data(rng, (4, 6), nan_frac=0.5)
    result = mx.nansum(a, axis=1)
    assert_same(mx.nanops._maybe_null_out(result, 1, np.isnan(a), 3),
                UPSTREAM._maybe_null_out(UPSTREAM.nansum(a, axis=1), 1, np.isnan(a), 3))


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_nancumsum_cumprod_match_upstream(rng, axis):
    a = _data(rng, (4, 6))
    assert_same(mx.nancumsum(a, axis=axis), UPSTREAM.nancumsum(a, axis=axis))
    assert_same(mx.nancumprod(a, axis=axis), UPSTREAM.nancumprod(a, axis=axis))


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_nanquantile_matches_numpy(rng, q, axis):
    # exact for every q that lands on a sample; one ulp out where the Mojo
    # backend contracts the lerp into an fma (documented in the README)
    a = _data(rng, (5, 7))
    got = nputils.nanquantile(a, q, axis=axis)
    ref = np.nanquantile(a, q, axis=axis)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=4 * np.finfo(np.float64).eps,
                               equal_nan=True)
    exact = np.array_equal(got, ref, equal_nan=True)
    if not exact:
        assert np.nanmax(np.abs(got - ref)) <= 4 * np.finfo(np.float64).eps * \
            np.nanmax(np.abs(ref))


def test_nanquantile_exact_when_q_hits_a_sample():
    a = np.array([3.0, 1.0, 2.0, 7.0])
    for q in (0.0, 1.0):
        assert nputils.nanquantile(a, q) == np.nanquantile(a, q)


def test_nanquantile_vector_q(rng):
    a = _data(rng, (5, 7))
    qs = [0.05, 0.5, 0.95]
    for axis in (None, 1):
        got = nputils.nanquantile(a, qs, axis=axis)
        ref = np.nanquantile(a, qs, axis=axis)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, rtol=4 * np.finfo(np.float64).eps,
                                   equal_nan=True)


def test_nanquantile_rejects_out_of_range_q(rng):
    a = _data(rng, (5,))
    with pytest.raises(ValueError):
        nputils.nanquantile(a, 1.5)


@pytest.mark.parametrize("name", NAN_FUNCS)
def test_integer_input_falls_back_and_still_matches(rng, name):
    a = rng.integers(-50, 50, size=(4, 5)).astype(np.int64)
    got = getattr(mx, name)(a, axis=1)
    ref = getattr(UPSTREAM, name)(a, axis=1)
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_float32_input_falls_back_and_still_matches(rng):
    a = _data(rng, (4, 5)).astype(np.float32)
    for name in NAN_FUNCS:
        got = getattr(mx, name)(a, axis=1)
        ref = getattr(UPSTREAM, name)(a, axis=1)
        assert np.asarray(got).dtype == np.asarray(ref).dtype
        assert_same(got, ref, rtol=1e-6, atol=1e-6)


def test_medians_of_even_and_odd_counts():
    odd = np.array([3.0, 1.0, 2.0])
    even = np.array([4.0, 1.0, 3.0, 2.0])
    with_nan = np.array([4.0, 1.0, np.nan, 3.0, 2.0])
    for a in (odd, even, with_nan):
        assert mx.nanmedian(a) == np.median(a[~np.isnan(a)])


def test_var_uses_the_same_guards_as_numpy():
    # count <= ddof must be NaN, not inf
    a = np.array([1.0, 2.0])
    for ddof in (0, 1, 2):
        got = mx.nanvar(a, ddof=ddof)
        ref = np.nanvar(a, ddof=ddof)
        assert np.isnan(got) == np.isnan(ref)
        if not np.isnan(ref):
            assert got == pytest.approx(ref)


# --- the parallel path -----------------------------------------------------
# Above `PAR_MIN_OUT * workers` output elements the kernels stripe the work
# across threads, and the sorting kernels need one private scratch per stripe.
# Every other test here is small enough to run serially, so these are the ones
# that would catch a shared-buffer race.

PARALLEL_OUT = 8192 * 16


def test_parallel_threshold_is_reachable():
    from mojo_xarray._ops import workers
    assert workers() >= 1


@pytest.mark.parametrize("name", ["nanmean", "nanmin", "nanmax", "nanmedian",
                                   "nanstd", "nansum", "nanargmin", "nanargmax"])
def test_parallel_axis_reduction_matches_upstream(rng, name):
    # argmin/argmax raise on an all-NaN slice, so keep the NaNs out of their way
    frac = 0.0 if "arg" in name else 0.2
    a = _data(rng, (PARALLEL_OUT + 7, 3), nan_frac=frac)
    for axis in (0, 1):
        assert_same(getattr(mx, name)(a, axis=axis), getattr(UPSTREAM, name)(a, axis=axis),
                    rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_parallel_nanquantile_matches_numpy(rng, q):
    a = _data(rng, (PARALLEL_OUT + 3, 4))
    got = nputils.nanquantile(a, q, axis=1)
    ref = np.nanquantile(a, q, axis=1)
    np.testing.assert_allclose(got, ref, rtol=4 * np.finfo(np.float64).eps,
                               equal_nan=True)


def test_parallel_axis_median_matches_upstream(rng):
    a = _data(rng, (200_003, 2))
    assert_same(mx.nanmedian(a, axis=1), UPSTREAM.nanmedian(a, axis=1), rtol=0)
    assert_same(mx.nanmedian(a, axis=0), UPSTREAM.nanmedian(a, axis=0), rtol=0)


def test_parallel_nancumsum_matches_upstream(rng):
    a = _data(rng, (200_003, 2))
    assert_same(mx.nancumsum(a, axis=1), UPSTREAM.nancumsum(a, axis=1))
    assert_same(mx.nancumsum(a, axis=0), UPSTREAM.nancumsum(a, axis=0))


# --- nanfirst / nanlast -----------------------------------------------------
# `xarray/core/nputils.py` computes both as an argmax over the notnull mask
# followed by a gather, so the parity here is against upstream's own functions
# including the all-missing case, where argmax of an all-False mask is 0 and
# the gather returns the NaN in position 0.

@pytest.mark.parametrize("name", ["nanfirst", "nanlast"])
@pytest.mark.parametrize("shape,axis", [((7,), 0), ((2, 5), 0), ((2, 5), 1),
                                         ((2, 5), -1), ((3, 4, 5), 0),
                                         ((3, 4, 5), 1), ((3, 4, 5), 2),
                                         ((3, 4, 5), -2)])
@pytest.mark.parametrize("nan_frac", [0.0, 0.2, 0.9])
def test_nanfirst_nanlast_match_upstream(rng, name, shape, axis, nan_frac):
    a = _data(rng, shape, nan_frac=nan_frac)
    got = getattr(nputils, name)(a, axis)
    ref = getattr(UP_NPUTILS, name)(a, axis)
    assert_same(got, ref, rtol=0)


@pytest.mark.parametrize("name", ["nanfirst", "nanlast"])
def test_nanfirst_nanlast_keepdims(rng, name):
    a = _data(rng, (3, 4))
    for axis in (0, 1):
        got = getattr(nputils, name)(a, axis, keepdims=True)
        ref = getattr(UP_NPUTILS, name)(a, axis, keepdims=True)
        assert got.shape == ref.shape
        assert_same(got, ref, rtol=0)


@pytest.mark.parametrize("name", ["nanfirst", "nanlast"])
def test_nanfirst_nanlast_all_nan_slices(rng, name):
    a = np.full((3, 4), np.nan)
    a[1, 0] = 2.0
    for axis in (0, 1):
        assert_same(getattr(nputils, name)(a, axis), getattr(UP_NPUTILS, name)(a, axis),
                    rtol=0)


@pytest.mark.parametrize("name", ["nanfirst", "nanlast"])
def test_nanfirst_nanlast_tuple_axis_matches_upstream(rng, name):
    a = _data(rng, (2, 5))
    # upstream unwraps a one-element tuple axis
    assert_same(getattr(nputils, name)(a, (1,)), getattr(UP_NPUTILS, name)(a, (1,)),
                rtol=0)


@pytest.mark.parametrize("name", ["nanfirst", "nanlast"])
def test_nanfirst_nanlast_infinities_count_as_observations(rng, name):
    a = np.array([np.nan, np.inf, -np.inf, 1.0])
    for axis in (0,):
        assert_same(getattr(nputils, name)(a, axis), getattr(UP_NPUTILS, name)(a, axis),
                    rtol=0)


@pytest.mark.parametrize("name", ["nanfirst", "nanlast"])
def test_nanfirst_nanlast_non_float64_falls_back(rng, name):
    a = _data(rng, (4, 5)).astype(np.float32)
    got = getattr(nputils, name)(a, 1)
    ref = getattr(UP_NPUTILS, name)(a, 1)
    assert got.dtype == ref.dtype
    assert_same(got, ref, rtol=0)


@pytest.mark.parametrize("name", ["nanfirst", "nanlast"])
def test_nanfirst_nanlast_parallel_path(rng, name):
    a = _data(rng, (PARALLEL_OUT + 5, 3), nan_frac=0.3)
    for axis in (0, 1):
        assert_same(getattr(nputils, name)(a, axis),
                    getattr(UP_NPUTILS, name)(a, axis), rtol=0)


def test_nanfirst_index_kernel_is_what_runs(rng):
    a = _data(rng, (6, 7), nan_frac=0.5)
    expected = np.argmax(~np.isnan(a), axis=1)
    got = _ops.first_valid_index(a, 1)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.intp
    # and the axis=0 form, where `inner` is not 1
    np.testing.assert_array_equal(_ops.first_valid_index(a, 0),
                                  np.argmax(~np.isnan(a), axis=0))


def test_module_mirrors_upstream_nanfirst_nanlast():
    for name in ("nanfirst", "nanlast"):
        assert hasattr(nputils, name)
        assert hasattr(UP_NPUTILS, name)
        assert name in mx.__all__
        assert getattr(mx, name) is getattr(nputils, name)
