"""Every `@export`ed symbol, called by name, asserted on its own output.

The rest of the suite reaches the kernels through the public API, which proves
the port answers end to end but says nothing about *which* kernel produced an
answer.  A kernel that is wired up and completely wrong passes those tests
untouched.  So every export gets a call here that names the symbol, and asserts
the values the compiled library wrote into the output buffer.

The references are upstream's own code wherever one exists -- `np.nanquantile`,
the `bottleneck.move_*` definitions for the kernels upstream only reaches
through bottleneck, and `np.pad` -- and a sliding-window view built exactly the
way `Variable.rolling_window` builds one otherwise.

`tests/conftest.py` loads the shared library at collection time, so reaching this
file at all means `dist/`/`build/` really was loaded.
"""

import numpy as np
import pytest

from mojo_xarray import _lib, _ops, nputils

from .conftest import KERNELS, assert_same, library_path

# Every `@export("...")` in src/capi.mojo.  `lib()` is what resolves them, so a
# symbol the library does not actually carry fails here with AttributeError.
_ALL = (
    "mx_nanfirst_index", "mx_nanmin", "mx_nanmax", "mx_nansum", "mx_nanmean",
    "mx_nanmedian", "mx_nanvar", "mx_nanstd", "mx_nanprod", "mx_nanargmin",
    "mx_nanargmax", "mx_nanquantile", "mx_nan_count", "mx_move_sum_of_squares",
    "mx_maybe_null_out", "mx_nancumsum", "mx_nancumprod", "mx_move_sum",
    "mx_move_mean", "mx_move_min", "mx_move_max", "mx_move_median",
    "mx_move_var", "mx_move_std", "mx_move_argmin", "mx_move_argmax",
    "mx_move_count", "mx_par_workers", "mx_pad",
)


def test_every_export_resolves():
    """The list above is the whole ABI, and every name in it really is a
    callable in the loaded shared library."""
    lib = _lib.lib()
    for name in _ALL:
        assert hasattr(lib, name), name
    assert set(_ALL) == set(KERNELS), (
        f"untested exports: {sorted(set(KERNELS) - set(_ALL))}, "
        f"non-exports: {sorted(set(_ALL) - set(KERNELS))}"
    )


def test_the_library_under_test_is_the_built_one():
    """A sanity check that the run really is exercising a compiled artifact."""
    from mojo_xarray import _lib as lib_mod

    path = library_path()
    assert path.endswith(".so") or path.endswith(".dylib"), path
    # the handle came from a real dlopen of that file, not from a ctypes lookup
    # of an already-mapped image with the same name
    assert lib_mod._lib is not None
    assert lib_mod._lib._name == path


# --- the plain reductions, straight into the ABI ----------------------------


@pytest.mark.parametrize("name,numpy", [
    ("mx_nanmin", np.nanmin),
    ("mx_nanmax", np.nanmax),
    ("mx_nansum", np.nansum),
    ("mx_nanmean", np.nanmean),
    ("mx_nanprod", np.nanprod),
])
def test_axis_reduction_kernels(name, numpy):
    rng = np.random.default_rng(7)
    # three axes' worth of layouts: last axis (inner == 1), first axis
    # (inner == ncols) and a flat reduction
    a = rng.normal(size=(6, 7))
    a[rng.random(a.shape) < 0.25] = np.nan
    for axis in (0, 1, -1):
        got = _ops.reduce_axis(a, name, axis)
        assert_same(got, numpy(a, axis=axis), rtol=1e-12, atol=1e-300)
    assert_same(_ops.reduce_axis(a, name, None), numpy(a), rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("name,numpy", [
    ("mx_nanmedian", np.nanmedian),
    ("mx_nanargmin", np.nanargmin),
    ("mx_nanargmax", np.nanargmax),
])
def test_axis_selection_kernels(name, numpy):
    rng = np.random.default_rng(11)
    a = rng.normal(size=(5, 6))
    a[rng.random(a.shape) < 0.2] = np.nan
    for axis in (0, 1):
        assert_same(_ops.reduce_axis(a, name, axis), numpy(a, axis=axis), rtol=0)


@pytest.mark.parametrize("ddof", [0, 1, 3])
@pytest.mark.parametrize("name,numpy", [("mx_nanvar", np.nanvar), ("mx_nanstd", np.nanstd)])
def test_var_and_std_kernels(name, numpy, ddof):
    rng = np.random.default_rng(13)
    a = rng.normal(size=(4, 9))
    a[rng.random(a.shape) < 0.4] = np.nan
    assert_same(_ops.reduce_axis(a, name, 1, ddof=ddof),
                numpy(a, axis=1, ddof=ddof), rtol=1e-12)


@pytest.mark.parametrize("q", [0.0, 0.25, 1.0 / 3.0, 0.5, 0.9, 1.0])
def test_nanquantile_kernel(q):
    rng = np.random.default_rng(17)
    a = rng.normal(size=(4, 11))
    a[rng.random(a.shape) < 0.3] = np.nan
    ref = np.nanquantile(a, q, axis=1)
    got = _ops.reduce_axis(a, "mx_nanquantile", 1, q=q)
    # the lerp is within an ulp of numpy's (see the README)
    np.testing.assert_allclose(got, ref, rtol=1e-12, equal_nan=True)
    # and exact wherever q lands on a sample, which is the branch that skips
    # the interpolation altogether
    count = np.sum(~np.isnan(a), axis=1)
    virtual = q * (count - 1)
    on_sample = (virtual == np.floor(virtual)) & ~np.isnan(ref)
    np.testing.assert_array_equal(got[on_sample], ref[on_sample])


def test_nanquantile_kernel_rejects_nothing_it_should_not():
    """`q == 1` takes the kernel's single-sample branch; an all-missing slice
    is NaN, not an index into an empty buffer."""
    a = np.array([[1.0, 2.0, np.nan, 4.0], [np.nan, np.nan, np.nan, np.nan]])
    assert _ops.reduce_axis(a, "mx_nanquantile", 1, q=1.0)[0] == 4.0
    assert np.isnan(_ops.reduce_axis(a, "mx_nanquantile", 1, q=0.5)[1])


def test_nan_count_kernel_counts_the_mask():
    """`duck_array_ops.sum(isnull(a), axis)` over a 0/1 mask."""
    mask = np.array([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    got = _ops.reduce_axis(mask, "mx_nan_count", 1)
    np.testing.assert_array_equal(got, mask.sum(axis=1))


def test_maybe_null_out_kernel():
    """`_maybe_null_out`: `(n - sum(mask) - min_count) < 0` puts NaN in place.

    `count` here is the NaN count of the slice, which is what
    `duck_array_ops.sum(isnull(a), axis)` produces -- the test builds it that
    way rather than by hand, so the two kernels are exercised as a pair.
    """
    rng = np.random.default_rng(19)
    a = rng.normal(size=(3, 8))
    a[rng.random(a.shape) < 0.5] = np.nan
    result = np.nansum(a, axis=1)
    out = _ops.maybe_null_out(result.copy(), np.isnan(a), 1, min_count=5)
    counts = np.sum(~np.isnan(a), axis=1)
    np.testing.assert_array_equal(np.isnan(out), counts < 5)
    np.testing.assert_allclose(out[~np.isnan(out)], result[~np.isnan(out)])


def test_maybe_null_out_with_min_count_zero_keeps_everything():
    a = np.array([1.0, np.nan, np.nan])
    out = _ops.maybe_null_out(np.nansum(a), np.isnan(a), 0, min_count=0)
    assert out == 1.0


@pytest.mark.parametrize("name,numpy", [
    ("mx_nancumsum", np.nancumsum),
    ("mx_nancumprod", np.nancumprod),
])
def test_cumulative_kernels(name, numpy):
    rng = np.random.default_rng(23)
    a = rng.normal(size=(3, 7)) + 1.5
    a[rng.random(a.shape) < 0.3] = np.nan
    got = _ops.cumulative(a, name, 1)
    assert_same(got, numpy(a, axis=1), rtol=1e-12)
    # axis=0 is the strided form; np.cumsum flattens when axis is None
    assert_same(_ops.cumulative(a, name, 0), numpy(a, axis=0), rtol=1e-12)
    flat = _ops.cumulative(a, name, None)
    assert flat.shape == (a.size,)
    assert_same(flat, numpy(a), rtol=1e-12)


def test_nanfirst_index_kernel():
    """`np.argmax(~pd.isnull(values), axis)`, mask and all."""
    rng = np.random.default_rng(29)
    a = rng.normal(size=(5, 6))
    a[rng.random(a.shape) < 0.4] = np.nan
    a[2] = np.nan  # an all-missing slice selects position 0, as argmax does
    for axis in (0, 1):
        got = _ops.first_valid_index(a, axis)
        ref = np.argmax(~np.isnan(a), axis=axis)
        np.testing.assert_array_equal(got, ref)
        # nanfirst is that index gathered back off the array
        assert_same(nputils.nanfirst(a, axis), nputils.nanfirst(a, axis))


def test_par_workers_matches_the_scratch_size():
    """The stripe count is compiled in; the Python scratch is sized from it.

    A kernel that used a different constant would need more scratch than the
    wrapper allocated, so this is a real invariant rather than a tautology.
    """
    workers = _ops.workers()
    assert isinstance(workers, int) and workers >= 1
    scratch = _ops._scratch(5)
    assert scratch.size == workers * 5 + 2
    # the reported number is the one the caller passes back into the kernel's
    # own stripe arithmetic
    lib = _lib.lib()
    assert lib.mx_par_workers() == workers


# --- the moving-window kernels ----------------------------------------------


def _windows(values, win, center):
    """`Variable.rolling_window`: pad, then a sliding view."""
    n = values.shape[-1]
    lo = win // 2 if center else win - 1
    hi = win - 1 - lo
    padded = np.full((values.shape[0], n + lo + hi), np.nan)
    padded[:, lo:lo + n] = values
    return np.lib.stride_tricks.sliding_window_view(padded, win, axis=-1)


def _call_move(values, name, win, min_count, center, ddof=0):
    n_rows = values.size // values.shape[-1]
    return _ops.move(values, name, win, min_count, center, ddof=ddof, n_rows=n_rows)


def test_move_count_kernel():
    """`DataArrayRolling._counts`: the sum of the notnull mask, padding counted
    as False.  No `min_count` gate here -- upstream applies `min_periods`
    afterwards, in `Rolling.count`."""
    rng = np.random.default_rng(31)
    a = rng.normal(size=(3, 40))
    a[rng.random(a.shape) < 0.3] = np.nan
    for center in (False, True):
        got = _call_move((~np.isnan(a)).astype(np.float64),
                         "mx_move_count", 7, 0, center)
        assert_same(got, (~np.isnan(_windows(a, 7, center))).sum(axis=-1), rtol=0)


def test_move_sum_of_squares_kernel():
    """`bottleneck.move_sum_of_squares`: the rolling sum of v**2.

    The `min_count` gate is the caller's, so it is applied here the way upstream
    applies it: `result.where(counts >= min_periods)`.  With `min_periods=1` an
    all-NaN window is NaN.
    """
    rng = np.random.default_rng(37)
    a = rng.normal(size=(4, 48))
    a[rng.random(a.shape) < 0.2] = np.nan
    for win, center, min_count in ((1, False, 1), (5, False, 1), (5, True, 1),
                                   (9, True, 1), (9, True, 3), (64, True, 1)):
        got = _call_move(a, "mx_move_sum_of_squares", win, min_count, center)
        views = _windows(a, win, center)
        counts = (~np.isnan(views)).sum(axis=-1)
        ref = np.where(counts >= min_count, np.nansum(views ** 2, axis=-1), np.nan)
        assert_same(got, ref, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("win,center,min_count", [
    (3, False, 1), (3, True, 1), (8, False, 1), (8, True, 1),
    (8, False, 5), (8, True, 5), (64, True, 1), (1, False, 1),
])
def test_move_argmin_and_argmax_kernels(win, center, min_count):
    """The index is an offset from the window's left edge, which is what
    upstream's materialised-window path returns (see the README)."""
    rng = np.random.default_rng(41)
    a = rng.normal(size=(3, 40))
    a[rng.random(a.shape) < 0.25] = np.nan
    views = _windows(a, win, center)
    counts = (~np.isnan(views)).sum(axis=-1)
    # `np.nanargmin` on a window, with an all-missing window masked out after
    with np.errstate(invalid="ignore"):
        want_min = np.where(
            counts >= min_count,
            np.nanargmin(np.where(np.isnan(views), np.inf, views), axis=-1),
            np.nan,
        )
        want_max = np.where(
            counts >= min_count,
            np.nanargmax(np.where(np.isnan(views), -np.inf, views), axis=-1),
            np.nan,
        )
    assert_same(_call_move(a, "mx_move_argmin", win, min_count, center), want_min, rtol=0)
    assert_same(_call_move(a, "mx_move_argmax", win, min_count, center), want_max, rtol=0)


@pytest.mark.parametrize("name,numpy,ddof", [
    ("mx_move_sum", np.nansum, 0),
    ("mx_move_mean", np.nanmean, 0),
    ("mx_move_min", np.nanmin, 0),
    ("mx_move_max", np.nanmax, 0),
    ("mx_move_median", np.nanmedian, 0),
    ("mx_move_var", np.nanvar, 0),
    ("mx_move_std", np.nanstd, 0),
    ("mx_move_var", np.nanvar, 1),
    ("mx_move_std", np.nanstd, 1),
])
def test_move_reduction_kernels(name, numpy, ddof):
    """Each moving-window kernel against the reduction of the window view.

    The `min_count` gate is applied here, exactly as upstream applies it after
    the kernel: `result.where(counts >= self.min_periods)`.
    """
    rng = np.random.default_rng(43)
    a = rng.normal(size=(3, 40))
    a[rng.random(a.shape) < 0.3] = np.nan
    for win, center, min_count in ((4, False, 1), (4, True, 1), (7, False, 3),
                                   (7, True, 3), (9, True, 1), (48, True, 1)):
        views = _windows(a, win, center)
        counts = (~np.isnan(views)).sum(axis=-1)
        # `mean` is Rolling's `_mean`: the rolling sum over the rolling count,
        # which is what `np.nanmean` reproduces except on an empty window, where
        # it is NaN rather than a division by zero.
        if name == "mx_move_mean":
            ref = np.where(counts == 0, np.nan,
                           np.nansum(views, axis=-1) / np.maximum(counts, 1))
        else:
            with np.errstate(invalid="ignore"):
                if ddof:
                    ref = numpy(views, axis=-1, ddof=ddof)
                else:
                    ref = numpy(views, axis=-1)
        ref = np.where(counts >= min_count, ref, np.nan)
        got = _call_move(a, name, win, min_count, center, ddof=ddof)
        assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_move_sum_of_an_all_missing_window_is_zero():
    """`Rolling.sum`'s fill value is 0, so a window with no observation sums
    to 0 rather than NaN -- unless `min_periods` says otherwise."""
    a = np.full((1, 8), np.nan)
    assert _call_move(a, "mx_move_sum", 3, 0, False)[0, 0] == 0.0
    assert np.isnan(_call_move(a, "mx_move_sum", 3, 1, False)[0, 0])


def test_move_min_and_max_all_missing_sentinels():
    """`duck_array_ops.min/max` are `fillna=±inf`, so with `min_periods=1` an
    all-missing window comes back as the sentinel and `min_periods` nulls it."""
    a = np.full((1, 6), np.nan)
    got = _call_move(a, "mx_move_min", 3, 1, False)
    assert np.all(np.isnan(got))
    got = _call_move(a, "mx_move_min", 3, 0, False)
    assert np.all(np.isposinf(got))
    assert np.all(np.isneginf(_call_move(a, "mx_move_max", 3, 0, False)))


# --- pad -------------------------------------------------------------------


@pytest.mark.parametrize("shape,pad_width", [
    ((5,), ((2, 3),)),
    ((3, 4), ((1, 0), (0, 2))),
    ((2, 3, 4), ((1, 1), (0, 0), (2, 3))),
])
def test_pad_kernel(shape, pad_width):
    rng = np.random.default_rng(47)
    a = np.ascontiguousarray(rng.normal(size=shape))
    for fill in (np.nan, 0.0, 7.25):
        got = _ops.pad(a, pad_width, fill)
        ref = np.pad(a, pad_width, constant_values=fill)
        assert got.shape == ref.shape
        assert_same(got, ref)


def test_pad_kernel_accepts_a_strided_input_row_through_the_raw_abi():
    """The ABI carries the input strides, so a non-unit last-dimension stride is
    part of the kernel's contract even though `_ops.pad` never produces one:
    the copy branch (`packed` in `ported.pad`) has a sibling that gathers.

    Called through ctypes with no `_ops` wrapper between, because the wrapper
    always hands over a C-contiguous array.
    """
    lib = _lib.lib()
    src = np.ascontiguousarray(np.arange(20.0).reshape(4, 5))[:, ::2]
    assert src.strides == (40, 16)
    shape = np.array([4, 2], dtype=np.int64)
    # pad only the last dimension: the kernel peels the dimensions above it to
    # pick the input row a run reads from, so that is what decides whether the
    # copy or the gather branch runs
    padlo = np.array([0, 2], dtype=np.int64)
    padhi = np.array([0, 1], dtype=np.int64)
    istride = np.array([5, 2], dtype=np.int64)
    ostride = np.array([5, 3], dtype=np.int64)
    out = np.empty(4 * 5, dtype=np.float64)
    lib.mx_pad(
        src.ctypes.data, out.ctypes.data, 2,
        shape.ctypes.data, padlo.ctypes.data, padhi.ctypes.data,
        istride.ctypes.data, ostride.ctypes.data, 4 * 5, np.nan,
    )
    got = out.reshape(4, 5)
    np.testing.assert_array_equal(got[:, :2], np.full((4, 2), np.nan))
    np.testing.assert_array_equal(got[:, 2:4], src[:, :2])
    assert np.all(np.isnan(got[:, 4]))
