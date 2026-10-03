"""The kernel paths the SIMD / selection / padding rewrite introduced.

Three of them are new shapes of work rather than new semantics, so each one is
pinned against NumPy rather than against a stored answer:

  * the vector bodies for `min`/`max`/`sum`/`count` and the variance second pass,
    including the scalar tail and a run shorter than one vector,
  * the order-statistic helpers behind `nanmedian` / `nanquantile`, which select
    two neighbouring samples instead of sorting the whole window,
  * `pad`, which now emits the last dimension a run at a time, on both the
    serial and the striped path.
"""

import numpy as np
import pytest
import xarray as xr

import mojo_xarray as mx
from mojo_xarray import _ops, nputils

from .conftest import assert_same

# `simd_width_of[float64]()` is 4 on this target; the tails below walk either
# side of it, and the sizes are the ones a real axis reduction produces.
LENGTHS = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129]
NAN_FRACS = [0.0, 0.05, 0.5, 1.0]


def _data(rng, n, nan_frac=0.2):
    a = rng.normal(size=n)
    a[rng.random(n) < nan_frac] = np.nan
    return a


def _column(rng, n, nan_frac):
    a = rng.normal(size=n)
    a[rng.random(n) < nan_frac] = np.nan
    return np.ascontiguousarray(a, dtype=np.float64)


# --- the vector bodies ------------------------------------------------------


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("nan_frac", NAN_FRACS)
@pytest.mark.parametrize("name,upstream", [
    ("nanmin", np.nanmin),
    ("nanmax", np.nanmax),
    ("nansum", np.nansum),
    ("nanmean", np.nanmean),
    ("nanvar", np.nanvar),
    ("nanstd", np.nanstd),
    ("nanprod", np.nanprod),
])
def test_contiguous_reduction_matches_numpy(rng, name, upstream, n, nan_frac):
    a = _column(rng, n, nan_frac)
    got = getattr(mx, name)(a)
    ref = upstream(a)
    assert_same(got, ref, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("nan_frac", NAN_FRACS)
def test_contiguous_min_max_are_exact(rng, n, nan_frac):
    """`min`/`max` skip NaNs, so the vector form is exact, not just close."""
    a = _column(rng, n, nan_frac)
    if np.all(np.isnan(a)):
        # an all-missing slice is NaN, which is the one value an order
        # statistic cannot take
        assert np.isnan(mx.nanmin(a)) and np.isnan(mx.nanmax(a))
        return
    assert mx.nanmin(a) == np.nanmin(a)
    assert mx.nanmax(a) == np.nanmax(a)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("nan_frac", [0.0, 0.3])
def test_contiguous_median_matches_numpy(rng, n, nan_frac):
    a = _column(rng, n, nan_frac)
    assert_same(mx.nanmedian(a), np.nanmedian(a), rtol=0)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_contiguous_quantile_matches_numpy(rng, n, q):
    a = _column(rng, n, 0.2)
    got = nputils.nanquantile(a, q)
    ref = np.nanquantile(a, q)
    if np.isnan(ref):
        assert np.isnan(got)
        return
    # the lerp is within an ulp of numpy's, as the README says
    assert abs(got - ref) <= 1e-12 * max(1.0, abs(ref))


@pytest.mark.parametrize("n", LENGTHS)
def test_every_other_column_matches_the_first(rng, n):
    """`inner > 1` takes the general kernel; the two must agree exactly."""
    a = _column(rng, (n, 5), 0.2)
    for name, ref_fn in (("mx_nanmin", lambda x: np.nanmin(x, axis=0)),
                         ("mx_nanmax", lambda x: np.nanmax(x, axis=0)),
                         ("mx_nansum", lambda x: np.nansum(x, axis=0))):
        got = _ops.reduce_axis(a, name, 0)
        assert_same(got, ref_fn(a), rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("n", LENGTHS)
def test_var_guards_agree_with_numpy(rng, n):
    """count <= ddof must be NaN on both sides, tail lengths included."""
    a = _column(rng, n, 0.4)
    for ddof in (0, 1, 2, 5):
        assert_same(mx.nanvar(a, ddof=ddof), np.nanvar(a, ddof=ddof), rtol=1e-12)
        assert_same(mx.nanstd(a, ddof=ddof), np.nanstd(a, ddof=ddof), rtol=1e-12)


def test_all_missing_slices_are_nan_not_a_sentinel(rng):
    a = _column(rng, 64, 1.0)
    assert np.isnan(mx.nanmin(a))
    assert np.isnan(mx.nanmax(a))
    assert mx.nansum(a) == 0.0
    assert mx.nanprod(a) == 1.0
    assert np.isnan(mx.nanmean(a))
    assert np.isnan(mx.nanmedian(a))


def test_infinities_are_valid_observations(rng):
    """A slice of infinities has an answer, not an all-missing one."""
    a = np.array([np.inf, np.inf, 1.0, np.nan, np.inf])
    assert mx.nanmin(a) == 1.0
    assert mx.nanmax(a) == np.inf
    # the infinities are observations, so the slice has an answer and it is
    # the same one numpy gives
    assert mx.nanmean(a) == np.nanmean(a)
    assert not np.isnan(mx.nanmin(a))


# --- the striped path -------------------------------------------------------


def test_striped_reduction_matches_the_serial_one(rng):
    """Above the stripe threshold the same kernel runs across workers; the
    partials are combined by the same rules the serial path uses."""
    big = _column(rng, 200_003, 0.2)
    block = _column(rng, (200_003, 2), 0.2)
    assert_same(mx.nanmin(block, axis=1), np.nanmin(block, axis=1), rtol=0)
    assert_same(mx.nanmax(block, axis=1), np.nanmax(block, axis=1), rtol=0)
    assert_same(mx.nansum(big), np.nansum(big), rtol=1e-10, atol=1e-12)
    assert_same(mx.nanmin(big), np.nanmin(big), rtol=0)
    assert_same(mx.nanmedian(big), np.nanmedian(big), rtol=0)
    for q in (0.25, 0.75):
        assert abs(nputils.nanquantile(big, q) - np.nanquantile(big, q)) <= 1e-12


def test_nanfirst_index_striped_path(rng):
    """Above the stripe threshold the first-valid scan splits across workers.

    A leading run of NaNs is what makes the scan the work: with observations
    everywhere the answer is index 0 and the loop exits on its first step.
    """
    a = _column(rng, (2_100_000, 2), 0.5)
    a[:1_050_000] = np.nan
    for axis in (0, 1):
        got = _ops.first_valid_index(a, axis)
        np.testing.assert_array_equal(got, np.argmax(~np.isnan(a), axis=axis))



# --- pad --------------------------------------------------------------------


@pytest.mark.parametrize("shape,pad_width", [
    ((5,), ((3, 2),)),
    ((1, 1), ((0, 0), (0, 0))),
    ((2, 5), ((0, 0), (3, 2))),
    ((3, 4), ((1, 0), (0, 2))),
    ((7, 9), ((2, 1), (0, 4))),
    ((2, 3, 4), ((1, 1), (0, 0), (2, 3))),
    ((2, 3, 4, 5), ((1, 0), (0, 1), (1, 1), (2, 2))),
])
def test_pad_matches_np_pad(rng, shape, pad_width):
    a = np.ascontiguousarray(rng.normal(size=shape))
    for fill in (np.nan, 0.0, -1.5):
        got = _ops.pad(a, pad_width, fill)
        ref = np.pad(a, pad_width, constant_values=fill)
        assert got.shape == ref.shape
        assert_same(got, ref)


def test_pad_striped_path_matches_the_serial_one(rng):
    """More runs than the stripe threshold, so `pad` splits across workers."""
    a = np.ascontiguousarray(rng.normal(size=(200_003, 2)))
    got = _ops.pad(a, ((0, 0), (3, 2)), np.nan)
    ref = np.pad(a, ((0, 0), (3, 2)), constant_values=np.nan)
    assert got.shape == ref.shape
    assert_same(got, ref)


# --- rolling ----------------------------------------------------------------


@pytest.mark.parametrize("window", [3, 5, 9])
@pytest.mark.parametrize("center", [False, True])
def test_rolling_window_longer_than_the_row(rng, window, center):
    """Every window hangs off both ends, which is the whole live range."""
    a = _data(rng, 4, nan_frac=0.3)
    g = mx.DataArray(a.copy(), dims="x")
    r = xr.DataArray(a.copy(), dims="x")
    for method in ("sum", "mean", "min", "max", "std", "var", "median", "count"):
        got = getattr(g.rolling(x=window, center=center, min_periods=1), method)()
        ref = getattr(r.rolling(x=window, center=center, min_periods=1), method)()
        assert_same(got, ref, rtol=1e-11, atol=1e-13)


def test_move_sum_of_squares_matches_numpy(rng):
    """`bottleneck.move_sum_of_squares`: the rolling sum of v**2.  It is ported
    and exported, so it needs a test of its own -- nothing in the public API
    calls it."""
    from mojo_xarray import _ops
    for shape, win, center, min_count in (
        ((1, 33), 5, False, 1), ((1, 33), 5, True, 1),
        ((3, 64), 8, False, 1), ((3, 64), 8, True, 1),
        ((2, 40), 9, False, 0), ((2, 40), 64, True, 2),
    ):
        a = rng.normal(size=shape)
        a[rng.random(shape) < 0.15] = np.nan
        a = np.ascontiguousarray(a, dtype=np.float64)
        got = _ops.move(a, "mx_move_sum_of_squares", win, min_count, center,
                        n_rows=a.size // a.shape[-1])
        lo = win // 2 if center else win - 1
        n = a.shape[-1]
        padded = np.full((a.shape[0], n + lo + (win - 1 - lo)), np.nan)
        padded[:, lo:lo + n] = a
        view = np.lib.stride_tricks.sliding_window_view(padded, win, axis=-1)
        gate = min_count if min_count > 0 else 0
        valid = win - np.isnan(view).sum(-1)
        ref = np.where(valid >= gate, np.nansum(view ** 2, axis=-1), np.nan)
        np.testing.assert_allclose(got, ref, rtol=1e-12, equal_nan=True)
