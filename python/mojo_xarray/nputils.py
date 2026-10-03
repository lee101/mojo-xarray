"""Port of `xarray/core/nputils.py`.

Upstream builds these out of `_create_method`, which forwards to NumPy -- or to
bottleneck, or numbagg, depending on the options and the dtype.  Here the
float64 case is a Mojo kernel.  Every other dtype is delegated back to NumPy:
upstream's result dtype for those is the input dtype, and accumulating int64 in
float64 would lose exactness above 2**53, so pretending otherwise would be a
silent numerical divergence rather than a port.
"""

from __future__ import annotations

import numpy as np

from . import _ops

__all__ = [
    "nanmin", "nanmax", "nanmean", "nanmedian", "nanvar", "nanstd", "nanprod",
    "nancumsum", "nancumprod", "nanargmin", "nanargmax", "nanquantile",
    "nanfirst", "nanlast",
    "USES_MOJO",
]

# The dtype the Mojo kernels compute in.
USES_MOJO = np.dtype(np.float64)



def _select_along_axis(values, idx, axis):
    other_ind = np.ix_(*[np.arange(s) for s in idx.shape])
    sl = other_ind[:axis] + (idx,) + other_ind[axis:]
    return values[sl]


def _first_valid_index(values, axis):
    """Upstream's `np.argmax(~pd.isnull(values), axis=axis)`.

    The Mojo kernel walks the slice until the first observation, which is the
    same index without materialising the bool mask; anything that is not a
    contiguous float64 array keeps upstream's expression.
    """
    if _uses_mojo(values):
        return _ops.first_valid_index(values, axis)
    import pandas as pd

    return np.argmax(~pd.isnull(values), axis=axis)


def nanfirst(values, axis, keepdims=False):
    if isinstance(axis, tuple):
        (axis,) = axis
    axis = _ops.normalize_axis_index(axis, values.ndim)
    idx_first = _first_valid_index(values, axis)
    result = _select_along_axis(values, idx_first, axis)
    if keepdims:
        return np.expand_dims(result, axis=axis)
    else:
        return result


def nanlast(values, axis, keepdims=False):
    if isinstance(axis, tuple):
        (axis,) = axis
    axis = _ops.normalize_axis_index(axis, values.ndim)
    rev = (slice(None),) * axis + (slice(None, None, -1),)
    idx_last = -1 - _first_valid_index(values[rev], axis)
    result = _select_along_axis(values, idx_last, axis)
    if keepdims:
        return np.expand_dims(result, axis=axis)
    else:
        return result

def _uses_mojo(a) -> bool:
    return isinstance(a, np.ndarray) and a.dtype == USES_MOJO and a.ndim > 0


def _dispatch(a, axis, func):
    """Reduce one axis at a time, which is what NumPy means by a tuple axis.

    Highest index first, so the remaining axis numbers stay valid as the array
    shrinks.
    """
    if axis is None:
        return func(a, None)
    axes = axis if isinstance(axis, tuple) else (axis,)
    out = a
    for ax in sorted((_ops.normalize_axis_index(ax, a.ndim) for ax in axes),
                     reverse=True):
        out = func(out, ax)
    return out


def _pick(a, name, numpy):
    if _uses_mojo(a):
        return lambda x, ax: _ops.reduce_axis(x, name, ax)
    return numpy


def nanmin(a, axis=None, **kwargs):
    return _dispatch(a, axis, _pick(a, "mx_nanmin",
                                    lambda x, ax: np.nanmin(x, axis=ax, **kwargs)))


def nanmax(a, axis=None, **kwargs):
    return _dispatch(a, axis, _pick(a, "mx_nanmax",
                                    lambda x, ax: np.nanmax(x, axis=ax, **kwargs)))


def _single_axis_only(axis) -> bool:
    """mean/var/std are not associative across axes when the valid counts
    differ per slice, so a multi-axis tuple goes back to NumPy."""
    return not (isinstance(axis, tuple) and len(axis) > 1)


def nanmean(a, axis=None, **kwargs):
    if _uses_mojo(a) and _single_axis_only(axis):
        return _dispatch(a, axis, lambda x, ax: _ops.reduce_axis(x, "mx_nanmean", ax))
    return np.nanmean(a, axis=axis, **kwargs)


def nanmedian(a, axis=None, **kwargs):
    return _dispatch(a, axis, _pick(a, "mx_nanmedian",
                                    lambda x, ax: np.nanmedian(x, axis=ax, **kwargs)))


def nanvar(a, axis=None, ddof=0, **kwargs):
    if _uses_mojo(a) and _single_axis_only(axis):
        return _dispatch(a, axis,
                         lambda x, ax: _ops.reduce_axis(x, "mx_nanvar", ax, ddof=ddof))
    return np.nanvar(a, axis=axis, ddof=ddof, **kwargs)


def nanstd(a, axis=None, ddof=0, **kwargs):
    if _uses_mojo(a) and _single_axis_only(axis):
        return _dispatch(a, axis,
                         lambda x, ax: _ops.reduce_axis(x, "mx_nanstd", ax, ddof=ddof))
    return np.nanstd(a, axis=axis, ddof=ddof, **kwargs)


def nanprod(a, axis=None, **kwargs):
    return _dispatch(a, axis, _pick(a, "mx_nanprod",
                                    lambda x, ax: np.nanprod(x, axis=ax, **kwargs)))


def nanargmin(a, axis=None):
    # The kernel signals an all-NaN slice with NaN, the one value an index
    # cannot take; NumPy raises ValueError("All-NaN slice encountered").
    if _uses_mojo(a):
        out = _dispatch(a, axis, lambda x, ax: _ops.reduce_axis(x, "mx_nanargmin", ax))
        if np.isnan(out).any():
            raise ValueError("All-NaN slice encountered")
        return out.astype(np.intp)
    return _dispatch(a, axis, lambda x, ax: np.nanargmin(x, axis=ax))


def nanargmax(a, axis=None):
    if _uses_mojo(a):
        out = _dispatch(a, axis, lambda x, ax: _ops.reduce_axis(x, "mx_nanargmax", ax))
        if np.isnan(out).any():
            raise ValueError("All-NaN slice encountered")
        return out.astype(np.intp)
    return _dispatch(a, axis, lambda x, ax: np.nanargmax(x, axis=ax))


def nancumsum(a, axis=None, **kwargs):
    if _uses_mojo(a):
        return _ops.cumulative(a, "mx_nancumsum", axis)
    return np.nancumsum(a, axis=axis, **kwargs)


def nancumprod(a, axis=None, **kwargs):
    if _uses_mojo(a):
        return _ops.cumulative(a, "mx_nancumprod", axis)
    return np.nancumprod(a, axis=axis, **kwargs)


def nanquantile(a, q, axis=None, method="linear"):
    """`np.nanquantile` with `method="linear"`, which is what xarray asks for.

    numpy replaces the reduced axis with the q axis, and puts that axis first.
    """
    if method != "linear" or isinstance(axis, tuple) or not _uses_mojo(a):
        return np.nanquantile(a, q, axis=axis, method=method)
    qs = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if qs.size == 0 or np.any(qs < 0.0) or np.any(qs > 1.0):
        raise ValueError("q values must be in the range [0, 1]")
    scalar_q = np.ndim(q) == 0
    if axis is None:
        out = np.array([_ops.reduce_axis(a, "mx_nanquantile", None, q=qi)[()]
                        for qi in qs])
        return out.reshape(()) if scalar_q else out
    ax = _ops.normalize_axis_index(axis, a.ndim)
    cols = [_ops.reduce_axis(a, "mx_nanquantile", ax, q=qi) for qi in qs]
    out = np.stack(cols, axis=0)
    return out[0] if scalar_q else out
