"""Port of `xarray/computation/nanops.py`.

Upstream is a thin layer over `nputils`: it branches on object dtype, and for
`nansum`/`nanprod` it applies `_maybe_null_out` when a `min_count` was asked
for.  That structure is kept here verbatim; only the object-dtype branches are
dropped, since an object array has no Mojo analogue and no fast path upstream
either.
"""

from __future__ import annotations

import numpy as np

from . import _ops, nputils

__all__ = [
    "nanmin", "nanmax", "nanargmin", "nanargmax", "nansum", "nanmean",
    "nanmedian", "nanvar", "nanstd", "nanprod", "nancumsum", "nancumprod",
    "_maybe_null_out",
]


def nanmin(a, axis=None, out=None):
    return nputils.nanmin(a, axis=axis)


def nanmax(a, axis=None, out=None):
    return nputils.nanmax(a, axis=axis)


def nanargmin(a, axis=None):
    return nputils.nanargmin(a, axis=axis)


def nanargmax(a, axis=None):
    return nputils.nanargmax(a, axis=axis)


def nansum(a, axis=None, dtype=None, out=None, min_count=None):
    # mask = isnull(a)
    # result = sum_where(a, axis=axis, dtype=dtype, where=mask)
    result = _nansum(a, axis)
    if min_count is not None:
        return _maybe_null_out(result, axis, np.isnan(a), min_count)
    return result


def _nansum(a, axis=None):
    if nputils._uses_mojo(a):
        return nputils._dispatch(a, axis,
                                 lambda x, ax: _ops.reduce_axis(x, "mx_nansum", ax))
    return np.nansum(a, axis=axis)


def nanmean(a, axis=None, dtype=None, out=None):
    return nputils.nanmean(a, axis=axis)


def nanmedian(a, axis=None, out=None):
    # The dask algorithm works by rechunking to one chunk along axis; make sure
    # we trigger the dask error when passing all dimensions so that we don't
    # rechunk the entire array to one chunk and possibly blow memory.
    if axis is not None and len(np.atleast_1d(axis)) == a.ndim:
        axis = None
    return nputils.nanmedian(a, axis=axis)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0):
    return nputils.nanvar(a, axis=axis, ddof=ddof)


def nanstd(a, axis=None, dtype=None, out=None, ddof=0):
    return nputils.nanstd(a, axis=axis, ddof=ddof)


def nanprod(a, axis=None, dtype=None, out=None, min_count=None):
    # mask = isnull(a)
    # result = nputils.nanprod(a, axis=axis, dtype=dtype)
    result = nputils.nanprod(a, axis=axis)
    if min_count is not None:
        return _maybe_null_out(result, axis, np.isnan(a), min_count)
    return result


def nancumsum(a, axis=None, dtype=None, out=None):
    return nputils.nancumsum(a, axis=axis)


def nancumprod(a, axis=None, dtype=None, out=None):
    return nputils.nancumprod(a, axis=axis)


def _maybe_null_out(result, axis, mask, min_count=1):
    """xarray version of pandas.core.nanops._maybe_null_out.

    `mask` is `isnull(a)`.  For float data `dtypes.maybe_promote` hands back NaN
    as the fill value, and the null test is
    `take(mask.shape, axis).prod() - sum(mask, axis) - min_count < 0`.
    """
    return _ops.maybe_null_out(result, mask, axis, min_count)
