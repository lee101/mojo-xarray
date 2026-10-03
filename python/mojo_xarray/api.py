"""The labelled-array surface: `mojo_xarray.DataArray` and `mojo_xarray.Dataset`.

Both subclass their xarray counterparts, so a `mojo_xarray.DataArray` *is* an
`xarray.DataArray` -- it keeps dims, coords, indexes, attrs and name, and every
method this port does not override behaves exactly as upstream.  What is
overridden is the reduction and rolling machinery, and only where the underlying
array is a contiguous float64 NumPy array; anything else (dask, cupy, int,
object, `skipna=False`, ...) falls through to `super()`.

`Dataset` needs its own overrides for a reason that is not obvious from reading
it: upstream's `Dataset.reduce` hands each variable to `Variable.reduce` with
`duck_array_ops.<name>`, i.e. NumPy, and `Dataset.__getitem__` builds the
DataArray wrappers out of `xarray.core.dataarray.DataArray` itself.  So the
ninety per cent of `Dataset.sum` that is `DataArray.sum` never runs this port's
overrides.  Each override below keeps upstream's signature, its guard clauses and
its `self.reduce(duck_array_ops.<name>, ...)` call, and only substitutes the
callable -- which is the same substitution `inject_reduce_methods` performs for
the `Coarsen` classes.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Sequence

import numpy as np
import xarray as xr
from xarray.core import utils as xr_utils

from . import nanops, nputils
from .rolling import DataArrayCoarsen, DataArrayRolling, DatasetCoarsen, DatasetRolling

__all__ = ["DataArray", "Dataset"]

# `Dataset.<reduction>` passes the function to `Dataset.reduce` under the name
# `duck_array_ops` gives it.  Each is `xarray.computation.nanops.nan<name>` when
# the dtype has a missing-value sentinel and NumPy otherwise, which is the
# dispatch `_create_nan_agg_method` does.
#
# `min_count` is only accepted where upstream accepts it.  `_create_nan_agg_method`
# pops it before calling NumPy for `sum`/`prod` on a dtype that has no NaN, and
# `_maybe_null_out` only ever sees it on the `nan*` branch; `DataArray.sum`
# raises for a negative `min_count` and `DataArray.prod` routes a `min_count` to
# `super()` because `duck_array_ops.prod` has no `available_min_count`.
_MOJO_AGG = {
    "max": nanops.nanmax,
    "min": nanops.nanmin,
    "mean": nanops.nanmean,
    "prod": nanops.nanprod,
    "sum": nanops.nansum,
    "std": nanops.nanstd,
    "var": nanops.nanvar,
    "median": nanops.nanmedian,
    # `_create_nan_agg_method` has no `nan` prefix for these two: the name is
    # already the positional one, and the kernel that implements them is
    # `mx_nanargmin` / `mx_nanargmax`.
    "argmin": nanops.nanargmin,
    "argmax": nanops.nanargmax,
}

# `Dataset.<name>`'s own `numeric_only`, which decides whether a non-numeric
# variable survives the reduction.  Upstream passes `False` for `min`/`max` and
# `True` for everything else.
_NUMERIC_ONLY = {name: name not in ("min", "max") for name in _MOJO_AGG}


def _float64_ndarray(obj) -> bool:
    data = obj.variable.data
    return isinstance(data, np.ndarray) and data.dtype == nputils.USES_MOJO


def _agg(name, values, axis=None, skipna=None, **kwargs):
    """`duck_array_ops.<name>`, i.e. `_create_nan_agg_method`, with the Mojo
    kernel in place of `nanops.nan<name>`.

    Upstream's order, unchanged: `out` is rejected outright; an `invariant_0d`
    name asked for an empty tuple of axes returns the data untouched, so neither
    the dtype nor the handling of a scalar changes; strings are coerced to object
    for the four `coerce_strings` names; then the `skipna` test picks the `nan*`
    branch or `getattr(xp, name)`.  Only the assignment of `func` differs, and
    every dtype without a kernel -- and every `skipna=False` -- lands on the same
    NumPy function upstream would have called.
    """
    if kwargs.pop("out", None) is not None:
        raise TypeError(f"`out` is not valid for {name}")

    # See https://github.com/pydata/xarray/issues/4885
    if axis == () and name in ("max", "min", "sum", "median", "prod"):
        return values

    xp = np
    if name in ("max", "min", "argmin", "argmax") and values.dtype.kind in ("U", "S"):
        values = values.astype(object)

    func = None
    if skipna or (
        skipna is None
        and (
            np.issubdtype(values.dtype, np.complexfloating)
            or np.issubdtype(values.dtype, np.floating)
            or values.dtype == object
        )
    ):
        func = _MOJO_AGG[name]

    if func is None:
        if name in ("sum", "prod"):
            kwargs.pop("min_count", None)
        func = getattr(xp, name)

    return func(values, axis=axis, **kwargs)


def _wants_arg_dict(dim) -> bool:
    """`Variable._unravel_argminmax` returns a dict for a sequence of dims, and
    that path goes through `stack`/`unravel_index`; leave it to upstream."""
    return dim is ... or (isinstance(dim, Sequence) and not isinstance(dim, str))


class DataArray(xr.DataArray):
    """`xarray.DataArray` with the reductions, `rolling` and `coarsen` running
    in Mojo wherever the data is a contiguous float64 NumPy array."""

    __slots__ = ()

    # -- rolling / coarsen ------------------------------------------------

    def rolling(
        self,
        dim=None,
        min_periods: int | None = None,
        center=False,
        **window_kwargs: int,
    ) -> DataArrayRolling:
        """Rolling window object for DataArrays.

        See `xarray.DataArray.rolling`; this returns a
        `mojo_xarray.rolling.DataArrayRolling` whose reductions run in Mojo.
        """
        windows = xr_utils.either_dict_or_kwargs(dim, window_kwargs, "rolling")
        return DataArrayRolling(self, windows, min_periods=min_periods, center=center)

    def coarsen(
        self,
        dim=None,
        boundary: str = "exact",
        side: str = "left",
        coord_func="mean",
        **window_kwargs: int,
    ) -> DataArrayCoarsen:
        """Coarsen object for DataArrays.

        See `xarray.DataArray.coarsen`.
        """
        windows = xr_utils.either_dict_or_kwargs(dim, window_kwargs, "coarsen")
        return DataArrayCoarsen(self, windows, boundary, side, coord_func)

    # -- reductions -------------------------------------------------------

    def sum(self, dim=None, *, skipna=None, min_count=None, keep_attrs=None, **kwargs):
        if not _float64_ndarray(self) or skipna is False:
            return super().sum(dim=dim, skipna=skipna, min_count=min_count,
                               keep_attrs=keep_attrs, **kwargs)
        if min_count is None:
            return self.reduce(nanops.nansum, dim=dim, keep_attrs=keep_attrs, **kwargs)
        if min_count < 0:
            raise ValueError("min_count must be greater than or equal to 0")
        return self.reduce(_kw(nanops.nansum, min_count=min_count), dim=dim,
                           keep_attrs=keep_attrs, **kwargs)

    def mean(self, dim=None, *, skipna=None, keep_attrs=None, **kwargs):
        if not _float64_ndarray(self) or skipna is False:
            return super().mean(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)
        return self.reduce(nanops.nanmean, dim=dim, keep_attrs=keep_attrs, **kwargs)

    def min(self, dim=None, *, skipna=None, keep_attrs=None, **kwargs):
        if not _float64_ndarray(self) or skipna is False:
            return super().min(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)
        return self.reduce(nanops.nanmin, dim=dim, keep_attrs=keep_attrs, **kwargs)

    def max(self, dim=None, *, skipna=None, keep_attrs=None, **kwargs):
        if not _float64_ndarray(self) or skipna is False:
            return super().max(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)
        return self.reduce(nanops.nanmax, dim=dim, keep_attrs=keep_attrs, **kwargs)

    def std(self, dim=None, *, skipna=None, ddof=0, keep_attrs=None, **kwargs):
        if not _float64_ndarray(self) or skipna is False:
            return super().std(dim=dim, skipna=skipna, ddof=ddof,
                               keep_attrs=keep_attrs, **kwargs)
        return self.reduce(_kw(nanops.nanstd, ddof=ddof), dim=dim,
                           keep_attrs=keep_attrs, **kwargs)

    def var(self, dim=None, *, skipna=None, ddof=0, keep_attrs=None, **kwargs):
        if not _float64_ndarray(self) or skipna is False:
            return super().var(dim=dim, skipna=skipna, ddof=ddof,
                               keep_attrs=keep_attrs, **kwargs)
        return self.reduce(_kw(nanops.nanvar, ddof=ddof), dim=dim,
                           keep_attrs=keep_attrs, **kwargs)

    def median(self, dim=None, *, skipna=None, keep_attrs=None, **kwargs):
        if not _float64_ndarray(self) or skipna is False:
            return super().median(dim=dim, skipna=skipna, keep_attrs=keep_attrs,
                                  **kwargs)
        return self.reduce(nanops.nanmedian, dim=dim, keep_attrs=keep_attrs, **kwargs)

    def prod(self, dim=None, *, skipna=None, min_count=None, keep_attrs=None, **kwargs):
        if not _float64_ndarray(self) or skipna is False or min_count is not None:
            return super().prod(dim=dim, skipna=skipna, min_count=min_count,
                                keep_attrs=keep_attrs, **kwargs)
        return self.reduce(nanops.nanprod, dim=dim, keep_attrs=keep_attrs, **kwargs)

    def argmin(self, dim=None, *, axis=None, keep_attrs=None, skipna=None):
        return self._argminmax("argmin", dim, axis, keep_attrs, skipna)

    def argmax(self, dim=None, *, axis=None, keep_attrs=None, skipna=None):
        return self._argminmax("argmax", dim, axis, keep_attrs, skipna)

    def _argminmax(self, name, dim, axis, keep_attrs, skipna):
        """`DataArray.argmin` / `argmax`, with `duck_array_ops.{argmin,argmax}`
        swapped for the Mojo kernel.  The dict-of-dims form keeps upstream's
        `stack`/`unravel_index` machinery and is not on the fast path."""
        func = getattr(nanops, "nan" + name)
        if not _float64_ndarray(self) or skipna is False or _wants_arg_dict(dim):
            return getattr(super(), name)(dim, axis=axis, keep_attrs=keep_attrs,
                                          skipna=skipna)
        if dim is None and axis is None:
            warnings.warn(
                "Behaviour of argmin/argmax with neither dim nor axis argument "
                "will change to return a dict of indices of each dimension. To "
                "get a single, flat index, please use np.argmin(da.data) or "
                "np.argmax(da.data) instead of da.argmin() or da.argmax().",
                FutureWarning,
                stacklevel=3,
            )
        # `duck_array_ops.argmin` consumes `skipna` in its own dispatch and only
        # then calls `nanops.nanargmin`, so the kernel must not see it
        result = self.variable.reduce(func, dim=dim, axis=axis,
                                      keep_attrs=keep_attrs)
        return self._replace_maybe_drop_dims(result)


def _kw(func, **fixed):
    """Bind the reduction's extra arguments the way `DataArray.reduce` would."""
    def inner(a, axis=None, **kwargs):
        for k, v in fixed.items():
            kwargs[k] = v
        return func(a, axis=axis, **kwargs)

    inner.__name__ = func.__name__
    return inner


class Dataset(xr.Dataset):
    """`xarray.Dataset` with the reductions, `rolling` and `coarsen` running in
    Mojo wherever a variable's data is a contiguous float64 NumPy array."""

    __slots__ = ()

    def rolling(
        self,
        dim=None,
        min_periods: int | None = None,
        center=False,
        **window_kwargs: int,
    ) -> DatasetRolling:
        """Rolling window object for Datasets."""
        windows = xr_utils.either_dict_or_kwargs(dim, window_kwargs, "rolling")
        return DatasetRolling(self, windows, min_periods=min_periods, center=center)

    def coarsen(
        self,
        dim=None,
        boundary: str = "exact",
        side: str = "left",
        coord_func="mean",
        **window_kwargs: int,
    ) -> DatasetCoarsen:
        """Coarsen object for Datasets."""
        windows = xr_utils.either_dict_or_kwargs(dim, window_kwargs, "coarsen")
        return DatasetCoarsen(self, windows, boundary, side, coord_func)

    def argmin(self, dim=None, **kwargs):
        """`Dataset.argmin`, with `duck_array_ops.argmin` swapped for the Mojo
        kernel.

        Upstream's dict-of-dims branch raises before reaching the reduction, and
        the flat-index branch -- `dim=None` or a single dim -- is the one that
        runs `self.reduce(duck_array_ops.argmin, dim=[dim])` on a `Variable`
        rather than on a DataArray, which is the path that needed overriding.
        """
        if _wants_arg_dict(dim):
            return super().argmin(dim=dim, **kwargs)
        return self.reduce(functools.partial(_agg, "argmin"),
                           dim=None if dim is None else [dim], **kwargs)

    def argmax(self, dim=None, **kwargs):
        """`Dataset.argmax`, with `duck_array_ops.argmax` swapped for the Mojo
        kernel.  See `argmin`."""
        if _wants_arg_dict(dim):
            return super().argmax(dim=dim, **kwargs)
        return self.reduce(functools.partial(_agg, "argmax"),
                           dim=None if dim is None else [dim], **kwargs)


def _dataset_reduction(name):
    """Build a `Dataset.<name>` override with upstream's exact signature.

    The body is upstream's, in upstream's order: the same `self.reduce(
    duck_array_ops.<name>, dim=dim, skipna=..., numeric_only=..., keep_attrs=...,
    **kwargs)` call.  Only the function handed to `reduce` differs -- `_agg`
    above, which is `_create_nan_agg_method` with the Mojo kernel in place of
    `nanops.nan<name>`, so `skipna=False`, a 0-d reduction and every dtype
    without a kernel behave exactly as upstream.

    The keyword order and the `numeric_only` value differ per name: `min`/`max`
    pass `numeric_only=False`, everything else `True`.
    """
    upstream = getattr(xr.Dataset, name)
    func = functools.partial(_agg, name)

    def reduction(self, dim=None, *, skipna=None, keep_attrs=None, **kwargs):
        out = self.reduce(
            func,
            dim=dim,
            skipna=skipna,
            numeric_only=_NUMERIC_ONLY[name],
            keep_attrs=keep_attrs,
            **kwargs,
        )
        return out

    if name in ("sum", "prod"):
        def reduction(self, dim=None, *, skipna=None, min_count=None,
                      keep_attrs=None, **kwargs):  # noqa: F811
            out = self.reduce(
                func,
                dim=dim,
                skipna=skipna,
                min_count=min_count,
                numeric_only=True,
                keep_attrs=keep_attrs,
                **kwargs,
            )
            return out

    elif name in ("std", "var"):
        def reduction(self, dim=None, *, skipna=None, ddof=0,  # noqa: F811
                      keep_attrs=None, **kwargs):
            out = self.reduce(
                func,
                dim=dim,
                skipna=skipna,
                ddof=ddof,
                numeric_only=True,
                keep_attrs=keep_attrs,
                **kwargs,
            )
            return out

    reduction.__name__ = name
    reduction.__qualname__ = f"Dataset.{name}"
    reduction.__doc__ = upstream.__doc__
    return reduction


# `Dataset.argmin` / `Dataset.argmax` keep the signature written out above --
# `(dim=None, **kwargs)`, with no `skipna`/`keep_attrs` keywords and the
# dict-of-dims branch raising upstream's `ValueError` before any reduction runs
# -- so they are installed here rather than through `_dataset_reduction`.
for _name in _MOJO_AGG:
    if _name in ("argmin", "argmax"):
        continue
    setattr(Dataset, _name, _dataset_reduction(_name))
