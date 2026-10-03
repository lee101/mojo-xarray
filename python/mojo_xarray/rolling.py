"""Port of `xarray/computation/rolling.py`.

Same classes, same method names, same argument order and defaults as upstream.
The reduction methods keep upstream's three-way dispatch -- a moving-window
kernel, a `rolling_agg_func`, or a materialised-window `reduce` -- with a Mojo
kernel standing in for `bottleneck.move_*`.

Two divergences, both documented in the README:

  * `argmin`/`argmax` return an offset from the window's left edge, which is
    what the materialised-window path produces.  Upstream's bottleneck fast
    path returns right-edge-relative indices for full windows instead
    (`values = self.window[0] - 1 - values`, GH #8541), and reproducing that
    would mean reproducing a quirk that only exists when bottleneck is
    installed.
  * `prod` and any reduction that falls off the fast path build the window
    through the Mojo `pad` kernel and NumPy's `sliding_window_view`, exactly as
    `Variable.pad` then `duck_array_ops.sliding_window_view` does upstream.
"""

from __future__ import annotations

import functools
import itertools
import math
import warnings
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any, Generic, TypeVar

import numpy as np
import xarray as xr
from xarray.core import duck_array_ops, utils as xr_utils
from xarray.core.dtypes import INF, NINF, NA
from xarray.core.options import _get_keep_attrs

from . import _ops, nanops, nputils

__all__ = [
    "Rolling", "DataArrayRolling", "DatasetRolling",
    "Coarsen", "DataArrayCoarsen", "DatasetCoarsen",
]

T_Xarray = TypeVar("T_Xarray")

_ROLLING_REDUCE_DOCSTRING_TEMPLATE = """\
Reduce this object's data windows by applying `{name}` along its dimension.

Parameters
----------
keep_attrs : bool, default: None
    If True, the attributes (``attrs``) will be copied from the original
    object to the new one. If False, the new object will be returned
    without attributes. If None uses the global default.
**kwargs : dict
    Additional keyword arguments passed on to `{name}`.

Returns
-------
reduced : same type as caller
    New object with `{name}` applied along its rolling dimension.
"""

# The name of the Mojo kernel standing in for `bottleneck.move_<name>`, the
# same set of names bottleneck offers.
_MOVE_KERNELS = {
    "sum": "mx_move_sum",
    "mean": "mx_move_mean",
    "min": "mx_move_min",
    "max": "mx_move_max",
    "median": "mx_move_median",
    "std": "mx_move_std",
    "var": "mx_move_var",
    "argmin": "mx_move_argmin",
    "argmax": "mx_move_argmax",
}


def _is_movable(obj, ndim: int) -> bool:
    """The moving-window kernels cover a contiguous float64 array rolled along
    one dimension, which is exactly what bottleneck and numbagg accept too."""
    data = obj.variable.data
    return (
        ndim == 1
        and isinstance(data, np.ndarray)
        and data.dtype == nputils.USES_MOJO
    )


def _axis_last(obj, dim):
    """`np.moveaxis` the rolling dimension last; return the view and its axis."""
    axis = obj.get_axis_num(dim)
    return np.ascontiguousarray(np.moveaxis(obj.variable.data, axis, -1)), axis


def _restore_axis(values, axis, ndim):
    return np.moveaxis(values, -1, axis) if axis != ndim - 1 else values


class Rolling(Generic[T_Xarray]):
    """A object that implements the moving window pattern."""

    __slots__ = ("center", "dim", "min_periods", "obj", "window")
    _attributes = ("window", "min_periods", "center", "dim")
    dim: list[Hashable]
    window: list[int]
    center: list[bool]
    obj: T_Xarray
    min_periods: int

    def __init__(
        self,
        obj: T_Xarray,
        windows: Mapping[Any, int],
        min_periods: int | None = None,
        center: bool | Mapping[Any, bool] = False,
    ) -> None:
        self.dim = []
        self.window = []
        for d, w in windows.items():
            self.dim.append(d)
            if w <= 0:
                raise ValueError("window must be > 0")
            self.window.append(w)

        self.center = self._mapping_to_list(center, default=False)
        self.obj = obj

        missing_dims = tuple(dim for dim in self.dim if dim not in self.obj.dims)
        if missing_dims:
            # NOTE: we raise KeyError here but ValueError in Coarsen.
            raise KeyError(
                f"Window dimensions {missing_dims} not found in "
                f"{self.obj.__class__.__name__} dimensions {tuple(self.obj.dims)}"
            )

        if min_periods is not None and min_periods <= 0:
            raise ValueError("min_periods must be greater than zero or None")

        self.min_periods = (
            math.prod(self.window) if min_periods is None else min_periods
        )

    def __repr__(self) -> str:
        """provide a nice str repr of our rolling object"""
        attrs = ",".join(
            f"{k}->{w}{'(center)' if c else ''}"
            for k, w, c in zip(self.dim, self.window, self.center, strict=True)
        )
        return f"{self.__class__.__name__} [{attrs}]"

    def __len__(self) -> int:
        return math.prod(self.obj.sizes[d] for d in self.dim)

    @property
    def ndim(self) -> int:
        return len(self.dim)

    def _reduce_method(
        name: str,
        fillna: Any,
        rolling_agg_func: Callable | None = None,
        automatic_rechunk: bool = False,
    ) -> Callable[..., T_Xarray]:
        """Constructs reduction methods built on a moving-window reduction (e.g.
        `move_sum`), a Rolling reduction (`_mean`), or a materialised numpy
        reduction (e.g. sum).

        Set automatic_rechunk=True when the reduction method makes a memory copy.
        """
        if rolling_agg_func:
            array_agg_func = None
        else:
            array_agg_func = getattr(duck_array_ops, name)

        def method(self, keep_attrs=None, **kwargs):
            keep_attrs = self._get_keep_attrs(keep_attrs)
            return self._array_reduce(
                name=name,
                array_agg_func=array_agg_func,
                rolling_agg_func=rolling_agg_func,
                keep_attrs=keep_attrs,
                fillna=fillna,
                **kwargs,
            )

        method.__name__ = name
        method.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name=name)
        return method

    def _mean(self, keep_attrs, **kwargs):
        result = self.sum(keep_attrs=False, **kwargs)
        # use dtype of result for casting of count; this allows for GH #7062 and
        # GH #8864, fixes GH #10340
        result /= duck_array_ops.astype(
            self.count(keep_attrs=False), dtype=result.dtype, copy=False
        )
        if keep_attrs:
            result.attrs = self.obj.attrs
        return result

    _mean.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name="mean")

    # automatic_rechunk is set to True for reductions that make a copy.
    # std, var could be optimized after which we can set it to False
    argmax = _reduce_method("argmax", NINF, automatic_rechunk=True)
    argmin = _reduce_method("argmin", INF, automatic_rechunk=True)
    max = _reduce_method("max", NINF)
    min = _reduce_method("min", INF)
    prod = _reduce_method("prod", 1)
    sum = _reduce_method("sum", 0)
    mean = _reduce_method("mean", None, _mean)
    std = _reduce_method("std", None, automatic_rechunk=True)
    var = _reduce_method("var", None, automatic_rechunk=True)
    median = _reduce_method("median", None, automatic_rechunk=True)

    def _counts(self, keep_attrs: bool | None) -> T_Xarray:
        raise NotImplementedError()

    def count(self, keep_attrs: bool | None = None) -> T_Xarray:
        keep_attrs = self._get_keep_attrs(keep_attrs)
        rolling_count = self._counts(keep_attrs=keep_attrs)
        enough_periods = rolling_count >= self.min_periods
        return rolling_count.where(enough_periods)

    count.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name="count")

    def _mapping_to_list(
        self,
        arg,
        default=None,
        allow_default: bool = True,
        allow_allsame: bool = True,
    ) -> list:
        if xr_utils.is_dict_like(arg):
            if allow_default:
                return [arg.get(d, default) for d in self.dim]
            for d in self.dim:
                if d not in arg:
                    raise KeyError(f"Argument has no dimension key {d}.")
            return [arg[d] for d in self.dim]
        if allow_allsame:  # for single argument
            return [arg] * self.ndim
        if self.ndim == 1:
            return [arg]
        raise ValueError(f"Mapping argument is necessary for {self.ndim}d-rolling.")

    def _get_keep_attrs(self, keep_attrs):
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        return keep_attrs


class DataArrayRolling(Rolling["xr.DataArray"]):
    __slots__ = ("window_labels",)

    def __init__(
        self,
        obj: xr.DataArray,
        windows: Mapping[Any, int],
        min_periods: int | None = None,
        center: bool | Mapping[Any, bool] = False,
    ) -> None:
        super().__init__(obj, windows, min_periods=min_periods, center=center)
        # TODO legacy attribute
        self.window_labels = self.obj[self.dim[0]]

    def __iter__(self) -> Iterator[tuple[xr.DataArray, xr.DataArray]]:
        if self.ndim > 1:
            raise ValueError("__iter__ is only supported for 1d-rolling")

        dim0 = self.dim[0]
        window0 = int(self.window[0])
        offset = (window0 + 1) // 2 if self.center[0] else 1
        stops = np.arange(offset, self.obj.sizes[dim0] + offset)
        starts = stops - window0
        starts[: window0 - offset] = 0

        for label, start, stop in zip(self.window_labels, starts, stops, strict=True):
            window = self.obj.isel({dim0: slice(start, stop)})
            counts = window.count(dim=[dim0])
            window = window.where(counts >= self.min_periods)
            yield (label, window)

    def construct(
        self,
        window_dim: Hashable | Mapping[Any, Hashable] | None = None,
        *,
        stride: int | Mapping[Any, int] = 1,
        fill_value: Any = NA,
        keep_attrs: bool | None = None,
        sliding_window_view_kwargs: Mapping[Any, Any] | None = None,
        **window_dim_kwargs: Hashable,
    ) -> xr.DataArray:
        """Convert this rolling object to xr.DataArray,
        where the window dimension is stacked as a new dimension.
        """
        if sliding_window_view_kwargs is None:
            sliding_window_view_kwargs = {}
        return self._construct(
            self.obj,
            window_dim=window_dim,
            stride=stride,
            fill_value=fill_value,
            keep_attrs=keep_attrs,
            sliding_window_view_kwargs=sliding_window_view_kwargs,
            **window_dim_kwargs,
        )

    def _construct(
        self,
        obj: xr.DataArray,
        *,
        window_dim: Hashable | Mapping[Any, Hashable] | None = None,
        stride: int | Mapping[Any, int] = 1,
        fill_value: Any = NA,
        keep_attrs: bool | None = None,
        sliding_window_view_kwargs: Mapping[Any, Any] | None = None,
        **window_dim_kwargs: Hashable,
    ) -> xr.DataArray:
        if sliding_window_view_kwargs is None:
            sliding_window_view_kwargs = {}

        keep_attrs = self._get_keep_attrs(keep_attrs)

        if window_dim is None:
            if len(window_dim_kwargs) == 0:
                raise ValueError(
                    "Either window_dim or window_dim_kwargs need to be specified."
                )
            window_dim = {d: window_dim_kwargs[str(d)] for d in self.dim}

        window_dims = self._mapping_to_list(
            window_dim, allow_default=False, allow_allsame=False
        )
        strides = self._mapping_to_list(stride, default=1)

        window = _rolling_window(
            obj.variable, self.dim, self.window, self.center, window_dims,
            fill_value, sliding_window_view_kwargs,
        )

        attrs = obj.attrs if keep_attrs else {}
        result = xr.DataArray(
            window,
            dims=obj.dims + tuple(window_dims),
            coords=obj.coords,
            attrs=attrs,
            name=obj.name,
        )
        return result.isel(
            {d: slice(None, None, s) for d, s in zip(self.dim, strides, strict=True)}
        )

    def reduce(
        self,
        func: Callable,
        keep_attrs: bool | None = None,
        *,
        sliding_window_view_kwargs: Mapping[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Reduce each window by applying `func`.

        Equivalent to ``.construct(...).reduce(func, ...)``.
        """
        keep_attrs = self._get_keep_attrs(keep_attrs)

        rolling_dim = {
            d: xr_utils.get_temp_dimname(self.obj.dims, f"_rolling_dim_{d}")
            for d in self.dim
        }

        # save memory with reductions GH4325
        fillna = kwargs.pop("fillna", NA)
        if fillna is not NA:
            obj = self.obj.fillna(fillna)
        else:
            obj = self.obj
        windows = self._construct(
            obj,
            window_dim=rolling_dim,
            keep_attrs=keep_attrs,
            fill_value=fillna,
            sliding_window_view_kwargs=sliding_window_view_kwargs,
        )

        dim = list(rolling_dim.values())
        result = windows.reduce(func, dim=dim, keep_attrs=keep_attrs, **kwargs)

        # Find valid windows based on count.
        counts = self._counts(keep_attrs=False)
        return result.where(counts >= self.min_periods)

    def _counts(self, keep_attrs: bool | None) -> xr.DataArray:
        """Number of non-nan entries in each rolling window."""
        if _is_movable(self.obj, self.ndim):
            data, axis = _axis_last(self.obj, self.dim[0])
            # We use 0 rather than NaN for the padding, so padded positions
            # never count, which is what `construct(fill_value=False)` does.
            mask = (~np.isnan(data)).astype(np.float64)
            counts = _ops.move(mask, "mx_move_count", self.window[0], 0,
                               self.center[0], n_rows=int(np.prod(data.shape[:-1])))
            counts = _restore_axis(counts.reshape(data.shape), axis, data.ndim)
            return _rebuild(self.obj, counts, keep_attrs)

        rolling_dim = {
            d: xr_utils.get_temp_dimname(self.obj.dims, f"_rolling_dim_{d}")
            for d in self.dim
        }
        # We use False as the fill_value instead of np.nan, since boolean array
        # is faster to be reduced than object array.
        # The use of skipna==False is also faster since it does not need to
        # copy the strided array.
        dim = list(rolling_dim.values())
        counts = (
            self.obj.notnull(keep_attrs=keep_attrs)
            .rolling(
                dict(zip(self.dim, self.window, strict=True)),
                center={d: self.center[i] for i, d in enumerate(self.dim)},
            )
            .construct(rolling_dim, fill_value=False, keep_attrs=keep_attrs)
            .sum(dim=dim, skipna=False, keep_attrs=keep_attrs)
        )
        return counts

    def _move_reduce(self, name, keep_attrs, ddof=0):
        """The Mojo counterpart of `Rolling._bottleneck_reduce`.

        The kernel pads each window itself, so there is no `shift`, no `valid`
        slice and no `min_count` clamp: the kernel's own count gate *is*
        upstream's trailing `result.where(counts >= self.min_periods)`.
        """
        data, axis = _axis_last(self.obj, self.dim[0])
        n = data.shape[-1]
        # bottleneck doesn't allow min_count to be 0, although it should work
        # the same as if min_count = 1
        if self.min_periods == 0:
            min_count = 1
        else:
            min_count = self.min_periods
        values = _ops.move(data, name, self.window[0], min_count, self.center[0],
                           ddof=ddof, n_rows=int(np.prod(data.shape[:-1])))
        values = _restore_axis(values.reshape(data.shape), axis, data.ndim)
        return _rebuild(self.obj, values, keep_attrs)

    def _array_reduce(
        self, name, array_agg_func, rolling_agg_func, keep_attrs, fillna, **kwargs
    ):
        if "dim" in kwargs:
            warnings.warn(
                f"Reductions are applied along the rolling dimension(s) "
                f"'{self.dim}'. Passing the 'dim' kwarg to reduction "
                f"operations has no effect.",
                FutureWarning,
                stacklevel=3,
            )
            del kwargs["dim"]

        move_name = _MOVE_KERNELS.get(name)
        if (
            move_name is not None
            and _is_movable(self.obj, self.ndim)
            and kwargs.get("skipna") is not False
        ):
            return self._move_reduce(move_name, keep_attrs, ddof=kwargs.get("ddof", 0))

        if rolling_agg_func:
            return rolling_agg_func(self, keep_attrs=self._get_keep_attrs(keep_attrs))

        if fillna is not None:
            if fillna is INF:
                fillna = np.inf
            elif fillna is NINF:
                fillna = -np.inf
            kwargs.setdefault("skipna", False)
            kwargs.setdefault("fillna", fillna)

        return self.reduce(array_agg_func, keep_attrs=keep_attrs, **kwargs)


def _rebuild(obj, values, keep_attrs):
    """Re-wrap `values` with `obj`'s dims, coords, name and (optionally) attrs."""
    attrs = obj.attrs if keep_attrs else {}
    return obj.__class__(values, obj.coords, dims=obj.dims, attrs=attrs, name=obj.name)


def _rolling_window(var, dims, window, center, window_dims, fill_value, kwargs):
    """`core/variable.py: Variable.rolling_window` -- pad exactly as upstream
    does, then take a sliding-window view of the padded array.

    `pad` is the Mojo kernel only for a C-contiguous float64 NumPy array, which
    is the buffer the FFI is written against.  Anything else -- dask, cupy,
    float32, an integer or object dtype -- is handed straight to upstream:
    `dtypes.maybe_promote` keeps a float32 array in float32 (and promotes bool
    to object), so padding through float64 here would silently change the
    result's dtype, and converting a lazy array to NumPy would compute it.
    """
    data = var.data
    if not (isinstance(data, np.ndarray) and data.dtype == nputils.USES_MOJO
            and data.flags["C_CONTIGUOUS"]):
        return var.rolling_window(
            dims, window, window_dims, center=center, fill_value=fill_value,
            **kwargs,
        )

    if fill_value is NA:  # np.nan is passed
        # dtypes.maybe_promote(float64) is float64 with NaN as the fill, and
        # astype is a no-op on the array we already have.
        fill_value = np.nan

    pads = {}
    for d, win, cent in zip(dims, window, center, strict=True):
        if cent:
            start = win // 2  # 10 -> 5,  9 -> 4
            end = win - 1 - start
            pads[d] = (start, end)
        else:
            pads[d] = (win - 1, 0)

    axis = list(var.dims).index(dims[0]) if len(dims) == 1 else \
        tuple(list(var.dims).index(d) for d in dims)
    pad_width = [pads.get(d, (0, 0)) for d in var.dims]
    if any(p != (0, 0) for p in pad_width):
        data = _ops.pad(data, pad_width, float(fill_value))
    view = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=tuple(window), axis=axis, **kwargs
    )
    return xr.Variable(tuple(var.dims) + tuple(window_dims), view)


class DatasetRolling(Rolling["xr.Dataset"]):
    __slots__ = ("rollings",)

    def __init__(
        self,
        obj: xr.Dataset,
        windows: Mapping[Any, int],
        min_periods: int | None = None,
        center: bool | Mapping[Any, bool] = False,
    ) -> None:
        super().__init__(obj, windows, min_periods, center)

        # Keep each Rolling object as a dictionary
        self.rollings = {}
        for key, da in self.obj.data_vars.items():
            # keeps rollings only for the dataset depending on self.dim
            dims, ctr = [], {}
            for i, d in enumerate(self.dim):
                if d in da.dims:
                    dims.append(d)
                    ctr[d] = self.center[i]

            if dims:
                w = {d: windows[d] for d in dims}
                self.rollings[key] = DataArrayRolling(da, w, min_periods, ctr)

    def _dataset_implementation(self, func, keep_attrs, **kwargs) -> xr.Dataset:
        keep_attrs = self._get_keep_attrs(keep_attrs)

        reduced = {}
        for key, da in self.obj.data_vars.items():
            if any(d in da.dims for d in self.dim):
                reduced[key] = func(self.rollings[key], keep_attrs=keep_attrs, **kwargs)
            else:
                reduced[key] = self.obj[key].copy()
                # we need to delete the attrs of the copied DataArray
                if not keep_attrs:
                    reduced[key].attrs = {}

        attrs = self.obj.attrs if keep_attrs else {}
        return xr.Dataset(reduced, coords=self.obj.coords, attrs=attrs)

    def reduce(
        self,
        func: Callable,
        keep_attrs: bool | None = None,
        sliding_window_view_kwargs: Mapping[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Reduce the items in this group by applying `func` along some
        dimension(s).
        """
        return self._dataset_implementation(
            functools.partial(DataArrayRolling.reduce, func=func),
            keep_attrs=keep_attrs,
            sliding_window_view_kwargs=sliding_window_view_kwargs,
            **kwargs,
        )

    def _counts(self, keep_attrs: bool | None) -> xr.Dataset:
        return self._dataset_implementation(
            DataArrayRolling._counts, keep_attrs=keep_attrs
        )

    def _array_reduce(
        self, name, array_agg_func, rolling_agg_func, keep_attrs, fillna, **kwargs
    ) -> xr.Dataset:
        return self._dataset_implementation(
            functools.partial(
                DataArrayRolling._array_reduce,
                name=name,
                array_agg_func=array_agg_func,
                rolling_agg_func=rolling_agg_func,
            ),
            keep_attrs=keep_attrs,
            fillna=fillna,
            **kwargs,
        )

    def construct(
        self,
        window_dim=None,
        *,
        stride: int | Mapping[Any, int] = 1,
        fill_value: Any = NA,
        keep_attrs: bool | None = None,
        sliding_window_view_kwargs: Mapping[Any, Any] | None = None,
        **window_dim_kwargs: Hashable,
    ) -> xr.Dataset:
        """Convert this rolling object to xr.Dataset,
        where the window dimension is stacked as a new dimension
        """
        keep_attrs = self._get_keep_attrs(keep_attrs)
        if window_dim is None:
            if len(window_dim_kwargs) == 0:
                raise ValueError(
                    "Either window_dim or window_dim_kwargs need to be specified."
                )
            window_dim = {d: window_dim_kwargs[str(d)] for d in self.dim}

        window_dims = self._mapping_to_list(
            window_dim, allow_default=False, allow_allsame=False
        )
        strides = self._mapping_to_list(stride, default=1)

        dataset = {}
        for key, da in self.obj.data_vars.items():
            # keeps rollings only for the dataset depending on self.dim
            dims = [d for d in self.dim if d in da.dims]
            if dims:
                wi = {d: window_dims[i] for i, d in enumerate(self.dim) if d in da.dims}
                st = {d: strides[i] for i, d in enumerate(self.dim) if d in da.dims}

                dataset[key] = self.rollings[key].construct(
                    window_dim=wi,
                    fill_value=fill_value,
                    stride=st,
                    keep_attrs=keep_attrs,
                    sliding_window_view_kwargs=sliding_window_view_kwargs,
                )
            else:
                dataset[key] = da.copy()

            # as the DataArrays can be copied we need to delete the attrs
            if not keep_attrs:
                dataset[key].attrs = {}

        # Need to stride coords as well. TODO: is there a better way?
        coords = self.obj.isel(
            {d: slice(None, None, s) for d, s in zip(self.dim, strides, strict=True)}
        ).coords

        attrs = self.obj.attrs if keep_attrs else {}

        return xr.Dataset(dataset, coords=coords, attrs=attrs)


def _coarsen_reshape(var, windows, boundary, side):
    """`core/variable.py: Variable.coarsen_reshape` -- trim or pad each window
    dimension, then split it into (n_blocks, block).  Returns the reshaped
    array and the axes the reduction runs over."""
    if not xr_utils.is_dict_like(boundary):
        boundary = dict.fromkeys(windows.keys(), boundary)
    if not xr_utils.is_dict_like(side):
        side = dict.fromkeys(windows.keys(), side)

    boundary = {k: v for k, v in boundary.items() if k in windows}
    side = {k: v for k, v in side.items() if k in windows}

    for d, window in windows.items():
        if window <= 0:
            raise ValueError(f"window must be > 0. Given {window} for dimension {d}")

    data = np.asarray(var.data)
    pad_widths = {}
    for d, window in windows.items():
        # trim or pad the object
        ax = list(var.dims).index(d)
        size = data.shape[ax]
        n = int(size / window)
        if boundary[d] == "exact":
            if n * window != size:
                raise ValueError(
                    f"Could not coarsen a dimension of size {size} with "
                    f"window {window} and boundary='exact'. Try a different "
                    f"'boundary' option."
                )
        elif boundary[d] == "trim":
            if side[d] == "left":
                data = np.take(data, np.arange(window * n), axis=ax)
            else:
                excess = size - window * n
                data = np.take(data, np.arange(excess, size), axis=ax)
        elif boundary[d] == "pad":  # pad
            pad = window * n - size
            if pad < 0:
                pad += window
            elif pad == 0:
                continue
            pad_widths[d] = (0, pad) if side[d] == "left" else (pad, 0)
        else:
            raise TypeError(
                f"{boundary[d]} is invalid for boundary. Valid option is "
                "'exact', 'trim' and 'pad'"
            )
    if pad_widths:
        pad_width = [pad_widths.get(d, (0, 0)) for d in var.dims]
        data = np.pad(data, pad_width, mode="constant", constant_values=np.nan)

    shape = []
    axes = []
    axis_count = 0
    for i, d in enumerate(var.dims):
        if d in windows:
            size = data.shape[i]
            shape.extend((int(size / windows[d]), windows[d]))
            axis_count += 1
            axes.append(i + axis_count)
        else:
            shape.append(data.shape[i])
    return data.reshape(shape), tuple(axes)


def _coarsen_reduce(var, windows, func, boundary, side, **kwargs):
    """`core/variable.py: Variable.coarsen` -- reshape into blocks, then reduce
    over the block axes."""
    if isinstance(func, str):
        if func not in ("sum", "prod", "max", "min", "mean", "std", "var",
                        "median", "count", "all", "any"):
            raise NameError(f"{func} is not a valid method.")
        # Upstream reduces the reshaped blocks through `duck_array_ops`, which
        # is NumPy.  Every name with a kernel here goes to the Mojo one
        # instead; `count`/`all`/`any` have none, and a dtype without a kernel
        # falls back inside the port's own wrapper.
        func = _COARSEN_MOJO.get(func) or getattr(duck_array_ops, func)
    reshaped, axes = _coarsen_reshape(var, windows, boundary, side)
    return xr.Variable(var.dims, func(reshaped, axis=axes, **kwargs))


# The `coarsen` reductions this port has a kernel for.  `count`, `all` and `any`
# are absent on purpose: they have no kernel, and `count` is a sum over a mask.
_COARSEN_MOJO = {
    "sum": nanops.nansum,
    "prod": nanops.nanprod,
    "mean": nanops.nanmean,
    "min": nanops.nanmin,
    "max": nanops.nanmax,
    "std": nanops.nanstd,
    "var": nanops.nanvar,
    "median": nanops.nanmedian,
}


class Coarsen(Generic[T_Xarray]):
    """A object that implements the coarsen."""

    __slots__ = ("boundary", "coord_func", "obj", "side", "trim_excess", "windows")
    _attributes = ("windows", "side", "trim_excess")
    obj: T_Xarray
    windows: Mapping[Hashable, int]
    side: Any
    boundary: Any
    coord_func: Mapping[Hashable, Any]

    def __init__(
        self,
        obj: T_Xarray,
        windows: Mapping[Any, int],
        boundary: Any,
        side: Any,
        coord_func: Any,
    ) -> None:
        self.obj = obj
        self.windows = windows
        self.side = side
        self.boundary = boundary

        missing_dims = tuple(dim for dim in windows.keys() if dim not in self.obj.dims)
        if missing_dims:
            raise ValueError(
                f"Window dimensions {missing_dims} not found in "
                f"{self.obj.__class__.__name__} dimensions {tuple(self.obj.dims)}"
            )

        if xr_utils.is_dict_like(coord_func):
            coord_func_map = coord_func
        else:
            coord_func_map = dict.fromkeys(self.obj.dims, coord_func)
        for c in self.obj.coords:
            if c not in coord_func_map:
                coord_func_map[c] = nanops.nanmean
        self.coord_func = coord_func_map

    def _get_keep_attrs(self, keep_attrs):
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        return keep_attrs

    def __repr__(self) -> str:
        """provide a nice str repr of our coarsen object"""
        attrs = ",".join(
            f"{k}->{getattr(self, k)}"
            for k in self._attributes
            if getattr(self, k, None) is not None
        )
        return f"{self.__class__.__name__} [{attrs}]"

    def construct(self, window_dim=None, keep_attrs=None, **window_dim_kwargs):
        """Convert this Coarsen object to a DataArray or Dataset,
        where the coarsening dimension is split or reshaped to two
        new dimensions.
        """
        window_dim = xr_utils.either_dict_or_kwargs(
            window_dim, window_dim_kwargs, "Coarsen.construct"
        )
        if not window_dim:
            raise ValueError(
                "Either window_dim or window_dim_kwargs need to be specified."
            )

        bad_new_dims = tuple(
            win
            for win, dims in window_dim.items()
            if len(dims) != 2 or isinstance(dims, str)
        )
        if bad_new_dims:
            raise ValueError(
                f"Please provide exactly two dimension names for the following "
                f"coarsening dimensions: {bad_new_dims}"
            )

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        missing_dims = set(window_dim) - set(self.windows)
        if missing_dims:
            raise ValueError(
                f"'window_dim' must contain entries for all dimensions to coarsen. "
                f"Missing {missing_dims}"
            )
        extra_windows = set(self.windows) - set(window_dim)
        if extra_windows:
            raise ValueError(
                f"'window_dim' includes dimensions that will not be coarsened: "
                f"{extra_windows}"
            )

        if isinstance(self.obj, xr.DataArray):
            reshaped = self.obj._to_temp_dataset()
        else:
            reshaped = self.obj

        reshaped.attrs = self.obj.attrs if keep_attrs else {}

        for key, var in reshaped.variables.items():
            reshaped_dims = tuple(
                itertools.chain(*[window_dim.get(dim, [dim]) for dim in list(var.dims)])
            )
            if reshaped_dims != var.dims:
                windows = {w: self.windows[w] for w in window_dim if w in var.dims}
                reshaped_var, _ = _coarsen_reshape(var, windows, self.boundary, self.side)
                attrs = var.attrs if keep_attrs else {}
                reshaped[key] = (reshaped_dims, reshaped_var, attrs)
            else:
                reshaped[key] = var

        # should handle window_dim being unindexed
        should_be_coords = (set(window_dim) & set(self.obj.coords)) | set(self.obj.coords)
        result = reshaped.set_coords(should_be_coords)
        if isinstance(self.obj, xr.DataArray):
            return self.obj._from_temp_dataset(result)
        return result


class DataArrayCoarsen(Coarsen["xr.DataArray"]):
    __slots__ = ()

    _reduce_extra_args_docstring = """"""

    @classmethod
    def _reduce_method(
        cls, func: Callable, include_skipna: bool = False, numeric_only: bool = False
    ) -> Callable[..., xr.DataArray]:
        """Return a wrapped function for injecting reduction methods."""
        kwargs: dict[str, Any] = {}
        if include_skipna:
            kwargs["skipna"] = None

        def wrapped_func(self, keep_attrs: bool | None = None, **kwargs2):
            keep_attrs = self._get_keep_attrs(keep_attrs)
            reduced = _coarsen_reduce(self.obj.variable, self.windows, func,
                                      self.boundary, self.side, **kwargs2)
            coords = {}
            for c, v in self.obj.coords.items():
                if c == self.obj.name:
                    coords[c] = reduced
                elif any(d in self.windows for d in v.dims):
                    coords[c] = _coarsen_reduce(
                        v.variable, self.windows, self.coord_func[c],
                        self.boundary, self.side, **kwargs2)
                else:
                    coords[c] = v
            return xr.DataArray(
                reduced, dims=self.obj.dims, coords=coords, name=self.obj.name
            )

        return wrapped_func

    def reduce(self, func: Callable, keep_attrs: bool | None = None, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).
        """
        wrapped_func = self._reduce_method(func)
        return wrapped_func(self, keep_attrs=keep_attrs, **kwargs)


class DatasetCoarsen(Coarsen["xr.Dataset"]):
    __slots__ = ()

    _reduce_extra_args_docstring = """"""

    @classmethod
    def _reduce_method(
        cls, func: Callable, include_skipna: bool = False, numeric_only: bool = False
    ) -> Callable[..., xr.Dataset]:
        """Return a wrapped function for injecting reduction methods."""
        kwargs: dict[str, Any] = {}
        if include_skipna:
            kwargs["skipna"] = None

        def wrapped_func(self, keep_attrs: bool | None = None, **kwargs2):
            keep_attrs = self._get_keep_attrs(keep_attrs)
            attrs = self.obj.attrs if keep_attrs else {}
            reduced = {
                key: _coarsen_reduce(da.variable, self.windows, func, self.boundary,
                                     self.side, **kwargs2)
                for key, da in self.obj.data_vars.items()
            }
            coords = {
                c: _coarsen_reduce(v.variable, self.windows, self.coord_func[c],
                                   self.boundary, self.side, **kwargs2)
                for c, v in self.obj.coords.items()
            }
            return xr.Dataset(reduced, coords=coords, attrs=attrs)

        return wrapped_func

    def reduce(self, func: Callable, keep_attrs=None, **kwargs) -> xr.Dataset:
        """Reduce the items in this group by applying `func` along some
        dimension(s).
        """
        wrapped_func = self._reduce_method(func)
        return wrapped_func(self, keep_attrs=keep_attrs, **kwargs)


# `CoarsenArithmetic(IncludeReduceMethods)` upstream hands the injected
# reductions to every subclass; calling the injector directly keeps the same set
# of names (all, any, max, min, mean, prod, sum, std, var, median, count).
from xarray.computation.ops import inject_reduce_methods  # noqa: E402

inject_reduce_methods(DataArrayCoarsen)
inject_reduce_methods(DatasetCoarsen)

# `inject_reduce_methods` binds `duck_array_ops.<name>`, which is NumPy, so
# re-inject every method this port has a kernel for.  `count`, `all` and `any`
# keep upstream's; `reduce(callable)` is untouched, since its callable is the
# caller's.
for _name, _kernel in _COARSEN_MOJO.items():
    for _cls in (DataArrayCoarsen, DatasetCoarsen):
        _injected = _cls._reduce_method(_kernel)
        _injected.__name__ = _name
        _injected.__doc__ = getattr(_cls, _name).__doc__
        setattr(_cls, _name, _injected)
