"""The `(outer, n, inner)` axis decomposition, and the padding in
`xarray/core/variable.py: rolling_window`.

Mojo forces this split: `@export` refuses parametric functions, so a kernel can
never see a shape tuple and takes three strides instead.  Everything upstream
knows about `axis` therefore lives here, in the order upstream writes it.
"""

from __future__ import annotations

import numpy as np

from ._lib import lib

__all__ = [
    "decompose", "normalize_axis_index", "reduce_axis", "move", "pad",
    "cumulative", "maybe_null_out", "first_valid_index",
]


def normalize_axis_index(axis, ndim) -> int:
    if axis < -ndim or axis >= ndim:
        raise np.exceptions.AxisError(axis, ndim=ndim)
    return axis % ndim


def decompose(shape, axis):
    """Split `shape` around `axis` into `(outer, n, inner)`.

    `axis=None` collapses to a single output element over the whole array, the
    same way NumPy's own reductions do.
    """
    if axis is None:
        return 1, int(np.prod(shape, dtype=np.int64)), 1
    if isinstance(axis, (list, tuple)):
        raise TypeError("decompose takes a single axis; loop over the tuple")
    axis = normalize_axis_index(axis, len(shape))
    outer = int(np.prod(shape[:axis], dtype=np.int64)) if axis else 1
    n = int(shape[axis])
    inner = int(np.prod(shape[axis + 1:], dtype=np.int64)) if axis + 1 < len(shape) else 1
    return outer, n, inner


_workers = None


def workers() -> int:
    """The stripe count the kernels were compiled with.  Each parallel worker
    owns one private sorting buffer, so the scratch must be this many times the
    reduction length."""
    global _workers
    if _workers is None:
        _workers = int(lib().mx_par_workers())
    return _workers


def _scratch(n: int) -> np.ndarray:
    # one private buffer per worker, plus a spare slot so `_quicksort` may read
    # a[hi - 1] even for tiny inputs
    return np.empty(workers() * n + 2, dtype=np.float64)


def reduce_axis(a, name, axis=None, ddof=0, q=None):
    """Run the `name` reduction kernel over `a` along `axis`."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    outer, n, inner = decompose(a.shape, axis)
    out = np.empty(outer * inner, dtype=np.float64)
    if out.size:
        # the scratch array must outlive the call: taking only its address would
        # let the temporary be freed while the kernel still writes into it
        scratch = _scratch(n)
        fn = getattr(lib(), name)
        args = [a.ctypes.data, out.ctypes.data, scratch.ctypes.data, outer, n, inner]
        if name == "mx_nanquantile":
            fn(*args, float(q))
        elif name in ("mx_nanvar", "mx_nanstd"):
            fn(*args, int(ddof))
        else:
            fn(*args)
    if axis is None:
        return out.reshape(())
    ax = normalize_axis_index(axis, a.ndim)
    return out.reshape(a.shape[:ax] + a.shape[ax + 1:])



def move(a, name, window, min_count, center, ddof=0, n_rows=1):
    """Run the `name` moving-window kernel over the last axis of `a`."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    n = a.shape[-1]
    out = np.empty(a.size, dtype=np.float64)
    scratch = _scratch(window)
    fn = getattr(lib(), name)
    head = [a.ctypes.data, out.ctypes.data, scratch.ctypes.data, n_rows, n, window]
    if name in ("mx_move_var", "mx_move_std"):
        fn(*head, min_count, int(ddof), int(center))
    elif name == "mx_move_count":
        fn(*head, int(center))
    else:
        fn(*head, min_count, int(center))
    return out.reshape(a.shape)


def pad(a, pad_width, fill_value=np.nan):
    """`duck_array_ops.pad(..., mode="constant", constant_values=fill_value)`."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    ndim = a.ndim
    padlo = np.array([p[0] for p in pad_width], dtype=np.int64)
    padhi = np.array([p[1] for p in pad_width], dtype=np.int64)
    shape = np.array(a.shape, dtype=np.int64)
    out_shape = tuple(int(s) for s in shape + padlo + padhi)
    istride = np.array([int(np.prod(shape[i + 1:])) for i in range(ndim)], dtype=np.int64)
    ostride = np.array([int(np.prod(out_shape[i + 1:])) for i in range(ndim)], dtype=np.int64)
    total = int(np.prod(out_shape))
    out = np.empty(total, dtype=np.float64)
    lib().mx_pad(a.ctypes.data, out.ctypes.data, ndim,
                 shape.ctypes.data, padlo.ctypes.data, padhi.ctypes.data,
                 istride.ctypes.data, ostride.ctypes.data, total, float(fill_value))
    return out.reshape(out_shape)


def cumulative(a, name, axis=None):
    """`_replace_nan(a, fill).cum{sum,prod}(axis)`."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    outer, n, inner = decompose(a.shape, axis)
    out = np.empty(a.size, dtype=np.float64)
    getattr(lib(), name)(a.ctypes.data, out.ctypes.data, outer, n, inner)
    # np.cumsum(axis=None) flattens, and so does this
    return out.reshape(a.shape if axis is not None else (a.size,))


def first_valid_index(a, axis):
    """`xarray/core/nputils.py: nanfirst`'s `np.argmax(~isnull(values), axis)`.

    The kernel carries the index in a float64 buffer like every other one here;
    upstream's `argmax` returns `intp`, so it is narrowed back on the way out.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    outer, n, inner = decompose(a.shape, axis)
    out = np.empty(outer * inner, dtype=np.float64)
    if out.size:
        lib().mx_nanfirst_index(a.ctypes.data, out.ctypes.data, outer, n, inner)
    idx = out.astype(np.intp)
    if axis is None:
        return idx.reshape(())
    ax = normalize_axis_index(axis, a.ndim)
    return idx.reshape(a.shape[:ax] + a.shape[ax + 1:])


def maybe_null_out(result, mask, axis, min_count=1):
    """`xarray/computation/nanops.py: _maybe_null_out`.

    `mask` is `isnull(a)`, so `count` is `sum(mask, axis)` -- the number of
    NaNs in each reduced slice.
    """
    shape = result.shape
    result = np.ascontiguousarray(result, dtype=np.float64).reshape(shape)
    outer, n, inner = decompose(mask.shape, axis)
    n_out = outer * inner
    if n_out:
        m = np.ascontiguousarray(mask, dtype=np.float64)
        count = np.empty(n_out, dtype=np.float64)
        scratch = _scratch(n)
        lib().mx_nan_count(m.ctypes.data, count.ctypes.data,
                           scratch.ctypes.data, outer, n, inner)
        lib().mx_maybe_null_out(result.ctypes.data, count.ctypes.data, n, min_count, n_out)
    return result.reshape(shape)
