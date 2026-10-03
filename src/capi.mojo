"""The C ABI the Python bindings call through.

Buffers cross as `Int` addresses: `@export` refuses parametric functions, and a
pointer whose origin is inferred is parametric.  Every export is a one-line
forwarding stub -- the upstream logic lives in `ported.mojo`, and the shape
bookkeeping (choosing `outer`/`n`/`inner` from `shape` and `axis`) lives in
`python/mojo_xarray/_ops.py`.  Nothing here allocates.
"""

import ported
from ported import (
    _maybe_null_out,
    move_argmax,
    move_argmin,
    move_count,
    move_sum_of_squares,
    move_max,
    move_mean,
    move_median,
    move_min,
    move_std,
    move_sum,
    move_var,
    nan_count,
    nanargmax,
    nanargmin,
    nancumprod,
    nancumsum,
    nanfirst_index,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanprod,
    nanquantile,
    nansum,
    nanstd,
    nanvar,
    pad,
)

comptime Mx = Pointer[Float64, AnyOrigin[mut=True]]
comptime Rx = Pointer[Float64, AnyOrigin[mut=True]]
comptime Ix = Pointer[Int64, AnyOrigin[mut=True]]


@export("mx_nanfirst_index")
def mx_nanfirst_index(a: Int, o: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nanfirst_index(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
                   outer, n, inner)


@export("mx_nanmin")
def mx_nanmin(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nanmin(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
           Mx(unsafe_from_address=s), outer, n, inner)


@export("mx_nanmax")
def mx_nanmax(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nanmax(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
           Mx(unsafe_from_address=s), outer, n, inner)


@export("mx_nansum")
def mx_nansum(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nansum(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
           Mx(unsafe_from_address=s), outer, n, inner)


@export("mx_nanmean")
def mx_nanmean(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nanmean(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
            Mx(unsafe_from_address=s), outer, n, inner)


@export("mx_nanmedian")
def mx_nanmedian(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nanmedian(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
              Mx(unsafe_from_address=s), outer, n, inner)


@export("mx_nanvar")
def mx_nanvar(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int,
              ddof: Int) abi("C"):
    nanvar(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
           Mx(unsafe_from_address=s), outer, n, inner, ddof)


@export("mx_nanstd")
def mx_nanstd(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int,
              ddof: Int) abi("C"):
    nanstd(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
           Mx(unsafe_from_address=s), outer, n, inner, ddof)


@export("mx_nanprod")
def mx_nanprod(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nanprod(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
            Mx(unsafe_from_address=s), outer, n, inner)


@export("mx_nanargmin")
def mx_nanargmin(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nanargmin(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
              Mx(unsafe_from_address=s), outer, n, inner)


@export("mx_nanargmax")
def mx_nanargmax(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nanargmax(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
              Mx(unsafe_from_address=s), outer, n, inner)


@export("mx_nanquantile")
def mx_nanquantile(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int,
                   q: Float64) abi("C"):
    nanquantile(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
                Mx(unsafe_from_address=s), outer, n, inner, q)


@export("mx_nan_count")
def mx_nan_count(a: Int, o: Int, s: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nan_count(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
              Mx(unsafe_from_address=s), outer, n, inner)


@export("mx_move_sum_of_squares")
def mx_move_sum_of_squares(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                           min_count: Int, center: Int) abi("C"):
    move_sum_of_squares(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
                        Mx(unsafe_from_address=s), n_rows, n, win, min_count, center)


@export("mx_maybe_null_out")
def mx_maybe_null_out(r: Int, c: Int, n: Int, min_count: Int, n_out: Int) abi("C"):
    _maybe_null_out(Mx(unsafe_from_address=r), Rx(unsafe_from_address=c), n,
                    min_count, n_out)


@export("mx_nancumsum")
def mx_nancumsum(a: Int, o: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nancumsum(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o), outer, n, inner)


@export("mx_nancumprod")
def mx_nancumprod(a: Int, o: Int, outer: Int, n: Int, inner: Int) abi("C"):
    nancumprod(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o), outer, n, inner)


@export("mx_move_sum")
def mx_move_sum(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                min_count: Int, center: Int) abi("C"):
    move_sum(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
             Mx(unsafe_from_address=s), n_rows, n, win, min_count, center)


@export("mx_move_mean")
def mx_move_mean(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                 min_count: Int, center: Int) abi("C"):
    move_mean(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
              Mx(unsafe_from_address=s), n_rows, n, win, min_count, center)


@export("mx_move_min")
def mx_move_min(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                min_count: Int, center: Int) abi("C"):
    move_min(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
             Mx(unsafe_from_address=s), n_rows, n, win, min_count, center)


@export("mx_move_max")
def mx_move_max(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                min_count: Int, center: Int) abi("C"):
    move_max(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
             Mx(unsafe_from_address=s), n_rows, n, win, min_count, center)


@export("mx_move_median")
def mx_move_median(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                   min_count: Int, center: Int) abi("C"):
    move_median(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
                Mx(unsafe_from_address=s), n_rows, n, win, min_count, center)


@export("mx_move_var")
def mx_move_var(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                min_count: Int, ddof: Int, center: Int) abi("C"):
    move_var(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
             Mx(unsafe_from_address=s), n_rows, n, win, min_count, ddof, center)


@export("mx_move_std")
def mx_move_std(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                min_count: Int, ddof: Int, center: Int) abi("C"):
    move_std(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
             Mx(unsafe_from_address=s), n_rows, n, win, min_count, ddof, center)


@export("mx_move_argmin")
def mx_move_argmin(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                   min_count: Int, center: Int) abi("C"):
    move_argmin(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
                Mx(unsafe_from_address=s), n_rows, n, win, min_count, center)


@export("mx_move_argmax")
def mx_move_argmax(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                   min_count: Int, center: Int) abi("C"):
    move_argmax(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
                Mx(unsafe_from_address=s), n_rows, n, win, min_count, center)


@export("mx_move_count")
def mx_move_count(a: Int, o: Int, s: Int, n_rows: Int, n: Int, win: Int,
                  center: Int) abi("C"):
    move_count(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o),
               Mx(unsafe_from_address=s), n_rows, n, win, center)


@export("mx_par_workers")
def mx_par_workers() abi("C") -> Int:
    return ported.PAR_WORKERS



@export("mx_pad")
def mx_pad(a: Int, o: Int, ndim: Int, shape: Int, padlo: Int, padhi: Int,
           istride: Int, ostride: Int, total: Int, fill: Float64) abi("C"):
    pad(Rx(unsafe_from_address=a), Mx(unsafe_from_address=o), ndim,
        Ix(unsafe_from_address=shape), Ix(unsafe_from_address=padlo),
        Ix(unsafe_from_address=padhi), Ix(unsafe_from_address=istride),
        Ix(unsafe_from_address=ostride), total, fill)
