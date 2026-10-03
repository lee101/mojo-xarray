"""Kernels ported from xarray, in upstream source order.

The sections below mirror, top to bottom:

  1. ``xarray/core/nputils.py``         the float/int reduction primitives that
                                       ``nanops`` and ``rolling`` dispatch to
  2. ``xarray/computation/nanops.py``   the ``skipna``-aware wrappers
  3. ``xarray/computation/rolling.py``  the moving-window reductions, i.e. the
                                       ``bottleneck.move_*`` fast path
  4. ``xarray/core/variable.py``        ``rolling_window``'s padding and the
                                       constant-mode ``pad`` behind it

The reductions keep upstream's arithmetic in upstream's order: the same NaN
substitution, the same short-circuits, the same all-missing sentinels.  The one
structural change is that a reduction over ``axis`` is expressed in the
``(outer, n, inner)`` form every ndarray reduction uses, with the axis
decomposition left to the Python wrapper.

The reduction tag is a compile-time parameter of every kernel below, so each
kernel is generated once per operation and the per-element dispatch on it
folds away; a strided reduction keeps the scalar loop, and a reduction over a
contiguous run (`inner == 1`) also gets a SIMD body that reads the input once
per pass.
"""

from max.algorithm import parallelize
from std.math import sqrt
from std.sys import simd_width_of

comptime F64 = Float64
comptime I64 = Int64

comptime W = simd_width_of[DType.float64]()
comptime VEC = SIMD[DType.float64, W]

comptime NAN = F64(0.0) / F64(0.0)
comptime PINF = F64(1.0) / F64(0.0)
comptime NINF = F64(-1.0) / F64(0.0)

# `min(v, 0) + max(v, 0)` is `v` with every NaN turned into a zero: the free
# min/max return the non-NaN operand, and zero is the identity for both when it
# meets the other side.  It is the whole NaN mask, in three vector ops.
def _nz(v: VEC) -> VEC:
    return min(v, 0.0) + max(v, 0.0)

# A `parallelize` launch costs a fixed amount on this machine whatever it is
# handed, so the threshold has to be on the work, not on the shape: these are
# the element counts above which the launch stops being the dominant cost.
comptime PAR_MIN_ELEMS = 1 << 22
comptime PAR_MIN_ELEMS_VEC = 1 << 24
# A moving-window reduction visits `win` inputs per output element, so its work
# is counted in visits rather than elements.
comptime PAR_MIN_VISITS = 1 << 23
# One stripe per worker: the sorting kernels need a private scratch, and a work
# item can only claim one if the item index IS the worker index.  `PAR_WORKERS`
# must match the `par_workers()` the Python side reads.
comptime PAR_WORKERS = 16

# Kernel modes.  These are compile-time parameters, so a kernel body is
# generated once per mode and the dispatch inside it folds away.
comptime M_SUM: Int = 0
comptime M_MEAN: Int = 1
comptime M_MIN: Int = 2
comptime M_MAX: Int = 3
comptime M_VAR: Int = 4
comptime M_STD: Int = 5
comptime M_PROD: Int = 6
comptime M_MEDIAN: Int = 7
comptime M_ARGMIN: Int = 8
comptime M_ARGMAX: Int = 9
comptime M_SUMSQ: Int = 10
comptime M_COUNT: Int = 11

# Reduction op tags.  `_missing` is what a reduction yields when a slice holds no
# valid observation: NaN for the skipna reductions, and for `sum`/`prod`/`min`/
# `max` the value upstream gets from `skipna=False` with the `fillna` that
# `Rolling._array_reduce` installs.
comptime OP_SUM: Int = 0
comptime OP_MEAN: Int = 1
comptime OP_MIN: Int = 2
comptime OP_MAX: Int = 3
comptime OP_MEDIAN: Int = 4
comptime OP_VAR: Int = 5
comptime OP_STD: Int = 6
comptime OP_PROD: Int = 7
comptime OP_ARGMIN: Int = 8
comptime OP_ARGMAX: Int = 9
comptime OP_COUNT: Int = 10
comptime OP_SUMSQ: Int = 11

# `_replace_nan(a, fill)`: NaN is the only value these reductions reason about.
# An integer array has none, and `v != v` folds away there.
def _is_nan(v: F64) -> Bool:
    return v != v


def _missing[mode: Int]() -> F64:
    # duck_array_ops fills with `dtypes.INF` for min and `dtypes.NINF` for max so
    # padded positions can never win a comparison; with skipna=False an
    # all-missing window therefore comes back as +inf / -inf, and only the
    # trailing count gate turns that into NaN.
    if mode == M_MIN:
        return PINF
    if mode == M_MAX:
        return NINF
    if mode == M_SUM or mode == M_COUNT:
        return 0.0
    if mode == M_PROD:
        return 1.0
    return NAN


def _sqrt(x: F64) -> F64:
    if x <= 0.0:
        return 0.0
    return sqrt(x)


# ---------------------------------------------------------------------------
# order statistics, for nanmedian / nanquantile / the rolling median
# ---------------------------------------------------------------------------


def _insertion_sort(v: Pointer[F64, AnyOrigin[mut=True]], lo: Int, hi: Int):
    var i = lo + 1
    while i <= hi:
        var k = v.unsafe_load(i)
        var j = i - 1
        while j >= lo and v.unsafe_load(j) > k:
            v.unsafe_store(j + 1, v.unsafe_load(j))
            j -= 1
        v.unsafe_store(j + 1, k)
        i += 1


def _swap(v: Pointer[F64, AnyOrigin[mut=True]], i: Int, j: Int):
    var t = v.unsafe_load(i)
    v.unsafe_store(i, v.unsafe_load(j))
    v.unsafe_store(j, t)


def _quickselect(v: Pointer[F64, AnyOrigin[mut=True]], lo: Int, hi: Int, k: Int):
    # Leaves v[k] holding the k-th smallest of v[lo:hi+1], with every element
    # below k no larger than it.  A median and a quantile want two or three
    # order statistics, not an ordering, so this is O(n) where a full sort is
    # O(n log n) -- and it needs no scratch beyond the window itself.
    var plo = lo
    var phi = hi
    while plo < phi:
        if phi - plo < 12:
            _insertion_sort(v, plo, phi)
            return
        var mid = plo + (phi - plo) // 2
        # order v[plo] <= v[mid] <= v[phi] by swapping, then move the median of
        # the three to the end: a pivot on the true median costs a fraction of
        # the mispredicted partitions an endpoint pivot does, and unlike a pivot
        # held in a register it is still there to be swapped home below.
        if v.unsafe_load(plo) > v.unsafe_load(mid):
            _swap(v, plo, mid)
        if v.unsafe_load(mid) > v.unsafe_load(phi):
            _swap(v, mid, phi)
        if v.unsafe_load(plo) > v.unsafe_load(mid):
            _swap(v, plo, mid)
        _swap(v, mid, phi)
        var y = v.unsafe_load(phi)
        # Lomuto over [plo, phi): one comparison and at most one swap per
        # element, and Hoare's degenerate all-equal case cannot arise because
        # only the pivot's own side keeps the equals.
        var p = plo
        var i = plo
        while i < phi:
            if v.unsafe_load(i) < y:
                if p != i:
                    _swap(v, i, p)
                p += 1
            i += 1
        _swap(v, p, phi)
        # [plo, p) < y = v[p] <= [p + 1, phi], so the k-th element is on one
        # side only
        if k < p:
            phi = p - 1
        elif k > p:
            plo = p + 1
        else:
            return


def _median_of(v: Pointer[F64, AnyOrigin[mut=True]], n: Int) -> F64:
    # the ordinary median of v[0:n] -- the mean of the two central values when
    # n is even -- in O(n) rather than the O(n log n) of a full sort
    if n == 0:
        return NAN
    var k = n // 2
    _quickselect(v, 0, n - 1, k)
    if n % 2 == 1:
        return v.unsafe_load(k)
    # everything below k is no larger than v[k], and there are exactly k of
    # them, so the largest of them is the value below the median
    var lo = v.unsafe_load(0)
    var i = 1
    while i < k:
        lo = max(lo, v.unsafe_load(i))
        i += 1
    return (lo + v.unsafe_load(k)) / 2.0


def _lerp_pair(v: Pointer[F64, AnyOrigin[mut=True]], n: Int, k: Int) -> Tuple[F64, F64]:
    # the two samples np.nanquantile interpolates between: v[k] and v[k + 1].
    # Selecting k brackets them -- everything below k is no larger than v[k],
    # everything above it no smaller -- so the second one is the smallest of
    # the tail, and the tail is only (1 - q) of the window for the quantiles
    # that scan quickly.
    _quickselect(v, 0, n - 1, k)
    var below = v.unsafe_load(k)
    var above = v.unsafe_load(k + 1)
    var i = k + 2
    while i < n:
        above = min(above, v.unsafe_load(i))
        i += 1
    return (below, above)



# ---------------------------------------------------------------------------
# xarray/core/nputils.py

def nanfirst_index(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    outer: Int, n: Int, inner: Int,
):
    # nanfirst: `idx_first = np.argmax(~pd.isnull(values), axis=axis)`.  The
    # bool mask is never materialised: the first index whose value is not NaN
    # is the first True in it, and a slice with no observation is all False,
    # where argmax returns 0 -- and so does this.  nanlast is upstream's
    # `-1 - argmax` over the reversed slice, which `nputils.nanlast` keeps.
    var n_out = outer * inner

    def work(o: Int) {imm}:
        var oo = o // inner
        var ii = o - oo * inner
        var base = oo * n * inner + ii
        var idx = 0
        var k = 0
        while k < n:
            if not _is_nan(src.unsafe_load(base + k * inner)):
                idx = k
                break
            k += 1
        dst.unsafe_store(o, F64(idx))

    if n * n_out < PAR_MIN_ELEMS:
        var i = 0
        while i < n_out:
            work(i)
            i += 1
    else:
        parallelize(work, n_out, PAR_WORKERS)


# ---------------------------------------------------------------------------

def _minmax_range[is_max: Int](
    src: Pointer[F64, AnyOrigin[mut=True]], base: Int, n: Int
) -> F64:
    # min/max over a contiguous run of at least W elements.  The free min and
    # max return whichever operand is not NaN, so a NaN never wins a comparison
    # and never poisons the accumulator: the accumulator starts as the first
    # vector, a lane that never sees a value stays NaN, and min/max of two NaNs
    # is NaN -- which is exactly upstream's all-missing answer, with no count to
    # keep.  Four accumulators, because one is a dependency chain at the
    # latency of the min rather than at its throughput.
    var a0 = src.unsafe_load[width=W](base)
    var a1 = a0
    var a2 = a0
    var a3 = a0
    var k = W
    while k + 4 * W <= n:
        var v0 = src.unsafe_load[width=W](base + k)
        var v1 = src.unsafe_load[width=W](base + k + W)
        var v2 = src.unsafe_load[width=W](base + k + 2 * W)
        var v3 = src.unsafe_load[width=W](base + k + 3 * W)
        if is_max == 1:
            a0 = max(a0, v0)
            a1 = max(a1, v1)
            a2 = max(a2, v2)
            a3 = max(a3, v3)
        else:
            a0 = min(a0, v0)
            a1 = min(a1, v1)
            a2 = min(a2, v2)
            a3 = min(a3, v3)
        k += 4 * W
    var best: VEC
    if is_max == 1:
        best = max(max(a0, a1), max(a2, a3))
    else:
        best = min(min(a0, a1), min(a2, a3))
    while k + W <= n:
        var v = src.unsafe_load[width=W](base + k)
        if is_max == 1:
            best = max(best, v)
        else:
            best = min(best, v)
        k += W
    var r = best[0]
    for i in range(1, W):
        if is_max == 1:
            r = max(r, best[i])
        else:
            r = min(r, best[i])
    while k < n:
        var v = src.unsafe_load(base + k)
        if is_max == 1:
            r = max(r, v)
        else:
            r = min(r, v)
        k += 1
    return r


def _sum_range[is_sumsq: Int](
    src: Pointer[F64, AnyOrigin[mut=True]], base: Int, n: Int
) -> F64:
    # sum of the non-NaN values over a contiguous run.  `min(v, 0) + max(v, 0)`
    # is `v` with the NaNs zeroed, so the whole mask is three vector ops and the
    # accumulator stays a plain SIMD sum -- four of them, for the same reason as
    # the min/max above.
    var a0 = VEC(0.0)
    var a1 = a0
    var a2 = a0
    var a3 = a0
    var k = 0
    while k + 4 * W <= n:
        var v0 = _nz(src.unsafe_load[width=W](base + k))
        var v1 = _nz(src.unsafe_load[width=W](base + k + W))
        var v2 = _nz(src.unsafe_load[width=W](base + k + 2 * W))
        var v3 = _nz(src.unsafe_load[width=W](base + k + 3 * W))
        if is_sumsq == 1:
            a0 += v0 * v0
            a1 += v1 * v1
            a2 += v2 * v2
            a3 += v3 * v3
        else:
            a0 += v0
            a1 += v1
            a2 += v2
            a3 += v3
        k += 4 * W
    while k + W <= n:
        var v = _nz(src.unsafe_load[width=W](base + k))
        if is_sumsq == 1:
            a0 += v * v
        else:
            a0 += v
        k += W
    var r = (a0 + a1 + (a2 + a3)).reduce_add()
    while k < n:
        var v = src.unsafe_load(base + k)
        if not _is_nan(v):
            if is_sumsq == 1:
                r += v * v
            else:
                r += v
        k += 1
    return r


def _sqdev_range(
    src: Pointer[F64, AnyOrigin[mut=True]], base: Int, n: Int, avg: F64
) -> F64:
    # Sum of the squared deviations from `avg` over the non-NaN values of a
    # contiguous run -- the second pass of `nanvar`, and the one that reads the
    # data.  `_nz(v - avg)` is the deviation with the NaN lanes turned into
    # zeros, so the squares need no mask of their own.
    var a0 = VEC(0.0)
    var a1 = a0
    var a2 = a0
    var a3 = a0
    var k = 0
    while k + 4 * W <= n:
        var d0 = _nz(src.unsafe_load[width=W](base + k) - avg)
        var d1 = _nz(src.unsafe_load[width=W](base + k + W) - avg)
        var d2 = _nz(src.unsafe_load[width=W](base + k + 2 * W) - avg)
        var d3 = _nz(src.unsafe_load[width=W](base + k + 3 * W) - avg)
        a0 += d0 * d0
        a1 += d1 * d1
        a2 += d2 * d2
        a3 += d3 * d3
        k += 4 * W
    while k + W <= n:
        var d = _nz(src.unsafe_load[width=W](base + k) - avg)
        a0 += d * d
        k += W
    var r = (a0 + a1 + (a2 + a3)).reduce_add()
    while k < n:
        var v = src.unsafe_load(base + k)
        if not _is_nan(v):
            var d = v - avg
            r += d * d
        k += 1
    return r


# The two contiguity families that need no count: min/max and the plain sums.
# They take the `_range` helpers above, one output element at a time, and stripe
# the output elements across workers exactly as the general kernel below does.
def _run_seq[is_max: Int, is_sum: Int, is_sumsq: Int](
    src: Pointer[F64, AnyOrigin[mut=True]],
    dst: Pointer[F64, AnyOrigin[mut=True]],
    n_out: Int,
    n: Int,
    missing: F64,
):
    def by_out(o: Int) {imm}:
        if is_sum == 1:
            dst.unsafe_store(o, _sum_range[is_sumsq](src, o * n, n))
        else:
            var r = _minmax_range[is_max](src, o * n, n)
            if r != r:
                r = missing
            dst.unsafe_store(o, r)

    def stripe_out(w: Int) {imm}:
        var lo = n_out * w // PAR_WORKERS
        var hi = n_out * (w + 1) // PAR_WORKERS
        var i = lo
        while i < hi:
            by_out(i)
            i += 1

    if n * n_out >= PAR_MIN_ELEMS_VEC:
        parallelize(stripe_out, PAR_WORKERS, PAR_WORKERS)
    else:
        var i = 0
        while i < n_out:
            by_out(i)
            i += 1



def _axis_generic[mode: Int](
    src: Pointer[F64, AnyOrigin[mut=True]],
    dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]],
    n_out: Int,
    n: Int,
    inner: Int,
    ddof: Int,
    missing: F64,
):
    # `scratch` holds PAR_WORKERS private selection buffers of `n` entries each.
    # Output element `o` covers src[base + k * inner] for k in [0, n), with
    # base = (o // inner) * n * inner + (o % inner).  This is the (outer, axis,
    # inner) decomposition NumPy's own reductions use, and it is the only place
    # the axis bookkeeping happens.  `mode` is a parameter, not an argument, so
    # the tests against it below are resolved before the loop is compiled.
    def work(o: Int, slot: Int) {imm}:
        var buf = scratch.unsafe_offset(slot * n)
        var oo = o // inner
        var ii = o - oo * inner
        var base = oo * n * inner + ii

        var acc = 0.0
        var count = 0
        var best = 0.0
        var best_i = 0
        var have = False
        var k = 0
        while k < n:
            var v = src.unsafe_load(base + k * inner)
            if not _is_nan(v):
                count += 1
                if (
                    mode == M_SUM or mode == M_SUMSQ
                    or mode == M_MEAN or mode == M_MEDIAN
                    or mode == M_VAR or mode == M_STD
                ):
                    acc += v
                elif mode == M_PROD:
                    if have:
                        acc = acc * v
                    else:
                        acc = v
                elif mode == M_MIN or mode == M_ARGMIN:
                    if not have or v < best:
                        best = v
                        best_i = k
                elif mode == M_MAX or mode == M_ARGMAX:
                    if not have or v > best:
                        best = v
                        best_i = k
                have = True
            k += 1

        var res = 0.0
        if mode == M_SUM or mode == M_SUMSQ or mode == M_PROD:
            res = acc
        elif mode == M_MIN or mode == M_MAX:
            res = best
        elif mode == M_ARGMIN or mode == M_ARGMAX:
            res = F64(best_i)
        elif mode == M_MEAN:
            res = acc / F64(count)
        elif mode == M_MEDIAN:
            # _nanmedian1d: drop the NaNs, NaN if none remain, then the ordinary
            # median -- the mean of the two central values when the count is even.
            if count == 0:
                res = NAN
            else:
                var m = 0
                k = 0
                while k < n:
                    var v = src.unsafe_load(base + k * inner)
                    if not _is_nan(v):
                        buf.unsafe_store(m, v)
                        m += 1
                    k += 1
                res = _median_of(buf, count)
        elif mode == M_VAR or mode == M_STD:
            # np.nanvar: the mean, then the sum of squared deviations over
            # count - ddof, NaN wherever count <= ddof.  np.nanstd is its sqrt.
            if count <= ddof:
                res = NAN
            else:
                var avg = acc / F64(count)
                var sqr = 0.0
                if inner == 1 and n >= W:
                    sqr = _sqdev_range(src, base, n, avg)
                else:
                    k = 0
                    while k < n:
                        var v = src.unsafe_load(base + k * inner)
                        if not _is_nan(v):
                            var d = v - avg
                            sqr += d * d
                        k += 1
                res = sqr / F64(count - ddof)
                if mode == M_STD:
                    res = _sqrt(res)

        if not have:
            res = missing
        dst.unsafe_store(o, res)

    def stripe(w: Int) {imm}:
        var lo = n_out * w // PAR_WORKERS
        var hi = n_out * (w + 1) // PAR_WORKERS
        var i = lo
        while i < hi:
            work(i, w)
            i += 1

    if n * n_out < PAR_MIN_ELEMS:
        var i = 0
        while i < n_out:
            work(i, 0)
            i += 1
    else:
        parallelize(stripe, PAR_WORKERS, PAR_WORKERS)


def _run_axis[mode: Int](
    src: Pointer[F64, AnyOrigin[mut=True]],
    dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]],
    n_out: Int,
    n: Int,
    inner: Int,
    ddof: Int,
    missing: F64,
):
    # `inner == 1` is a contiguous run per output element, which is the case the
    # vector bodies above can walk; anything strided, and every reduction that
    # needs a count the vector form cannot carry, takes the general kernel.
    if inner == 1 and n >= W:
        if mode == M_MIN:
            _run_seq[0, 0, 0](src, dst, n_out, n, missing)
            return
        elif mode == M_MAX:
            _run_seq[1, 0, 0](src, dst, n_out, n, missing)
            return
        elif mode == M_SUM:
            _run_seq[0, 1, 0](src, dst, n_out, n, missing)
            return
        elif mode == M_SUMSQ:
            _run_seq[0, 1, 1](src, dst, n_out, n, missing)
            return
    _axis_generic[mode](src, dst, scratch, n_out, n, inner, ddof, missing)


def _reduce_axis(
    src: Pointer[F64, AnyOrigin[mut=True]],
    dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]],
    outer: Int,
    n: Int,
    inner: Int,
    ddof: Int,
    op: Int,
    missing: F64,
):
    var n_out = outer * inner
    if op == OP_SUM:
        _run_axis[M_SUM](src, dst, scratch, n_out, n, inner, ddof, missing)
    elif op == OP_MEAN:
        _run_axis[M_MEAN](src, dst, scratch, n_out, n, inner, ddof, missing)
    elif op == OP_MIN:
        _run_axis[M_MIN](src, dst, scratch, n_out, n, inner, ddof, missing)
    elif op == OP_MAX:
        _run_axis[M_MAX](src, dst, scratch, n_out, n, inner, ddof, missing)
    elif op == OP_MEDIAN:
        _run_axis[M_MEDIAN](src, dst, scratch, n_out, n, inner, ddof, missing)
    elif op == OP_VAR:
        _run_axis[M_VAR](src, dst, scratch, n_out, n, inner, ddof, missing)
    elif op == OP_STD:
        _run_axis[M_STD](src, dst, scratch, n_out, n, inner, ddof, missing)
    elif op == OP_PROD:
        _run_axis[M_PROD](src, dst, scratch, n_out, n, inner, ddof, missing)
    elif op == OP_ARGMIN:
        _run_axis[M_ARGMIN](src, dst, scratch, n_out, n, inner, ddof, missing)
    elif op == OP_ARGMAX:
        _run_axis[M_ARGMAX](src, dst, scratch, n_out, n, inner, ddof, missing)
    elif op == OP_COUNT:
        # duck_array_ops.sum(isnull(a), axis): summing the 0/1 mask
        _run_axis[M_SUM](src, dst, scratch, n_out, n, inner, ddof, missing)
    else:
        _run_axis[M_SUMSQ](src, dst, scratch, n_out, n, inner, ddof, missing)


def nanmin(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
):
    _reduce_axis(src, dst, scratch, outer, n, inner, 0, OP_MIN, NAN)


def nanmax(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
):
    _reduce_axis(src, dst, scratch, outer, n, inner, 0, OP_MAX, NAN)


def nanmean(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
):
    _reduce_axis(src, dst, scratch, outer, n, inner, 0, OP_MEAN, NAN)


def nanmedian(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
):
    _reduce_axis(src, dst, scratch, outer, n, inner, 0, OP_MEDIAN, NAN)


def nanvar(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
    ddof: Int,
):
    _reduce_axis(src, dst, scratch, outer, n, inner, ddof, OP_VAR, NAN)


def nanstd(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
    ddof: Int,
):
    _reduce_axis(src, dst, scratch, outer, n, inner, ddof, OP_STD, NAN)


def nanprod(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
):
    # np._nanprod is _replace_nan(a, 1).prod: an all-missing slice is 1, not NaN.
    _reduce_axis(src, dst, scratch, outer, n, inner, 0, OP_PROD, 1.0)


def nanargmin(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
):
    _reduce_axis(src, dst, scratch, outer, n, inner, 0, OP_ARGMIN, NAN)


def nanargmax(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
):
    _reduce_axis(src, dst, scratch, outer, n, inner, 0, OP_ARGMAX, NAN)


def nan_count(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
):
    # duck_array_ops.sum(isnull(a), axis) over a 0/1 mask: how many NaNs each
    # reduced slice holds.
    _reduce_axis(src, dst, scratch, outer, n, inner, 0, OP_COUNT, 0.0)


def _nanquantile_lerp(a: F64, b: F64, t: F64) -> F64:
    # numpy's `_lerp`: the two forms agree in exact arithmetic but not in
    # floating point, and the t >= 0.5 form is the one numpy uses there.  The
    # Mojo backend contracts `a + (b - a) * t` into an fma, so results can
    # differ from numpy's separate multiply-then-add by one ulp; see the README.
    var diff_b_a = b - a
    if t >= 0.5:
        return b - diff_b_a * (1.0 - t)
    return a + diff_b_a * t


def nanquantile(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
    q: F64,
):
    # np.nanquantile with method="linear": drop the NaNs, then interpolate on the
    # virtual index q * (count - 1).
    var n_out = outer * inner

    def work(o: Int, slot: Int) {imm}:
        var buf = scratch.unsafe_offset(slot * n)
        var oo = o // inner
        var ii = o - oo * inner
        var base = oo * n * inner + ii
        var m = 0
        var k = 0
        while k < n:
            var v = src.unsafe_load(base + k * inner)
            if not _is_nan(v):
                buf.unsafe_store(m, v)
                m += 1
            k += 1
        if m == 0:
            dst.unsafe_store(o, NAN)
            return
        var virtual_index = q * F64(m - 1)
        var lo = Int(virtual_index)  # q is clamped to [0, 1], so trunc == floor
        if lo < 0:
            lo = 0
        elif lo > m - 1:
            lo = m - 1
        if lo >= m - 1:
            # q == 1 (or a single sample): the largest, and no sample above it to
            # interpolate towards
            _quickselect(buf, 0, m - 1, m - 1)
            dst.unsafe_store(o, buf.unsafe_load(m - 1))
            return
        # np.nanquantile's own short-circuit: an exact q lands on a sample
        var pair = _lerp_pair(buf, m, lo)
        dst.unsafe_store(o, _nanquantile_lerp(pair[0], pair[1], virtual_index - F64(lo)))

    def stripe(w: Int) {imm}:
        var lo = n_out * w // PAR_WORKERS
        var hi = n_out * (w + 1) // PAR_WORKERS
        var i = lo
        while i < hi:
            work(i, w)
            i += 1

    if n * n_out < PAR_MIN_ELEMS:
        var i = 0
        while i < n_out:
            work(i, 0)
            i += 1
    else:
        parallelize(stripe, PAR_WORKERS, PAR_WORKERS)


# ---------------------------------------------------------------------------
# xarray/computation/nanops.py
# ---------------------------------------------------------------------------

def nansum(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], outer: Int, n: Int, inner: Int,
):
    # sum_where(a, axis=axis, dtype=dtype, where=mask) with mask = isnull(a):
    # the NaNs drop out of the sum, and an all-missing slice sums to 0.
    _reduce_axis(src, dst, scratch, outer, n, inner, 0, OP_SUM, 0.0)


def _maybe_null_out(
    result: Pointer[F64, AnyOrigin[mut=True]],
    count: Pointer[F64, AnyOrigin[mut=True]],
    n: Int,
    min_count: Int,
    n_out: Int,
):
    # _maybe_null_out: mask = isnull(a) and count = sum(mask, axis) = the
    # number of NaNs in the slice, so upstream's
    #   (take(mask.shape, axis).prod() - sum(mask, axis) - min_count) < 0
    # is taken verbatim.  maybe_promote gives float data NaN as the fill value.
    var i = 0
    while i < n_out:
        var null_mask = F64(n) - count.unsafe_load(i) - F64(min_count)
        if null_mask < 0.0:
            result.unsafe_store(i, NAN)
        i += 1


# ---------------------------------------------------------------------------
# xarray/computation/rolling.py
# ---------------------------------------------------------------------------

def _pads(win: Int, center: Int) -> Int:
    """The leading pad `core/variable.py: rolling_window` applies.

    Centred: `(win // 2, win - 1 - win // 2)` -- 10 -> 5, 9 -> 4.  Otherwise
    `(win - 1, 0)`, so the window ends at the right edge of the label.
    """
    if center:
        return win // 2
    return win - 1


def _move_reduce[mode: Int](
    src: Pointer[F64, AnyOrigin[mut=True]],
    dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]],
    n_rows: Int,
    n: Int,
    win: Int,
    min_count: Int,
    ddof: Int,
    center: Int,
):
    # Each window is padded exactly as `Variable.rolling_window` pads it, and
    # output element `o` reduces a[o - lo_pad : o + (win - 1 - lo_pad)] with
    # out-of-range positions counted as missing.  The result is then gated on
    # the non-missing count, which is upstream's trailing
    var lo_pad = _pads(win, center)
    var n_out = n_rows * n

    def work(flat: Int, slot: Int) {imm}:
        var buf = scratch.unsafe_offset(slot * win)
        var row = flat // n
        var o = flat - row * n
        var first = o - lo_pad
        var row_off = row * n

        # The in-range part of the window, once: a window that hangs off either
        # end of the row has a shorter live range, and the per-element bounds
        # test it replaces was two compares and two branches on every step.
        var k_lo = 0
        if first < 0:
            k_lo = -first
        var k_hi = win
        if first + win > n:
            k_hi = n - first

        var acc = 0.0
        var count = 0
        var best = 0.0
        var best_k = 0
        var have = False
        var k = k_lo
        while k < k_hi:
            var v = src.unsafe_load(row_off + first + k)
            if not _is_nan(v):
                    count += 1
                    if mode == M_SUMSQ:
                        acc += v * v
                    elif (
                        mode == M_SUM or mode == M_COUNT
                        or mode == M_MEAN or mode == M_MEDIAN
                        or mode == M_VAR or mode == M_STD
                    ):
                        acc += v
                    elif mode == M_PROD:
                        if have:
                            acc = acc * v
                        else:
                            acc = v
                    elif mode == M_MIN or mode == M_ARGMIN:
                        if not have or v < best:
                            best = v
                            best_k = k
                    elif mode == M_MAX or mode == M_ARGMAX:
                        if not have or v > best:
                            best = v
                            best_k = k
                    have = True
            k += 1

        var res = 0.0
        if mode == M_SUM or mode == M_SUMSQ or mode == M_PROD or mode == M_COUNT:
            res = acc
        elif mode == M_MIN or mode == M_MAX:
            res = best
        elif mode == M_ARGMIN or mode == M_ARGMAX:
            res = F64(best_k)
        elif mode == M_MEAN:
            res = acc / F64(count)
        elif mode == M_MEDIAN:
            if count == 0:
                res = NAN
            else:
                var m = 0
                k = k_lo
                while k < k_hi:
                    var v = src.unsafe_load(row_off + first + k)
                    if not _is_nan(v):
                        buf.unsafe_store(m, v)
                        m += 1
                    k += 1
                res = _median_of(buf, count)
        elif mode == M_VAR or mode == M_STD:
            if count <= ddof:
                res = NAN
            else:
                var avg = acc / F64(count)
                var sqr = 0.0
                k = k_lo
                while k < k_hi:
                    var v = src.unsafe_load(row_off + first + k)
                    if not _is_nan(v):
                        var d = v - avg
                        sqr += d * d
                    k += 1
                res = sqr / F64(count - ddof)
                if mode == M_STD:
                    res = _sqrt(res)

        if not have:
            res = _missing[mode]()
        if min_count > 0 and count < min_count:
            res = NAN
        dst.unsafe_store(flat, res)

    def stripe(w: Int) {imm}:
        var lo = n_out * w // PAR_WORKERS
        var hi = n_out * (w + 1) // PAR_WORKERS
        var i = lo
        while i < hi:
            work(i, w)
            i += 1

    if n_out * win < PAR_MIN_VISITS:
        var i = 0
        while i < n_out:
            work(i, 0)
            i += 1
    else:
        parallelize(stripe, PAR_WORKERS, PAR_WORKERS)


def move_sum(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    min_count: Int, center: Int,
):
    _move_reduce[M_SUM](src, dst, scratch, n_rows, n, win, min_count, 0, center)


def move_mean(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    min_count: Int, center: Int,
):
    # Rolling._mean: the rolling sum divided by the rolling count.
    _move_reduce[M_MEAN](src, dst, scratch, n_rows, n, win, min_count, 0, center)


def move_min(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    min_count: Int, center: Int,
):
    _move_reduce[M_MIN](src, dst, scratch, n_rows, n, win, min_count, 0, center)


def move_max(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    min_count: Int, center: Int,
):
    _move_reduce[M_MAX](src, dst, scratch, n_rows, n, win, min_count, 0, center)


def move_median(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    min_count: Int, center: Int,
):
    _move_reduce[M_MEDIAN](src, dst, scratch, n_rows, n, win, min_count, 0, center)


def move_var(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    min_count: Int, ddof: Int, center: Int,
):
    _move_reduce[M_VAR](src, dst, scratch, n_rows, n, win, min_count, ddof, center)


def move_std(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    min_count: Int, ddof: Int, center: Int,
):
    _move_reduce[M_STD](src, dst, scratch, n_rows, n, win, min_count, ddof, center)


def move_argmin(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    min_count: Int, center: Int,
):
    # Without bottleneck upstream reduces a materialised window, so the index is
    # an offset from the window's left edge -- which is what `k` already is.
    _move_reduce[M_ARGMIN](src, dst, scratch, n_rows, n, win, min_count, 0, center)


def move_argmax(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    min_count: Int, center: Int,
):
    _move_reduce[M_ARGMAX](src, dst, scratch, n_rows, n, win, min_count, 0, center)


def move_sum_of_squares(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    min_count: Int, center: Int,
):
    # bottleneck.move_sum_of_squares: the rolling sum of v**2 over each window.
    # upstream reaches it through the same `_bottleneck_reduce` dispatch the
    # other move_* kernels use, so it is ported and exported here too.
    _move_reduce[M_SUMSQ](src, dst, scratch, n_rows, n, win, min_count, 0, center)


def move_count(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    scratch: Pointer[F64, AnyOrigin[mut=True]], n_rows: Int, n: Int, win: Int,
    center: Int,
):
    # DataArrayRolling._counts: the sum of the notnull mask over each window, with
    # the padding filled with False so padded positions never count.  There is no
    # min_count here because upstream gates on min_periods afterwards, in
    # `Rolling.count`.
    _move_reduce[M_COUNT](src, dst, scratch, n_rows, n, win, 0, 0, center)


# ---------------------------------------------------------------------------
# xarray/core/variable.py -- pad (mode="constant") and the cumulative ops
# ---------------------------------------------------------------------------

def pad(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    ndim: Int,
    shape: Pointer[I64, AnyOrigin[mut=True]],
    padlo: Pointer[I64, AnyOrigin[mut=True]],
    padhi: Pointer[I64, AnyOrigin[mut=True]],
    istride: Pointer[I64, AnyOrigin[mut=True]],
    ostride: Pointer[I64, AnyOrigin[mut=True]],
    total: Int,
    fill: F64,
):
    # duck_array_ops.pad(..., mode="constant", constant_values=fill_value).
    #
    # Only the last dimension actually moves, so the output is emitted one run
    # of it at a time: a fill band, a contiguous copy of the input row, another
    # fill band.  The dimensions above it only decide which input row the run
    # reads from -- or that the run is entirely fill -- so they are peeled once
    # per run rather than once per element, which is what the gather-and-fill
    # per element used to cost.
    if ndim <= 0 or total <= 0:
        return
    var last = ndim - 1
    var keep = Int(shape.unsafe_load(last))
    var lo = Int(padlo.unsafe_load(last))
    var hi = Int(padhi.unsafe_load(last))
    var row = keep + lo + hi
    var packed = Int(istride.unsafe_load(last)) == 1

    def work(b: Int) {imm}:
        # `b` counts runs of the last dimension; the index above it is peeled
        # with the same row-major strides the per-element form used.
        var rem = b
        var src_off = 0
        var ok = True
        var d = 0
        while d < last:
            # ostride[d] is the product of the extents after d, so dividing it
            # by the run length leaves the radix of the run index at d
            var ext = Int(ostride.unsafe_load(d)) // row
            var c = rem // ext
            rem = rem - c * ext
            var plo = Int(padlo.unsafe_load(d))
            if c < plo or c >= plo + Int(shape.unsafe_load(d)):
                ok = False
            else:
                src_off += (c - plo) * Int(istride.unsafe_load(d))
            d += 1
        var o = dst.unsafe_offset(b * row)
        var i = 0
        while i < lo:
            o.unsafe_store(i, fill)
            i += 1
        if not ok:
            while i < lo + keep:
                o.unsafe_store(i, fill)
                i += 1
        elif packed:
            var s = src.unsafe_offset(src_off)
            var j = 0
            while j + W <= keep:
                o.unsafe_store(lo + j, s.unsafe_load[width=W](j))
                j += W
            while j < keep:
                o.unsafe_store(lo + j, s.unsafe_load(j))
                j += 1
        else:
            # a strided input row, which `_ops.pad` never produces but the ABI
            # does not forbid
            var s = src.unsafe_offset(src_off)
            var step = Int(istride.unsafe_load(last))
            var j = 0
            while j < keep:
                o.unsafe_store(lo + j, s.unsafe_load(j * step))
                j += 1
        i = lo + keep
        while i < row:
            o.unsafe_store(i, fill)
            i += 1

    var n_runs = total // row
    if total < PAR_MIN_ELEMS:
        var b = 0
        while b < n_runs:
            work(b)
            b += 1
    else:
        parallelize(work, n_runs, PAR_WORKERS)


def nancumsum(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    outer: Int, n: Int, inner: Int,
):
    # _replace_nan(a, 0).cumsum(axis)
    var n_out = outer * inner

    def work(o: Int) {imm}:
        var oo = o // inner
        var ii = o - oo * inner
        var base = oo * n * inner + ii
        var acc = 0.0
        var k = 0
        while k < n:
            var v = src.unsafe_load(base + k * inner)
            if _is_nan(v):
                v = 0.0
            acc += v
            dst.unsafe_store(base + k * inner, acc)
            k += 1

    if n * n_out < PAR_MIN_ELEMS:
        var i = 0
        while i < n_out:
            work(i)
            i += 1
    else:
        parallelize(work, n_out, PAR_WORKERS)


def nancumprod(
    src: Pointer[F64, AnyOrigin[mut=True]], dst: Pointer[F64, AnyOrigin[mut=True]],
    outer: Int, n: Int, inner: Int,
):
    # _replace_nan(a, 1).cumprod(axis)
    var n_out = outer * inner

    def work(o: Int) {imm}:
        var oo = o // inner
        var ii = o - oo * inner
        var base = oo * n * inner + ii
        var acc = 1.0
        var k = 0
        while k < n:
            var v = src.unsafe_load(base + k * inner)
            if _is_nan(v):
                v = 1.0
            acc = acc * v
            dst.unsafe_store(base + k * inner, acc)
            k += 1

    if n * n_out < PAR_MIN_ELEMS:
        var i = 0
        while i < n_out:
            work(i)
            i += 1
    else:
        parallelize(work, n_out, PAR_WORKERS)
