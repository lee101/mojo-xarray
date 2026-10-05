# mojo-xarray

xarray's labelled-array reductions and rolling windows, written in Mojo.

`mojo_xarray` is a drop-in for the slice of [xarray](https://xarray.dev) that
spends its time in a reduction loop: `nanops`/`nputils` (`nansum`, `nanmean`,
`nanmin`, `nanmax`, `nanmedian`, `nanvar`, `nanstd`, `nanprod`, `nanargmin`,
`nanargmax`, `nancumsum`, `nancumprod`, `nanquantile`) and `rolling`
(`Rolling`, `DataArrayRolling`, `DatasetRolling`, `Coarsen`, `DataArrayCoarsen`,
`DatasetCoarsen`). The kernel names, argument order and defaults are upstream's,
so the two files can be read side by side: `src/ported.mojo` emits the upstream
functions in upstream's source order, with the same branch structure and the
same arithmetic.

`mojo_xarray.DataArray` and `mojo_xarray.Dataset` subclass their xarray
counterparts, so a `mojo_xarray.DataArray` *is* an `xarray.DataArray` -- it keeps
its dims, coords, indexes, attrs and name, and anything this port does not
override behaves exactly as upstream.

This is a CPU port. A moving-window reduction has enough arithmetic intensity to
be worth a GPU and a whole-array reduction has none, so the GPU is discussed in
**Why there is no GPU path** rather than benchmarked.

## Install

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles `src/capi.mojo` into `dist/libmojo-xarray.so` and
copies it to `build/capi.so`. The Python package loads that library through
ctypes and rebuilds it automatically when a `.mojo` file is newer, so a bare
`import mojo_xarray` works too. Set `MOJO_XARRAY_LIB` to point at a prebuilt
`.so` and nothing is ever compiled at import or call time.

## Usage

```python
import numpy as np
import mojo_xarray as mx

da = mx.DataArray(
    np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0]),
    dims="time",
    coords={"time": np.arange(7)},
    name="temperature",
    attrs={"units": "K"},
)

da.rolling(time=3, min_periods=1, center=True).mean()
# <xarray.DataArray 'temperature' (time: 7)> Size: 56B
# array([1.5, 1.5, 3. , 4.5, 5. , 6. , 6.5])
# Coordinates:
#   * time     (time) int64 56B 0 1 2 3 4 5 6
# Attributes:
#     units:    K

da.mean(dim="time")                            # Mojo kernel
da.rolling(time=3, min_periods=1).std()         # Mojo kernel
da.coarsen(time=1).median()                    # Mojo kernel
mx.nanquantile(da.data, [0.25, 0.75], axis=0)   # array level, Mojo kernel
```

The same object works everywhere an `xarray.DataArray` does. `mx.as_xarray(obj)`
strips the overrides and hands back a plain `xarray` object, which is the
escape hatch for checking parity by hand; the benchmarks get their reference
column by building an identical plain `xarray.DataArray` instead.

## What is covered

`src/ported.mojo`, in the order it reads:

| upstream | ported |
| --- | --- |
| `xarray/core/nputils.py` | `nanfirst`, `nanlast`, `nanmin`, `nanmax`, `nanmean`, `nanmedian`, `nanvar`, `nanstd`, `nanprod`, `nanargmin`, `nanargmax`, `nanquantile` |
| `xarray/computation/nanops.py` | `nansum`, `_maybe_null_out` |
| `xarray/computation/rolling.py` | `move_sum`, `move_mean`, `move_min`, `move_max`, `move_median`, `move_var`, `move_std`, `move_argmin`, `move_argmax`, `move_count`, `move_sum_of_squares` |
| `xarray/core/variable.py` | the `rolling_window` padding, and constant-mode `pad` |
| `xarray/core/nputils.py` | `nancumsum`, `nancumprod` |

`python/mojo_xarray/nputils.py` and `python/mojo_xarray/nanops.py` mirror the
reduction, cumulative and selection members of `xarray.core.nputils` and
`xarray.computation.nanops` function for function -- the two `nanfirst`/
`nanlast` included, down to upstream's "an all-missing slice selects position
0" convention. The rest of `xarray.core.nputils` is not a reduction
(`_create_method`'s option plumbing, `NumpyVIndexAdapter`,
`inverse_permutation`, `_nanpolyfit_1d`/`least_squares`) and is inherited.
`python/mojo_xarray/rolling.py` mirrors `xarray.computation.rolling` class for
class.

`python/mojo_xarray/api.py` overrides `DataArray.sum/mean/min/max/std/var/
median/prod/argmin/argmax`, `DataArray.rolling/coarsen` and the `Dataset`
equivalents. `Dataset` needs its own set, for a reason that is not visible from
reading it: `Dataset.__getitem__` builds the DataArray wrappers out of
`xarray.core.dataarray.DataArray` itself, and `Dataset.reduce` hands each
variable to `Variable.reduce` with `duck_array_ops.<name>`. So neither of the
`DataArray` overrides ever fires from `Dataset.mean` -- only `Dataset`'s own
reached the kernels at all before they were written. Each `Dataset` override
keeps upstream's signature keyword for keyword (`std`/`var` with `ddof`,
`sum`/`prod` with `min_count`, `min`/`max` with `numeric_only=False`) and its
`self.reduce(...)` call, and substitutes only the callable: `_create_nan_agg_method`
with the Mojo kernel in place of `nanops.nan<name>`, which is the same
substitution `inject_reduce_methods` performs for the `Coarsen` classes.

## What is not covered, and why

Each of these is a deliberate, named divergence.

- **Object dtype.** `nanops`' `_nan_minmax_object`, `_nan_argminmax_object`,
  `_nanmean_ddof_object` and `_nanvar_object` are dropped. Object arrays have no
  Mojo analogue and no fast path in upstream either; float input takes the
  kernel, everything else falls through to NumPy (see the next point).
- **Non-float64 input falls back to NumPy.** Upstream's result dtype for
  `nansum`/`nanmin`/`nanmax`/`nanprod`/`nanarg*`/`nancum*` is the input dtype, and
  accumulating `int64` in `float64` loses exactness above `2**53`. Rather than
  ship a silent numerical difference, `nputils` delegates every dtype but
  `float64` to the function upstream would have called. That includes
  `float32`, which NumPy accumulates in `float32` and this port would have
  accumulated in `float64`.
- **`DataArray.quantile` is not overridden.** xarray's `quantile` goes through
  `Dataset.quantile`, whose output layout (grouping, coordinate handling,
  method dispatch) is a large piece of machinery unrelated to the reduction. It
  is inherited unchanged. The kernel itself is ported and is reachable as
  `mojo_xarray.nanquantile` / `mx.nanquantile`, with exact layout parity against
  `np.nanquantile`.
- **`argmin`/`argmax` over a sequence of dims.** The dict-of-dims form
  (`da.argmin(dim=("a", "b"))`) goes through upstream's `stack` /
  `unravel_index` path, so it is inherited. The single-dim and `axis=` forms use
  the Mojo kernel.
- **Dask, cupy, sparse and other non-NumPy backends** are inherited, and they
  stay lazy: `rolling.construct` -- and so `rolling.reduce` -- hands anything
  that is not a C-contiguous `float64` NumPy array straight to upstream's
  `Variable.rolling_window`, so a dask-backed rolling object is not silently
  computed and a `float32` or `bool` one keeps the dtype upstream gives it.
  The kernels themselves are written against a C-contiguous `float64` buffer.
- **`skipna=False`** is inherited for every reduction, because it needs the
  `fillna` sentinel rather than a skip, which is a different kernel.
- **`nanquantile` with a tuple axis** and any `method` other than `"linear"` fall
  back to NumPy.
- **`rolling_exp`.** Upstream's `RollingExp` is a thin `apply_ufunc` wrapper
  around `numbagg.move_exp_nan{mean,sum}` and raises `ImportError` without
  numbagg, so there is no xarray-side algorithm to port: it is inherited, and
  it keeps raising the same error. `_get_alpha` is unchanged arithmetic and is
  likewise left to upstream.
- **`nanlast` reverses the slice.** Upstream negates an argmax over
  `values[::-1]`, and that reversal is kept rather than folded into the
  kernel, so the kernel sees the same argument `nanfirst`'s does. The cost is
  one contiguous copy of the input.
- **`mean`/`var`/`std` over a tuple of more than one axis** fall back to NumPy.
  They are not associative across axes when the per-slice valid counts differ,
  so a sequential axis loop would give the wrong answer.
- **`rolling.argmin`/`argmax` index convention.** They return an offset from the
  window's *left* edge, which is what upstream's materialised-window path
  produces and therefore what this port's parity tests check. When `bottleneck`
  happens to be installed, upstream instead takes the `bottleneck.move_argmin`
  fast path and returns `window - 1 - values`, i.e. right-edge-relative
  indices for full windows (GH #8541). That is a bottleneck quirk, not the
  algorithm, and it is not reproduced here. Upstream's `fillna=±inf` trick also
  means a NaN and a genuine `+inf` in the same window can tie; this port's
  ordering does not reproduce that tie.
- **Floating-point summation order.** Upstream reaches NumPy's pairwise
  summation; these kernels accumulate sequentially and stripe the work across
  threads, so `sum`/`mean`/`var`/`std` agree with upstream to roughly 1e-12
  relative, not bit for bit. `min`/`max`/`median`/`argmin`/`argmax`/`nancum*` are
  exact.
- **`nanquantile` is within one ulp of `np.nanquantile`.** The lerp is written
  as numpy's `_lerp` writes it, but the Mojo backend contracts
  `a + (b - a) * t` into an fma, which rounds differently from numpy's separate
  multiply-then-add. Results are exact whenever `q` lands on a sample.

## Benchmarks

Every number below comes from one `pixi run bench`, pasted verbatim; nothing
here is estimated or carried over from an earlier draft. The table is the
benchmark's own output, so re-running the task regenerates it.

`pixi run bench` takes the best of 7 reps after 2 warm-ups, holding the
machine-wide flock the pixi task takes. Every row is checked for value parity
against upstream (`yes` in the `ok` column means both columns agree to
`rtol=1e-10`, NaN for NaN; for the `Dataset` rows that means every data
variable). `bottleneck` and `numbagg` are not installed, so upstream's rolling
takes the materialised-window path -- the same default a `pip install xarray`
gives you.

The load average is printed with the table because it is the single biggest
caveat. The run below was taken at load 36.6 on a 72-core host: half the
machine was busy, so the `parallelize` kernels were still competing with
the rest of the machine for all 16 of their threads, though less severely than a
saturated host would force. The xarray column moves too, so the ratios are the
stable quantity and the absolute milliseconds are the ones to discount.

```
mojo-xarray vs xarray 2026.7.0
machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz (72 cores), kernel 6.8.0-142-generic
python 3.13.14, numpy 2.5.1
load average during the run: 36.6 on 72 cores; the host is shared, so this bounds how clean the numbers are

| case                               |       mojo |     xarray |  speedup | ok |
|------------------------------------|------------|------------|----------|-----|
| reduce 2M f64  sum                 |     1.18 ms |    22.79 ms |   19.25x | yes |
| reduce 2M f64  mean                |     4.11 ms |    13.43 ms |    3.27x | yes |
| reduce 2M f64  min                 |     0.74 ms |     0.74 ms |    1.01x | yes |
| reduce 2M f64  max                 |     0.88 ms |     0.92 ms |    1.04x | yes |
| reduce 2M f64  std                 |     5.04 ms |    23.27 ms |    4.62x | yes |
| reduce 2M f64  var                 |     5.81 ms |    25.71 ms |    4.42x | yes |
| reduce 2M f64  prod                |     4.29 ms |    12.12 ms |    2.82x | yes |
| reduce 500K f64 median             |     9.53 ms |     7.69 ms |    0.81x | yes |
| nputils 2M f64 nanfirst            |     0.70 ms |     1.16 ms |    1.67x | yes |
| nputils 2M f64 nanlast             |     3.62 ms |     3.16 ms |    0.87x | yes |
| reduce 64x1024 axis=1 sum          |     0.14 ms |     0.39 ms |    2.77x | yes |
| reduce 64x1024 axis=1 mean         |     0.23 ms |     0.43 ms |    1.83x | yes |
| reduce 64x1024 axis=1 min          |     0.15 ms |     0.14 ms |    0.96x | yes |
| reduce 64x1024 axis=1 max          |     0.14 ms |     0.15 ms |    1.05x | yes |
| reduce 64x1024 axis=1 std          |     0.26 ms |     0.76 ms |    2.98x | yes |
| reduce 64x1024 axis=1 var          |     0.30 ms |     0.83 ms |    2.80x | yes |
| reduce 64x1024 axis=1 median       |     1.18 ms |     3.81 ms |    3.22x | yes |
| rolling w=64  sum                  |    10.97 ms |    36.53 ms |    3.33x | yes |
| rolling w=64  mean                 |    13.52 ms |    66.38 ms |    4.91x | yes |
| rolling w=64  min                  |    10.42 ms |    43.20 ms |    4.14x | yes |
| rolling w=64  max                  |    11.97 ms |    42.77 ms |    3.57x | yes |
| rolling w=64  std                  |    15.99 ms |   393.59 ms |   24.62x | yes |
| rolling w=64  var                  |    14.11 ms |   400.37 ms |   28.37x | yes |
| rolling w=64  median               |    40.36 ms |  1279.78 ms |   31.71x | yes |
| rolling w=64  count                |    12.47 ms |    20.58 ms |    1.65x | yes |
| rolling w=8     mean               |     8.37 ms |    33.41 ms |    3.99x | yes |
| rolling w=256   mean               |    21.22 ms |   154.46 ms |    7.28x | yes |
| rolling w=512   mean               |    34.12 ms |   283.11 ms |    8.30x | yes |
| rolling 16x2048 w=32 median        |    18.17 ms |    45.78 ms |    2.52x | yes |
| rolling axis=0 w=64 mean           |     5.33 ms |    13.91 ms |    2.61x | yes |
| rolling construct w=64             |     0.34 ms |     0.51 ms |    1.48x | yes |
| coarsen 2 mean                     |     0.53 ms |     1.71 ms |    3.23x | yes |
| coarsen 64 std                     |     0.27 ms |     0.87 ms |    3.20x | yes |
| nanquantile q=0.1 axis=1 (vs np)   |     1.99 ms |    12.66 ms |    6.35x | yes |
| nanquantile q=0.5 axis=1 (vs np)   |     2.25 ms |    11.92 ms |    5.31x | yes |
| nanquantile q=0.9 axis=1 (vs np)   |     1.84 ms |    12.23 ms |    6.64x | yes |
| Dataset 4x64x1024 sum              |     0.55 ms |     1.50 ms |    2.73x | yes |
| Dataset 4x64x1024 mean             |     0.87 ms |     1.63 ms |    1.88x | yes |
| Dataset 4x64x1024 min              |     0.48 ms |     0.51 ms |    1.07x | yes |
| Dataset 4x64x1024 max              |     0.48 ms |     0.52 ms |    1.08x | yes |
| Dataset 4x64x1024 std              |     1.00 ms |     3.03 ms |    3.02x | yes |
| Dataset 4x64x1024 var              |     1.09 ms |     3.16 ms |    2.91x | yes |
| Dataset 4x64x1024 median           |     4.71 ms |    18.11 ms |    3.84x | yes |
| Dataset 4x64x1024 argmin           |     1.03 ms |     1.37 ms |    1.33x | yes |
| Dataset 4x64x1024 rolling w=64 sum |    43.32 ms |    50.85 ms |    1.17x | yes |
```

### What the table says

The shape of the result follows from where the work is. Three kernels carry the
design:

1. **The reduction tag is a compile-time parameter.** Every kernel body
   (`_axis_generic[mode]`, `_move_reduce[mode]`) tests `mode == M_SUM or
   mode == M_MEAN or ...`; as a parameter those comparisons are resolved before
   the loop is compiled rather than on every element, and the dispatch from the
   exported entry point to a mode is a chain of `if`s, once per call.
2. **The contiguous reductions (`inner == 1`) have a vector body.** `min`/`max`
   need no mask at all: the free `min`/`max` return whichever operand is not
   NaN, so a NaN never wins a comparison and never poisons an accumulator, and
   a slice with no observation leaves every lane NaN -- which *is* upstream's
   answer, so no count is needed. Sums mask NaNs with `min(v, 0) + max(v, 0)`,
   which is `v` with the NaNs turned into zeros in three vector ops. Four
   accumulators, because one is a dependency chain at the latency of the min
   rather than at its throughput.
3. **The moving-window kernels stripe by output element.** Output element `o`
   owns a private `win`-entry scratch and is reduced on one thread, so the
   branchy order statistics and the `count` that feeds `mean`'s divisor are the
   same code upstream runs, with the parallelism on the outside. That is why
   `std`/`var`/`median` gain two orders of magnitude over the materialised-window
   path while `sum` gains one.

`nanmedian`/`nanquantile` select rather than sort, which is O(n) against a
sort's O(n log n) and needs no ordering. `pad`, which `rolling.construct` is,
emits the last dimension a run at a time -- a fill band, a contiguous copy of
the input row, another fill band -- rather than re-deriving an index per element.

Where Mojo is slower than upstream, plainly:

- **`median` of a whole column: 0.81x.** The one row genuinely below parity on
  an idle host as well as a busy one, and it has been measured between 0.76x and
  0.82x across the runs taken here. Upstream reaches `np.nanmedian`, which is
  introselect -- O(n) with a much better constant than a Lomuto-partitioned
  quickselect, whose inner comparison is a coin flip the branch predictor cannot
  learn. Closing the rest means a branchless partition, and this dialect has no
  `select` on `SIMD` and no way to turn a `SIMD[DType.bool]` into numbers, so
  the branchless form would have to be hand-written on scalar selects. The
  *rolling* median, where there are many independent output elements to stripe
  across threads, is 32x faster than upstream on the same data.
- **`min`/`max` on a 2M column: 1.01x and 1.04x**, i.e. parity,
  and it has been between 0.88x and 1.04x across the runs taken here. Both sides
  stream 16 MB out of DRAM and NumPy's are SIMD too, so there is a few percent
  either way and no structural win available on a 1-flop-per-byte reduction.
  The strided `axis=1` rows of the same two operations sit either side of
  parity (0.96-1.05x) because there the kernel keeps a scalar loop and upstream is not
  the one that can vectorise.
- **`Dataset ... rolling w=64 sum` at 1.18x**, the weakest rolling row. Four
  variables of 64x1024 means four `DataArrayRolling` objects, each of which
  builds its own `_is_movable` decision and moves its own axes; the reduction
  itself is the same `mx_move_sum` that is 3.33x on the single-array row. This
  row is dominated by per-variable Python overhead, not by the kernel.
- **`mean` (3.3x) and `prod` (2.8x) on the same column.** Both need
  something the vector form cannot carry. `mean` needs the count of non-NaN
  observations, and this dialect has no lane-wise NaN mask to count with:
  `SIMD[DType.bool]` cannot be cast to numbers, `min`/`max` send a NaN to
  whichever operand they are given (so `min(v,0)+max(v,0)` cannot tell a NaN
  from a zero), and `+inf` and `-inf` are valid observations that any
  sentinel-based mask would swallow. `prod` has the same shape of problem in
  reverse -- a NaN has to become 1, not 0. `var`/`std` only need the count in
  their *first* pass, so their second pass -- the one that reads the data -- is
  vectorised, which is why they come out ahead of `mean` (4.4-4.6x) on the
  identical column.
- **`rolling construct` (1.48x) and `coarsen` (3.2x).** Both are
  allocation-bound: the kernel is a fraction of the call.

### What was tried and thrown away

- **Splitting a reduction's own run across threads** when there are too few
  output elements to keep the workers busy (the whole-column case: one output,
  2M inputs). It works and it is correct, and it is a loss: the `parallelize`
  launch costs more than the serial SIMD loop it replaces at every size this
  benchmark passes. Removed; the thresholds are on work, not on shape.
- **A branchless quickselect partition.** Unconditional pair of stores instead
  of a guarded swap: no measurable difference, so the simpler guarded swap
  stayed.

## Why there is no GPU path

`max.gpu.host` is available on this toolchain -- `std.gpu` is gone, but the host
API moved to `max`, and it compiles and runs. This port does not use it, and the
reason is worth stating rather than hiding.

The arithmetic intensity decides it per kernel. A moving-window reduction reads
`win` inputs per output element, so it does `win / 8` flops per byte read from
DRAM -- 8 at window 64, 64 at window 512, both comfortably above the ~2 flops
per byte where a transfer pays for itself. The whole-array reductions are the
opposite case: each input byte is read once and gets about one arithmetic op,
which is 0.1-0.25 flops per byte, and the input has to cross PCIe to reach the
device at all. So the reductions in this port stay on the CPU, and that is a
property of the kernels.

For the moving-window kernels the intensity is genuinely favourable, and a
device path is the obvious next pass: one `enqueue_create_buffer` pair, one
`enqueue_copy` in, one launch with one thread per output element and
`if o >= total: return` so the tail is written, one `enqueue_copy` out, one
`synchronize` -- the same arithmetic as `_move_reduce`, in the same order. It is
not here because this pass was required to be a faithful port first, and a
device path is a structural addition, not a fidelity fix. Claiming measured
device numbers without a `src/gpu.mojo` to point at would have been fabrication.

## How it works

**FFI.** One compilation unit, `src/capi.mojo`, emits the C ABI through
`@export`. `@export` refuses parametric functions and a pointer with an inferred
origin is parametric, so every buffer crosses as an `Int` address and is rebuilt
inside the wrapper as `Pointer[Float64, AnyOrigin[mut=True]]`. Every export is a
one-line forwarding stub: the upstream logic stays in `src/ported.mojo`, and the
shape bookkeeping lives in `python/mojo_xarray/_ops.py`, which is where Mojo
forces the divergence. The build is a single `mojo build --emit shared-lib`
over `capi.mojo` and `ported.mojo`, and it is flat in the number of functions, so
everything lives in one compilation unit; `ldd` on the result lists only `max`'s
own runtime, so the library loads on a machine with no GPU.

**Memory layout.** Everything is a C-contiguous `float64` buffer, and nothing
allocates inside the library. A reduction over `axis` is decomposed into
`(outer, n, inner)` -- output element `o` covers
`src[(o // inner) * n * inner + (o % inner) + k * inner]` for `k` in `[0, n)` --
which is the form NumPy's own reductions use and keeps the axis arithmetic out of
the kernel entirely. The `move_*` kernels work on the last axis with the same
decomposition, so a rolling window along any dimension is a `moveaxis` plus an
`ascontiguousarray` in the wrapper. `inner == 1` -- a reduction down the last
axis, or over a whole array -- is the contiguous case, and it is the one with a
vector body: `p.load[width=W]` with `W = simd_width_of[float64]()` (4 here), four
accumulators, and a scalar tail loop. Anything strided keeps the scalar loop,
because a strided gather is not what a vector load is for.

**The op tag.** `_axis_generic[mode]` and `_move_reduce[mode]` take the
operation as a compile-time parameter, so one kernel body serves all of them and
the `mode == ...` tests inside it are resolved before the loop is compiled
rather than per element. The dispatch from the exported `nan*`/`move_*` entry
points to a mode is a chain of `if`s, once per call.

**Order statistics.** A median wants one or two order statistics out of a
window, not an ordering, so `_quickselect` replaces the sort: a median-of-three
pivot swapped to the end, a Lomuto pass, and a recursion into whichever side
holds the wanted index. The median of a run is then one selection plus, for an
even count, a maximum over the lower half -- which the selection has already
bracketed, because everything below the selected index is no larger than it.
`nanquantile` is the same call for the two samples it interpolates between: the
upper one is the selected value, the lower is the smallest of the tail, and for
a high `q` that tail is a small fraction of the window.

**Parallelism.** A `parallelize` launch from `max.algorithm` costs a fixed
amount on this machine whatever work it is handed, and for every size the
benchmark passes that fixed cost exceeds the serial loop it would replace. So
the thresholds are on work rather than on shape: 4M elements for the scalar
kernels (`PAR_MIN_ELEMS`), 16M for the vector ones (`PAR_MIN_ELEMS_VEC`), and
8M element *visits* for the moving-window kernels (`PAR_MIN_VISITS`). Above a
threshold each family splits the work into one contiguous stripe per thread
(16; `mx.workers()` reports what the library was actually built with, and the
Python side sizes its scratch from that). Making the stripe index the thread
index is what lets the selection kernels give every thread a private `n`-entry
scratch buffer; sharing one scratch is a data race that a striped launch rules
out by construction. Below the threshold it is a plain serial loop.

**Fidelity.** `src/ported.mojo` reads top to bottom in upstream's order:
`nputils` primitives, then the `nanops` wrappers, then the `rolling` reductions,
then `variable.py`'s padding. Upstream's branch structure and arithmetic are
kept, including the details that are easy to get wrong: `_maybe_null_out`'s
`n - sum(mask) - min_count < 0` test, `nanvar`'s `where(count > ddof, ..., nan)`
guard, `nanprod` of an all-NaN slice being 1 rather than NaN,
the `(win // 2, win - 1 - win // 2)` / `(win - 1, 0)` pad derivation from
`rolling_window`, `min`/`max`'s
`fillna=±inf` all-missing sentinel, and `min_periods == 0` behaving as
`min_count == 1`.

`pixi run test` runs 1517 tests (122 in `test_api.py`, 793 in
`test_kernels.py`, 274 in `test_nanops.py`, 273 in `test_rolling.py`, 55 in
`test_exports.py`), all asserting numerical or behavioural parity rather than
"it ran":

- `tests/test_nanops.py` compares `mojo_xarray.nanops` / `nputils` against
  upstream's own `xarray.computation.nanops` and `xarray.core.nputils` on the
  same arrays, across shapes, axes, `ddof`, `min_count`, all-NaN slices, integer
  and float32 fallbacks, and the parallel path above the threshold.
  `nanfirst` / `nanlast` are compared on every axis of 1-D through 3-D inputs at
  three NaN fractions, with `keepdims`, a one-element tuple axis, infinities,
  all-missing slices, the `float32` fallback and the striped path, including
  upstream's "an all-missing slice selects position 0" convention.
- `tests/test_rolling.py` compares `Rolling`, `DataArrayRolling`,
  `DatasetRolling`, `Coarsen` and `DataArrayCoarsen` against a plain
  `xarray.DataArray` / `xarray.Dataset` twin, across window sizes 1-20, both
  `center` settings, every `min_periods`, n-d rolling, non-last axes, `construct`,
  `reduce(callable)`, all three `boundary` modes and both `side` modes, plus
  coordinate/attrs/name preservation and the integer and `skipna=False` fallbacks.
  `construct` is also checked *outside* the fast path: that a dask-backed
  rolling object is still dask afterwards, and that `float32`, `int32`, `int64`
  and `bool` come back with the dtype upstream gives them.
- `tests/test_api.py` checks the drop-in surface: the overridden methods have
  upstream's exact parameter names and defaults, every reduction matches on every
  axis, and the fast path really is the Mojo kernel (by spying on `_ops`).
  `Dataset` gets the same treatment -- signature parity including `ddof`,
  `min_count` and `numeric_only`, values on every axis, `skipna=False`, an
  integer variable keeping its dtype, the `argmin` dict-of-dims `ValueError`,
  and spies proving each of the ten `Dataset` reductions reaches its kernel.
- `tests/test_exports.py` calls **every one of the 30 `@export`ed symbols by
  name** and asserts the values the compiled library wrote into the output
  buffer. The rest of the suite reaches the kernels through the public API,
  which proves the port answers end to end but says nothing about *which* kernel
  produced an answer; a kernel that is wired up and completely wrong would pass
  every other test in this repository. References are upstream's own code where
  one exists -- `np.nanquantile`, the `bottleneck.move_*` definitions for the
  kernels upstream only reaches through bottleneck, `np.pad` -- and a sliding
  window view built the way `Variable.rolling_window` builds one otherwise.
  `mx_pad` is also driven through ctypes with a strided input row, which is the
  only way to reach the gather branch of `ported.pad`, and `mx_par_workers` is
  checked against the scratch size the Python wrapper allocates from it.
- `tests/test_kernels.py` pins the paths that are easy to get wrong, against
  NumPy rather than against a stored answer: the vector bodies and their scalar
  tails for run lengths either side of a vector (1, 2, 3, 4, 5, 7, ..., 129)
  and NaN fractions of 0, 5, 50 and 100 per cent; `min`/`max` exact rather than
  approximate; infinities treated as observations; the selection behind
  `nanmedian` and `nanquantile` on those same lengths; the striped path above
  its threshold; `pad` against `np.pad` for 1-D through 4-D shapes with padding
  on either side, on the serial and the striped path; and rolling windows longer
  than the row, which is where the live range of a window is empty on both ends.

`tests/conftest.py` loads the shared library at collection time, so a missing
`build/capi.so` stops the run with the build error rather than letting a
pure-Python path satisfy a test that is supposed to be about the compiled one.

## Layout

```
src/ported.mojo            the kernels, in upstream source order
src/capi.mojo              the C ABI: one forwarding stub per kernel
build/build.sh             mojo build --emit shared-lib -> dist/libmojo-xarray.so
python/mojo_xarray/
  _lib.py                  loads or rebuilds the .so
  _ops.py                  the (outer, n, inner) decomposition, pad and move glue
  nputils.py               port of xarray.core.nputils
  nanops.py                port of xarray.computation.nanops
  rolling.py               port of xarray.computation.rolling
  api.py                   DataArray / Dataset subclasses
tests/
  conftest.py              loads the .so; assert_same / array helpers
  test_exports.py          every @exported kernel, called by name
  test_kernels.py          the vector, selection and padding paths
  test_nanops.py           nanops / nputils against upstream's own functions
  test_rolling.py          Rolling / Coarsen against a plain xarray twin
  test_api.py              the drop-in surface, DataArray and Dataset
bench/bench.py             benchmarks, prints a markdown table
```
