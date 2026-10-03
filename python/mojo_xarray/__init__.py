"""mojo-xarray: xarray's labelled-array reductions and rolling windows, in Mojo.

    import mojo_xarray as mx

    da = mx.DataArray(data, dims="time", coords={"time": times})
    da.rolling(time=7, center=True).mean()
    da.mean(dim="time")

The kernel names mirror upstream's: `mx.nanops.nansum` is
`xarray.computation.nanops.nansum`, `mx.rolling.DataArrayRolling` is
`xarray.computation.rolling.DataArrayRolling`, and so on.  See the README for
what is covered and what is not.
"""

from __future__ import annotations

import xarray as xr

from . import nanops, nputils, rolling
from ._ops import workers
from .api import DataArray, Dataset
from .nanops import (
    nanargmax,
    nanargmin,
    nancumprod,
    nancumsum,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanprod,
    nanstd,
    nansum,
    nanvar,
)
from .nputils import nanfirst, nanlast, nanquantile
from .rolling import (
    Coarsen,
    DataArrayCoarsen,
    DataArrayRolling,
    DatasetCoarsen,
    DatasetRolling,
    Rolling,
)

__version__ = "0.1.0"

__all__ = [
    "DataArray", "Dataset",
    "nanops", "nputils", "rolling",
    "nanmin", "nanmax", "nansum", "nanmean", "nanmedian", "nanvar", "nanstd",
    "nanprod", "nancumsum", "nancumprod", "nanargmin", "nanargmax",
    "nanquantile", "nanfirst", "nanlast",
    "Rolling", "DataArrayRolling", "DatasetRolling",
    "Coarsen", "DataArrayCoarsen", "DatasetCoarsen", "workers",
    "as_xarray",
]


def as_xarray(obj):
    """View any xarray object as the plain upstream class, bypassing the Mojo
    overrides -- the escape hatch for checking parity and for the benchmarks."""
    if isinstance(obj, xr.DataArray):
        return xr.DataArray(obj.variable, obj.coords, name=obj.name, attrs=obj.attrs)
    if isinstance(obj, xr.Dataset):
        return xr.Dataset(obj._variables, obj._coord_names, obj._dims,
                          attrs=obj.attrs)
    raise TypeError(f"not an xarray object: {type(obj)!r}")
