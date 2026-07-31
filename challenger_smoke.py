# OceanBench challenger definition -- smoke variant.
#
# Same store as challenger.py, but restricted to two forecast start dates so a
# full OceanBench run can be exercised quickly.
#
# The store path is overridable via the WG_CHALLENGER_STORE environment
# variable (set it before invoking OceanBench, e.g. to evaluate a new model's
# challenger store without editing this file), falling back to the historical
# glorys_cont3_c1 store when unset.

import os

import xarray

CHALLENGER_ZARR = os.environ.get(
    "WG_CHALLENGER_STORE",
    "/e/scratch/hclimrep/nowak2/eval_output/wg_glorys_cont3_c1_challenger.zarr",
)

_full = xarray.open_zarr(CHALLENGER_ZARR, chunks={})

_wanted = [i for i in (0, 26) if i < _full.sizes["first_day_datetime"]]

challenger_dataset = _full.isel(first_day_datetime=_wanted)
