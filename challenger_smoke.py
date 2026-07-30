# OceanBench challenger definition -- smoke variant.
#
# Same store as challenger.py, but restricted to two forecast start dates so a
# full OceanBench run can be exercised quickly.

import xarray

CHALLENGER_ZARR = "/e/scratch/hclimrep/nowak2/eval_output/wg_glorys_cont3_c1_challenger.zarr"

_full = xarray.open_zarr(CHALLENGER_ZARR, chunks={})

_wanted = [i for i in (0, 26) if i < _full.sizes["first_day_datetime"]]

challenger_dataset = _full.isel(first_day_datetime=_wanted)
