# OceanBench challenger definition.
#
# OceanBench executes this file and reads the module-level `challenger_dataset`
# variable. Keep it side-effect free: no prints, no asserts (papermill would
# fail on them).
#
# The store is produced by export_glorys_nc.py.
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

challenger_dataset = xarray.open_zarr(CHALLENGER_ZARR, chunks={})
