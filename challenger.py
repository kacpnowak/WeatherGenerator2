# OceanBench challenger definition.
#
# OceanBench executes this file and reads the module-level `challenger_dataset`
# variable. Keep it side-effect free: no prints, no asserts (papermill would
# fail on them).
#
# The store is produced by export_glorys_nc.py.

import xarray

CHALLENGER_ZARR = "/e/scratch/hclimrep/nowak2/eval_output/wg_glorys_cont3_c1_challenger.zarr"

challenger_dataset = xarray.open_zarr(CHALLENGER_ZARR, chunks={})
