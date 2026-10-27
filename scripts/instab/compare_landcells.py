"""
Compare the GPU check rollouts of forecast.land_cells (scripts/instab/sbatch_landcells_check.sh).

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/compare_landcells.py OUT.txt

Leads 1-6 of TEMP, SALT, U, V, SURFACE (prediction, physical units, all points):
  lcchk0 (default, chunk 2) vs yr1fv003 (default, chunk 30, original rollout): bitwise equal
  lcchk1 (freeze, 1 chunk) vs lcchk2 (chunk 2) vs lcchk3 (4 steps + resume to 6)
  lcchk4 (no_forcing, 1 chunk) vs lcchk5 (chunk 2)
  lcchk1 / lcchk4 vs yr1fv003: differ
and the content of the rollout_state.pt files (freeze: land rows of the carried latent equal
the saved initial land tokens).
"""

import sys
from pathlib import Path

import numpy as np
import torch

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from compare_check import diff  # noqa: E402

from paper_eval.merged_zarr import open_merged_ranks  # noqa: E402

RES = Path("/work/ab0995/a270225/WeatherGenerator2/results")
STREAMS = ("TEMP", "SALT", "U", "V", "SURFACE")
RUNS = ("yr1fv003", "lcchk0", "lcchk1", "lcchk2", "lcchk3", "lcchk4", "lcchk5")


def load(run, leads=range(1, 7)):
    root = open_merged_ranks(RES / run, 0)
    s = root.store.samples[0]
    out = {}
    for st in STREAMS:
        for lead in leads:
            g = root[f"{s}/{st}/{lead}/prediction"]
            out[(st, lead)] = np.asarray(g["data"])[..., 0].astype(np.float64)
            out[(st, lead, "t")] = np.asarray(root[f"{s}/{st}/{lead}/target"]["data"])
    return out


def state_lines(run):
    st = torch.load(RES / run / "rollout_state.pt", map_location="cpu", weights_only=False)
    lines = [f"{run}/rollout_state.pt: keys {sorted(st)}, done_steps {st['done_steps']}, "
             f"land_cells {st.get('land_cells')}"]
    init = st.get("land_cells_init")
    if init is not None:
        z = st["latent_state"].z_pre_norm
        land = np.flatnonzero(~ocean_mask())
        lines.append(f"  land_cells_init {tuple(init.shape)} {init.dtype}; carried latent "
                     f"{tuple(z.shape)} {z.dtype}; land rows of the carried latent == "
                     f"land_cells_init: {torch.equal(z[:, land].to(init.dtype), init)}")
    return lines


def ocean_mask():
    from weathergen.model.land_cells import land_cell_mask

    mesh = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
    return ~land_cell_mask({"TEMP": {"type": "fesom", "mesh_path": mesh}}, 5)


def main():
    out = Path(sys.argv[1])
    runs = {r: load(r) for r in RUNS}
    lines = ["forecast.land_cells GPU checks (fvars003, 2209-01-01, leads 1-6, predictions in "
             "physical units, all mesh points)", ""]
    diff(runs["lcchk0"], runs["yr1fv003"], "lcchk0 (default, chunk 2) vs yr1fv003", lines)
    diff(runs["lcchk2"], runs["lcchk1"], "lcchk2 (freeze, chunk 2) vs lcchk1 (one chunk)", lines)
    diff(runs["lcchk3"], runs["lcchk2"], "lcchk3 (freeze, 4 steps resumed to 6) vs lcchk2", lines)
    diff(runs["lcchk5"], runs["lcchk4"], "lcchk5 (no_forcing, chunk 2) vs lcchk4 (one chunk)",
         lines)
    diff(runs["lcchk1"], runs["yr1fv003"], "lcchk1 (freeze) vs yr1fv003 (default)", lines)
    diff(runs["lcchk4"], runs["yr1fv003"], "lcchk4 (no_forcing) vs yr1fv003 (default)", lines)
    lines.append("")
    for r in ("lcchk0", "lcchk2", "lcchk3", "lcchk5"):
        lines += state_lines(r)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
