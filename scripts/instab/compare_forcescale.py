"""
Compare the GPU check rollouts of forecast.forcing_scale
(scripts/instab/sbatch_forcescale_check.sh).

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
        scripts/instab/compare_forcescale.py OUT.txt

Leads 1-6 of TEMP, SALT, U, V, SURFACE (prediction, physical units, all points):
  fschk0 (default, chunk 2) and fschk1 (forcing_scale 1.0) vs yr1fv003: bitwise equal
  fschk2 (freeze) and fschk5 (freeze, forcing_scale 1.0) vs lcfz003: bitwise equal
  fschk4 (freeze, 0.5, one chunk) vs fschk3 (4 steps chunk 2 + resume to 6): bitwise equal
  fschk4 vs fschk2: differ
and the content of the rollout_state.pt files (forcing_scale key only for alpha != 1).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_check import diff  # noqa: E402
from compare_landcells import RES, load  # noqa: E402

RUNS = ("yr1fv003", "lcfz003", "fschk0", "fschk1", "fschk2", "fschk3", "fschk4", "fschk5")


def state_line(run):
    if not (RES / run / "rollout_state.pt").exists():
        return f"{run}/rollout_state.pt: not written (single-chunk rollout)"
    st = torch.load(RES / run / "rollout_state.pt", map_location="cpu", weights_only=False)
    return (f"{run}/rollout_state.pt: keys {sorted(st)}, done_steps {st['done_steps']}, "
            f"land_cells {st.get('land_cells')}, forcing_scale {st.get('forcing_scale')}")


def main():
    out = Path(sys.argv[1])
    runs = {r: load(r) for r in RUNS}
    lines = ["forecast.forcing_scale GPU checks (fvars003, 2209-01-01, leads 1-6, predictions in "
             "physical units, all mesh points)", ""]
    diff(runs["fschk0"], runs["yr1fv003"], "fschk0 (default, chunk 2) vs yr1fv003", lines)
    diff(runs["fschk1"], runs["yr1fv003"], "fschk1 (forcing_scale 1.0, chunk 2) vs yr1fv003",
         lines)
    diff(runs["fschk2"], runs["lcfz003"], "fschk2 (freeze, chunk 2) vs lcfz003 (freeze, chunk 30)",
         lines)
    diff(runs["fschk5"], runs["lcfz003"], "fschk5 (freeze, forcing_scale 1.0) vs lcfz003", lines)
    diff(runs["fschk3"], runs["fschk4"],
         "fschk3 (freeze, 0.5, 4 steps resumed to 6) vs fschk4 (freeze, 0.5, one chunk)", lines)
    diff(runs["fschk4"], runs["fschk2"], "fschk4 (freeze, 0.5) vs fschk2 (freeze, 1.0)", lines)
    lines.append("")
    for r in ("fschk0", "fschk1", "fschk2", "fschk3", "fschk4", "fschk5"):
        lines.append(state_line(r))
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))  # noqa: T201


if __name__ == "__main__":
    main()
