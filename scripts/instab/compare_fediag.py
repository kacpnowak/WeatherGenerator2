"""
Compare the GPU check rollouts of forecast.forcing_split / forecast.fe_noise
(scripts/instab/sbatch_fediag_check.sh).

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
        scripts/instab/compare_fediag.py OUT.txt LOG_GLOB

Leads 1-6 of TEMP, SALT, U, V, SURFACE (prediction, physical units, all points):
  fdchk0 (default, chunk 2) vs yr1fv003, fdchk1 (freeze, chunk 2) vs lcfz003: bitwise equal
  fdchk3 (split, 4 steps + resume to 6) vs fdchk2 (split, one chunk): bitwise equal
  fdchk4 (split 1/1) vs lcfz003: rounding only; fdchk2 vs fdchk1: differ
  fdchk6 (fe_noise + dropout, 4 steps + resume) vs fdchk5 (one chunk): bitwise equal
  fdchk5 vs yr1fv003 and fdchk7 (token_rms 1e-3 first step) vs lcfz003: differ
the content of the rollout_state.pt files, and the refusals found in the check logs.
"""

import glob
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_check import diff  # noqa: E402
from compare_landcells import RES, load  # noqa: E402

RUNS = ("yr1fv003", "lcfz003", "fdchk0", "fdchk1", "fdchk2", "fdchk3", "fdchk4", "fdchk5",
        "fdchk6", "fdchk7")


def state_line(run):
    if not (RES / run / "rollout_state.pt").exists():
        return f"{run}/rollout_state.pt: not written"
    st = torch.load(RES / run / "rollout_state.pt", map_location="cpu", weights_only=False)
    return (f"{run}/rollout_state.pt: keys {sorted(st)}, done_steps {st['done_steps']}, "
            f"land_cells {st.get('land_cells')}, forcing_split {st.get('forcing_split')}, "
            f"fe_noise {st.get('fe_noise')}")


def main():
    out = Path(sys.argv[1])
    runs = {r: load(r) for r in RUNS}
    lines = ["forecast.forcing_split / forecast.fe_noise GPU checks (fvars003, 2209-01-01, "
             "leads 1-6, predictions in physical units, all mesh points)", ""]
    diff(runs["fdchk0"], runs["yr1fv003"], "fdchk0 (default, chunk 2) vs yr1fv003", lines)
    diff(runs["fdchk1"], runs["lcfz003"], "fdchk1 (freeze, chunk 2) vs lcfz003 (freeze, chunk 30)",
         lines)
    diff(runs["fdchk3"], runs["fdchk2"],
         "fdchk3 (freeze, split 1/0.5, 4 steps resumed to 6) vs fdchk2 (one chunk)", lines)
    diff(runs["fdchk4"], runs["lcfz003"], "fdchk4 (freeze, split 1/1) vs lcfz003", lines)
    diff(runs["fdchk2"], runs["fdchk1"], "fdchk2 (freeze, split 1/0.5) vs fdchk1 (freeze)", lines)
    diff(runs["fdchk6"], runs["fdchk5"],
         "fdchk6 (fe_noise train_like 1e-4 + dropout, 4 steps resumed to 6) vs fdchk5 (one "
         "chunk)", lines)
    diff(runs["fdchk5"], runs["yr1fv003"], "fdchk5 (fe_noise train_like + dropout) vs yr1fv003",
         lines)
    diff(runs["fdchk7"], runs["lcfz003"],
         "fdchk7 (freeze, fe_noise token_rms 1e-3, first step) vs lcfz003", lines)
    lines.append("")
    for r in ("fdchk0", "fdchk1", "fdchk2", "fdchk3", "fdchk4", "fdchk5", "fdchk6", "fdchk7"):
        lines.append(state_line(r))
    lines += ["", "Errors raised in the check logs (expected: 2 forcing_split resume mismatches, "
              "2 fe_noise resume mismatches, 1 forcing_split + forcing_scale refusal):"]
    for f in sorted(glob.glob(sys.argv[2])):
        for ln in Path(f).read_text(errors="replace").splitlines():
            if re.search(r"ValueError: ", ln):
                lines.append(f"  {Path(f).name}: {ln.split('ValueError: ', 1)[1][:240]}")
                break
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))  # noqa: T201


if __name__ == "__main__":
    main()
