"""
Compare the GPU check rollouts of the latent filter (scripts/instab/sbatch_instab_check.sh).

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/compare_check.py OUT.txt

Leads 1-6 of TEMP, SALT, SURFACE (prediction, physical units, all points):
  inschk0 (no filter, chunk 2)        vs yr1fv003 (no filter, chunk 30, original rollout)
  inschk1 (second_difference, 1 chunk) vs inschk2 (chunk 2) vs inschk3 (4 steps + resume to 6)
  inschk1 vs yr1fv003: lead 1 equal (the first step is not filtered), later leads differ
and the content of inschk3/rollout_state.pt.
"""

import sys
from pathlib import Path

import numpy as np
import torch

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from paper_eval.merged_zarr import open_merged_ranks  # noqa: E402

RES = Path("/work/ab0995/a270225/WeatherGenerator2/results")
STREAMS = ("TEMP", "SALT", "SURFACE")


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


def diff(a, b, label, lines):
    lines.append(f"{label}")
    for (st, lead) in sorted(k for k in a if len(k) == 2):
        x, y = a[(st, lead)], b[(st, lead)]
        fin = np.isfinite(x) & np.isfinite(y)
        same_nan = np.array_equal(np.isnan(x), np.isnan(y))
        d = np.abs(x[fin] - y[fin]).max() if fin.any() else np.nan
        ref = np.abs(y[fin]).max()
        tgt_eq = np.array_equal(a[(st, lead, "t")], b[(st, lead, "t")], equal_nan=True)
        lines.append(f"  {st:8s} lead {lead}: max|diff| = {d:.3e} (max|ref| {ref:.3e}), "
                     f"bitwise equal: {np.array_equal(x, y, equal_nan=True)}, nan pattern equal: "
                     f"{same_nan}, targets equal: {tgt_eq}")


def main():
    out = Path(sys.argv[1])
    runs = {r: load(r) for r in ("yr1fv003", "inschk0", "inschk1", "inschk2", "inschk3")}
    lines = ["Latent filter GPU checks (fvars003, 2209-01-01, leads 1-6, predictions in physical"
             " units, all mesh points)", ""]
    diff(runs["inschk0"], runs["yr1fv003"], "inschk0 (no filter, chunk 2) vs yr1fv003", lines)
    diff(runs["inschk2"], runs["inschk1"],
         "inschk2 (second_difference 0.25, chunk 2) vs inschk1 (one chunk)", lines)
    diff(runs["inschk3"], runs["inschk2"],
         "inschk3 (4 steps, resumed to 6) vs inschk2 (chunk 2, 6 steps)", lines)
    diff(runs["inschk1"], runs["yr1fv003"],
         "inschk1 (second_difference 0.25) vs yr1fv003 (no filter)", lines)
    st = torch.load(RES / "inschk3" / "rollout_state.pt", map_location="cpu", weights_only=False)
    prev = st.get("latent_filter_prev")
    lines += ["", "inschk3/rollout_state.pt: keys " + str(sorted(st)),
              f"  done_steps {st['done_steps']}, latent_filter {st.get('latent_filter')}, "
              f"latent_filter_prev {None if prev is None else tuple(prev.shape)}"]
    st0 = torch.load(RES / "inschk0" / "rollout_state.pt", map_location="cpu", weights_only=False)
    lines += [f"inschk0/rollout_state.pt (no filter): keys {sorted(st0)}"]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
