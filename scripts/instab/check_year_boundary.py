"""
Truth-archive continuity check for TEST 1: wet-area-weighted global mean temperature per
level and the volume mean (0-2150 m) on consecutive days, across year boundaries
2199/2200, 2204/2205, 2207/2208 and 2208/2209, plus the daily change inside a year.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/instab/check_year_boundary.py

Reads /work/ab0995/a270088/Kacper/weathergenertor/AWICM3/ocean_node/ocean_node_YYYY.zarr
(read only; rows are day-major, 126858 mesh nodes per day) -> OUT/test1_year_boundary.txt
"""

import sys
from pathlib import Path

import numpy as np
import zarr

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
import diag_fvars_year as dfy  # noqa: E402

ARCH = Path("/work/ab0995/a270088/Kacper/weathergenertor/AWICM3/ocean_node")
OUT = Path("/home/a/a270225/claude_api_tools/instab_tests")
MESH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
N = 126858
LEVELS = ("temp_2", "temp_75", "temp_450", "temp_975", "temp_2035")


def day_means(year, day, geo, cols):
    z = zarr.open(str(ARCH / f"ocean_node_{year}.zarr"), mode="r")
    names = list(z["data"].attrs["colnames"])
    rows = z["data"][day * N:(day + 1) * N]
    date = str(z["dates"][day * N][0])[:10]
    temps = [f"temp_{c.split('_')[1]}" for c in cols]
    nlev = len(temps)
    wet = np.stack([geo.wet(k) for k in range(nlev)])
    w = geo.area[:nlev] * wet
    dz = np.diff(geo.nz)[:nlev]
    T = np.stack([rows[:, names.index(c)] for c in temps]).astype(np.float64)
    T = np.where(wet, T, 0.0)
    lev = {c: float((w[k] * T[k]).sum() / w[k].sum()) for k, c in enumerate(temps)}
    vm = float((w * dz[:, None] * T).sum() / (w * dz[:, None]).sum())
    return date, lev, vm


def main():
    geo = dfy.Geometry(MESH)
    cols = [f"temp_{d}" for d in (2, 7, 15, 25, 35, 45, 55, 65, 75, 85, 95, 107, 125, 147, 175,
                                  210, 255, 310, 375, 450, 535, 630, 735, 850, 975, 1110, 1255,
                                  1415, 1600, 1810, 2035)]
    pairs = [((2199, 364), (2200, 0)), ((2204, 364), (2205, 0)), ((2207, 364), (2208, 0)),
             ((2208, 364), (2209, 0)), ((2208, 180), (2208, 181)), ((2209, 0), (2209, 1)),
             ((2209, 180), (2209, 181))]
    lines = ["Truth archive (ocean_node) continuity: wet-area mean T per level and volume mean",
             "0-2150 m on consecutive days (second minus first), across year boundaries and "
             "inside a year", ""]
    head = f"  {'day a':<12}{'day b':<12}{'vmT_b - vmT_a (K)':>20}" + "".join(
        f"{c + ' (K)':>16}" for c in LEVELS)
    lines.append(head)
    for (ya, da), (yb, db) in pairs:
        a_date, a_lev, a_vm = day_means(ya, da, geo, cols)
        b_date, b_lev, b_vm = day_means(yb, db, geo, cols)
        lines.append(f"  {a_date:<12}{b_date:<12}{b_vm - a_vm:>+20.5f}" + "".join(
            f"{b_lev[c] - a_lev[c]:>+16.5f}" for c in LEVELS))
    lines.append("")
    lines.append("Volume-mean T (0-2150 m) on 1 January of selected years (K)")
    for y in (2150, 2190, 2200, 2205, 2208, 2209):
        date, _, vm = day_means(y, 0, geo, cols)
        lines.append(f"  {date}  {vm:.5f}")
    txt = "\n".join(lines) + "\n"
    (OUT / "test1_year_boundary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
