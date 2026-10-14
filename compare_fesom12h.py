"""
Compare the forced (frcadd01) and baseline (frcbas01) 12 h fine-tunes of ykwx7bl2.

    .venv/bin/python compare_fesom12h.py [forced_run_id baseline_run_id]

Prints, from each run's results/<run_id>/<run_id>_train_metrics.json:
  1. validation MSE per mini-epoch (stream averages) and the forced/baseline ratio;
  2. for the latest validation both runs have: MSE per forecast step, averaged over channels;
  3. a few individual channels at the first, middle and last forecast step;
  4. the training MSE, averaged over each run's logged windows so far.
A ratio below 1 means the forced run has the lower error.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
STREAMS = ("FESOM_NODE", "FESOM_ELEM")
CHANNELS = {"FESOM_NODE": ("temp_2", "sss", "temp_95", "temp_535"), "FESOM_ELEM": ("u_2", "v_2")}
KEY = re.compile(r"^LossPhysical\.(?P<stream>[A-Z_]+)\.mse\.(?P<ch>.+)\.(?P<step>\d+)$")


def load(run_id: str) -> list[dict]:
    path = REPO / "results" / run_id / f"{run_id}_train_metrics.json"
    if not path.is_file():
        sys.exit(f"no metrics yet: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def per_step(row: dict, stream: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for key, value in row.items():
        m = KEY.match(key)
        if m and m["stream"] == stream:
            out.setdefault(int(m["step"]), []).append(value)
    return out


def fmt_ratio(a: float, b: float) -> str:
    return f"{a / b:6.3f}" if b else "   n/a"


def main(forced: str, baseline: str) -> None:
    runs = {"forced": load(forced), "baseline": load(baseline)}
    val = {k: [r for r in rows if r.get("stage") == "val"] for k, rows in runs.items()}
    train = {k: [r for r in rows if r.get("stage") == "train"] for k, rows in runs.items()}

    print(f"forced = {forced}, baseline = {baseline}\n")
    print("1. Validation MSE per mini-epoch (stream average)")
    print(f"   {'epoch':>5}  {'stream':10}  {'forced':>10}  {'baseline':>10}  {'ratio':>6}")
    for i in range(min(len(val["forced"]), len(val["baseline"]))):
        for s in STREAMS:
            k = f"LossPhysical.{s}.mse.avg"
            a, b = val["forced"][i].get(k, np.nan), val["baseline"][i].get(k, np.nan)
            print(f"   {i:>5}  {s:10}  {a:10.5f}  {b:10.5f}  {fmt_ratio(a, b)}")
    if not val["forced"] or not val["baseline"]:
        print("   (no validation in one of the runs yet)")
        return

    n = min(len(val["forced"]), len(val["baseline"])) - 1
    vf, vb = val["forced"][n], val["baseline"][n]
    print(f"\n2. Validation MSE per forecast step, mini-epoch {n} (mean over channels)")
    for s in STREAMS:
        pf, pb = per_step(vf, s), per_step(vb, s)
        print(f"   {s}: step  forced     baseline   ratio")
        for step in sorted(set(pf) & set(pb)):
            a, b = float(np.mean(pf[step])), float(np.mean(pb[step]))
            print(f"   {'':{len(s)}}  {step:>4}  {a:.5f}  {b:.5f}  {fmt_ratio(a, b)}")

    print(f"\n3. Selected channels, mini-epoch {n}")
    steps = sorted({int(KEY.match(k)["step"]) for k in vf if KEY.match(k)})
    picks = sorted({steps[0], steps[len(steps) // 2], steps[-1]}) if steps else []
    for s, chans in CHANNELS.items():
        for ch in chans:
            cells = []
            for step in picks:
                k = f"LossPhysical.{s}.mse.{ch}.{step}"
                if k in vf and k in vb:
                    cells.append(f"step {step}: {vf[k]:.4g} vs {vb[k]:.4g} ({fmt_ratio(vf[k], vb[k]).strip()})")
            if cells:
                print(f"   {s} {ch:9}  " + "  |  ".join(cells))

    print("\n4. Training MSE (mean of logged windows so far)")
    for s in STREAMS:
        k = f"LossPhysical.{s}.mse.avg"
        a = np.nanmean([r.get(k, np.nan) for r in train["forced"]]) if train["forced"] else np.nan
        b = np.nanmean([r.get(k, np.nan) for r in train["baseline"]]) if train["baseline"] else np.nan
        print(f"   {s:10}  forced {a:.5f} ({len(train['forced'])} windows)  "
              f"baseline {b:.5f} ({len(train['baseline'])} windows)  ratio {fmt_ratio(a, b)}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["frcadd01", "frcbas01"]
    main(*args)
