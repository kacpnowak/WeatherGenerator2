#!/usr/bin/env python
"""Acceptance test for Task 7 (glorys_v2 / glorys_v2_eval stream configs).

Verifies the widened GLORYS v2 channel set (real training parquet, real
DataReaderMesh, no mocks):
  1. config/streams/glorys_v2: source_channels == target_channels == channels_order
     (from the yaml), channel count, the 6 OceanBench scoring depths present, the 4
     ice channels present, finite mean/stdev for every selected channel, and a spot
     read of a real 2010 window showing >=30 channels with finite data.
  2. config/streams/glorys_v2_eval/glorys.yml: parse-level only (yaml loads, channel
     lists identical to training v2, filenames == the not-yet-built v3 placeholder
     path) -- the file does not exist, so no reader is instantiated on it.
  3. ATMO: config/streams/glorys_v2/atmo.yml is parse-level only (anemoi type,
     unchanged from glorys/atmo.yml). config/streams/glorys_v2_eval/atmo.yml is
     instantiated (real parquet, mesh type): 6 source channels in checkpoint order,
     0 target channels.

Run:  .venv/bin/python .superpowers/sdd/wondrous-coalescing-taco/verify_task7.py
(needs WEATHERGEN_PRIVATE_REPO_PATH set, see task-1 verify script)
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from weathergen.common.config import load_streams
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.readers_extra.data_reader_mesh import DataReaderMesh

REPO = Path(__file__).resolve().parents[3]
V2_DIR = REPO / "config" / "streams" / "glorys_v2"
V2_EVAL_DIR = REPO / "config" / "streams" / "glorys_v2_eval"

TRAIN_PARQ = "/e/data1/climateai/hclimrep/data/glorys_025/glorys_025_mesh_fixed.parq"
V3_PLACEHOLDER = "/e/data1/climateai/hclimrep/data/glorys_forcings/glo_nowcast_025_v3_fixed.parq"

OCEANBENCH_DEPTHS = [0.494025, 47.3737, 92.3261, 222.475, 318.127, 541.089]
ICE_CHANNELS = ["siconc", "sithick", "usi", "vsi"]
EXCLUDED = ["mlotst", "bottomT", "pbo", "sob", "tob", "ist"]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def make_reader(stream_info, start="2010-01-01T00:00", end="2010-02-01T00:00"):
    tw = TimeWindowHandler(
        np.datetime64(start, "ns"),
        np.datetime64(end, "ns"),
        np.timedelta64(24, "h"),
        np.timedelta64(24, "h"),
    )
    return DataReaderMesh(tw, Path(stream_info["filenames"][0]), stream_info)


def main() -> int:
    # ============================================================ Section 1: glorys_v2
    streams_v2 = load_streams(V2_DIR)
    yaml_glorys = yaml.safe_load((V2_DIR / "glorys.yml").read_text())["GLORYS"]
    yaml_channels = list(yaml_glorys["channels_order"])

    glorys = make_reader(streams_v2["GLORYS"])

    check(
        "1. GLORYS source_channels == channels_order (yaml)",
        list(glorys.source_channels) == yaml_channels,
        f"n={len(glorys.source_channels)}",
    )
    check(
        "1b. GLORYS target_channels == channels_order (yaml)",
        list(glorys.target_channels) == yaml_channels,
        f"n={len(glorys.target_channels)}",
    )
    check(
        "1c. GLORYS source_channels == target_channels",
        list(glorys.source_channels) == list(glorys.target_channels),
    )

    n_channels = len(yaml_channels)
    print(f"\n--- GLORYS v2 channel count: {n_channels} ---\n")

    depth_suffixes_present = {
        d: any(f"_{d}m" in ch for ch in yaml_channels) for d in OCEANBENCH_DEPTHS
    }
    check(
        "2. all 6 OceanBench scoring depths present in channels_order",
        all(depth_suffixes_present.values()),
        f"{depth_suffixes_present}",
    )
    check(
        "3. all 4 ice channels present",
        all(c in yaml_channels for c in ICE_CHANNELS),
        f"ice channels in set: {[c for c in ICE_CHANNELS if c in yaml_channels]}",
    )
    check(
        "4. no excluded channel (mlotst/bottomT/pbo/sob/tob/ist) present",
        not any(c in yaml_channels for c in EXCLUDED),
        f"leaked: {[c for c in EXCLUDED if c in yaml_channels]}",
    )
    check("5. zos is channel 0", yaml_channels[0] == "zos", f"got {yaml_channels[0]!r}")

    # ------------------------------------------------- finite mean/stdev for every channel
    nonfinite_stats = []
    for ch in glorys.source_channels:
        idx = glorys.available_channels.index(ch)
        mu, sd = float(glorys.mean[idx]), float(glorys.stdev[idx])
        if not (np.isfinite(mu) and np.isfinite(sd)):
            nonfinite_stats.append((ch, mu, sd))
    check(
        "6. finite mean/stdev for every selected channel",
        not nonfinite_stats,
        f"non-finite: {nonfinite_stats[:10]}",
    )

    # ------------------------------------------------- real 2010 data read, per-channel finiteness
    rd = glorys.get_source(np.int64(5))
    print(f"GLORYS source block {rd.data.shape} at {rd.datetimes[0]}")
    finite_counts = {}
    for i, ch in enumerate(glorys.source_channels):
        col = rd.data[:, i]
        finite_counts[ch] = int(np.isfinite(col).sum())
    n_finite_channels = sum(1 for v in finite_counts.values() if v > 0)
    check(
        "7. >=30 channels have finite data in the 2010 spot-read window",
        n_finite_channels >= 30,
        f"{n_finite_channels}/{n_channels} channels have finite data",
    )
    ice_finite = {c: finite_counts.get(c, 0) for c in ICE_CHANNELS}
    check(
        "8. all 4 ice channels have finite data somewhere (polar-only is fine)",
        all(v > 0 for v in ice_finite.values()),
        f"{ice_finite}",
    )
    all_channels_finite = all(v > 0 for v in finite_counts.values())
    print(
        f"(informational) full-channel finite check: {sum(1 for v in finite_counts.values() if v > 0)}"
        f"/{n_channels} channels finite; all_finite={all_channels_finite}"
    )
    if not all_channels_finite:
        zero = sorted(c for c, v in finite_counts.items() if v == 0)
        print(f"  channels with 0 finite values in THIS window (may just be sparse/masked): {zero}")

    # ============================================================ Section 2: glorys_v2_eval (parse-only)
    eval_glorys_path = V2_EVAL_DIR / "glorys.yml"
    eval_glorys_yaml = yaml.safe_load(eval_glorys_path.read_text())["GLORYS"]
    check(
        "9. glorys_v2_eval/glorys.yml parses",
        eval_glorys_yaml is not None,
    )
    check(
        "10. glorys_v2_eval channels_order == training v2 channels_order",
        list(eval_glorys_yaml["channels_order"]) == yaml_channels,
    )
    check(
        "10b. glorys_v2_eval source == training v2 channels",
        list(eval_glorys_yaml["source"]) == yaml_channels,
    )
    check(
        "10c. glorys_v2_eval target == training v2 channels",
        list(eval_glorys_yaml["target"]) == yaml_channels,
    )
    check(
        "11. glorys_v2_eval filenames == not-yet-built v3 placeholder",
        list(eval_glorys_yaml["filenames"]) == [V3_PLACEHOLDER],
        f"got {eval_glorys_yaml['filenames']}",
    )
    check(
        "11b. glorys_v2_eval target_file == training parquet",
        eval_glorys_yaml["target_file"] == TRAIN_PARQ,
        f"got {eval_glorys_yaml.get('target_file')}",
    )
    v3_exists = Path(V3_PLACEHOLDER).exists()
    print(f"(informational) v3 placeholder file exists on disk: {v3_exists} (expected: False)")
    check(
        "12. v3 placeholder file does NOT exist yet (as documented in the LOUD comment)",
        not v3_exists,
        "if this now exists, the eval config's warning comment is stale",
    )

    # ============================================================ Section 3: ATMO
    v2_atmo_yaml = yaml.safe_load((V2_DIR / "atmo.yml").read_text())["ATMO"]
    check(
        "13. glorys_v2/atmo.yml parses and is unchanged (anemoi type, 6 source channels)",
        v2_atmo_yaml["type"] == "anemoi" and list(v2_atmo_yaml["source"]) == [
            "2t",
            "2d",
            "10u",
            "10v",
            "msl",
            "cp",
        ],
        f"type={v2_atmo_yaml.get('type')} source={v2_atmo_yaml.get('source')}",
    )

    streams_v2_eval = load_streams(V2_EVAL_DIR)
    atmo_eval = make_reader(streams_v2_eval["ATMO"])
    expected_atmo = ["10u", "10v", "2d", "2t", "cp", "msl"]
    check(
        "14. glorys_v2_eval ATMO source_channels == checkpoint order",
        list(atmo_eval.source_channels) == expected_atmo,
        f"got {list(atmo_eval.source_channels)}",
    )
    check(
        "15. glorys_v2_eval ATMO target_channels == [] (forcing-only)",
        list(atmo_eval.target_channels) == [],
        f"got {list(atmo_eval.target_channels)}",
    )

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
