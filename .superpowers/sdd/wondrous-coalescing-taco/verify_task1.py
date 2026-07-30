#!/usr/bin/env python
"""Acceptance test for Task 1 (channels_order + glorys_eval stream configs).

Instantiates the real DataReaderMesh on the two fixed stream configs in
config/streams/glorys_eval/ (real parquet/virtual-zarr files, no mocks) and
checks that the resulting channel order and normalization statistics match the
checkpoint glorys_cont3_c1.

Run:  .venv/bin/python .superpowers/sdd/wondrous-coalescing-taco/verify_task1.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from weathergen.common.config import load_streams
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.readers_extra.data_reader_mesh import DataReaderMesh

REPO = Path(__file__).resolve().parents[3]
STREAMS_DIR = REPO / "config" / "streams" / "glorys_eval"
CKPT_JSON = Path(
    "/e/scratch/weatherai/shared_work/models/glorys_cont3_c1/model_glorys_cont3_c1_latest.json"
)

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def make_reader(stream_info):
    tw = TimeWindowHandler(
        np.datetime64("2024-01-01T00:00", "ns"),
        np.datetime64("2024-02-01T00:00", "ns"),
        np.timedelta64(24, "h"),
        np.timedelta64(24, "h"),
    )
    return DataReaderMesh(tw, Path(stream_info["filenames"][0]), stream_info)


def main() -> int:
    ckpt = json.loads(CKPT_JSON.read_text())["streams"]
    streams = load_streams(STREAMS_DIR)

    # ---------------------------------------------------------------- GLORYS
    glorys = make_reader(streams["GLORYS"])
    expected_glorys = list(ckpt["GLORYS"]["train_source_channels"])
    got = list(glorys.source_channels)
    check(
        "1. GLORYS source_channels == checkpoint train_source_channels",
        got == expected_glorys,
        f"n={len(got)}, first={got[:2]}"
        if got == expected_glorys
        else f"got {len(got)} channels, first 5 {got[:5]}",
    )
    check(
        "1b. GLORYS target_channels == checkpoint train_target_channels",
        list(glorys.target_channels) == list(ckpt["GLORYS"]["train_target_channels"]),
        f"n={len(glorys.target_channels)}",
    )

    # ------------------------------------------------------------------ ATMO
    atmo = make_reader(streams["ATMO"])
    expected_atmo = ["10u", "10v", "2d", "2t", "cp", "msl"]
    got_atmo = list(atmo.source_channels)
    check(
        "2. ATMO source_channels == ['10u','10v','2d','2t','cp','msl']",
        got_atmo == expected_atmo,
        f"got {got_atmo}",
    )

    # ------------------------------------------------- training stats restored
    zos_mean = float(glorys.mean[glorys.source_idx[0]])
    zos_std = float(glorys.stdev[glorys.source_idx[0]])
    ok = (
        glorys.source_channels[0] == "zos"
        and zos_mean != 0.0
        and np.isclose(zos_mean, -0.29641, rtol=1e-3)
    )
    check(
        "3. GLORYS zos mean == training-file stats (-0.29641), not the NaN nowcast stats",
        ok,
        f"mean={zos_mean!r} stdev={zos_std!r} channel={glorys.source_channels[0]!r}",
    )
    # the same for the other channels whose nowcast stats are NaN (so_*)
    nan_in_nowcast = [c for c in expected_glorys if c.startswith("so_")]
    bad = [
        c for c in nan_in_nowcast if float(glorys.mean[glorys.available_channels.index(c)]) == 0.0
    ]
    check("3b. all so_* channels got non-zero (training) means", not bad, f"zero-mean: {bad}")

    # ------------------------------------------------------- R2 diagnostics
    print("\n--- R2 diagnostics (informational) ---")
    sample = glorys.col_map["zos"]
    print(f"col_map value type: {type(sample).__name__} -> {sample}")
    print("col_map values are {'var','sel'} descriptors, NOT per-file integer column indices;")
    print("_load_block_from_ds resolves data by channel NAME, so reordering col_map is safe.")

    import fsspec
    import xarray as xr

    from weathergen.readers_extra.data_reader_mesh import _open_dataset_fixed

    def probe_colmap(p):
        m = fsspec.get_mapper("reference://", fo=str(p), remote_protocol="file")
        with xr.open_dataset(m, engine="zarr", chunks={}, consolidated=False, decode_cf=False) as d:
            cm = d.attrs.get("weathergen_col_map", {})
            return json.loads(cm) if isinstance(cm, str) else cm

    cm_src = probe_colmap(glorys.filename_source)
    cm_trg = probe_colmap(glorys.filename_target)
    differing = [c for c in expected_glorys if cm_src.get(c) != cm_trg.get(c)]
    print(
        f"source/target col_map entries differing among the {len(expected_glorys)} used channels: "
        f"{differing if differing else 'none'} "
        "(none => the target probe's update() cannot mis-address source reads)"
    )

    # time-axis alignment between source and target file (target reads reuse SOURCE time idxs)
    mapper = fsspec.get_mapper(
        "reference://", fo=str(glorys.filename_target), remote_protocol="file"
    )
    with _open_dataset_fixed(mapper, engine="zarr", chunks={}, consolidated=False) as dst:
        t_trg = dst.time.values
    for i in range(10):
        t_idxs, dtr = glorys._get_persistent_time_idxs(np.int64(i))
        if len(t_idxs):
            print(f"window {i} = [{dtr.start}, {dtr.end}); t_idxs={t_idxs}")
            print(f"  source-file time at those idxs: {glorys._time_values_cached[t_idxs]}")
            print(f"  target-file time at those idxs: {t_trg[t_idxs]}  <-- target reads reuse the")
            print("      SOURCE time indices; the two files do NOT share a time axis.")
            break

    # ATMO: does the col_map in the eval IFS file actually resolve to variables in the file?
    atmo._lazy_init()
    atmo_missing = [
        (ch, atmo.col_map[ch]["var"])
        for ch in atmo.source_channels
        if atmo.col_map[ch]["var"] not in atmo.ds_source
    ]
    print(
        f"ATMO channels whose col_map 'var' is absent from {atmo.filename_source.name}: "
        f"{atmo_missing if atmo_missing else 'none'}"
    )
    if atmo_missing:
        rd = atmo.get_source(np.int64(2))
        print(
            f"  -> ATMO source block {rd.data.shape}: per-channel nanmean="
            f"{np.nanmean(rd.data, axis=0)} (silently zero-filled)"
        )

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
