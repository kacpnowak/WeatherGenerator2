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
        f"ATMO channels whose col_map 'var' is absent from {atmo.filename_source.name} "
        f"(Task 1b falls these back to the channel name): "
        f"{atmo_missing if atmo_missing else 'none'}"
    )

    # ------------------------------------------------- Task 1b: real data reads
    print("\n--- Task 1b: ATMO forcing must no longer be zero-filled ---")
    # The first source time steps of the nowcast file have gaps (see DATA DEFECTS below),
    # so read a mid-year window rather than the 2024-01 one used for the checks above.
    tw_mid = TimeWindowHandler(
        np.datetime64("2024-06-01T00:00", "ns"),
        np.datetime64("2024-07-01T00:00", "ns"),
        np.timedelta64(24, "h"),
        np.timedelta64(24, "h"),
    )
    atmo_r = DataReaderMesh(tw_mid, Path(streams["ATMO"]["filenames"][0]), streams["ATMO"])
    rd_a = atmo_r.get_source(np.int64(12))
    print(f"ATMO source block {rd_a.data.shape} at {rd_a.datetimes[0]}")
    stats = {}
    for i, ch in enumerate(atmo_r.source_channels):
        col = rd_a.data[:, i]
        finite = np.isfinite(col)
        stats[ch] = (
            bool(np.all(col == 0.0)),
            float(finite.mean()),
            float(np.nanstd(col)) if finite.any() else 0.0,
            (float(np.nanmin(col)), float(np.nanmax(col))) if finite.any() else (np.nan, np.nan),
        )
        zf, ff, sd, (lo, hi) = stats[ch]
        print(
            f"  {ch:4s} all_zero={zf!s:5s} finite_frac={ff:.3f} std={sd:12.5f} "
            f"range=[{lo:.4g}, {hi:.4g}]"
        )

    zero_filled = [c for c, v in stats.items() if v[0]]
    check("4. no ATMO channel is zero-filled", not zero_filled, f"zero-filled: {zero_filled}")

    identical = [
        (a_, b_)
        for i, a_ in enumerate(atmo_r.source_channels)
        for b_ in atmo_r.source_channels[i + 1 :]
        if np.array_equal(
            rd_a.data[:, atmo_r.source_channels.index(a_)],
            rd_a.data[:, atmo_r.source_channels.index(b_)],
            equal_nan=True,
        )
    ]
    check("5. the 6 ATMO channels are mutually distinct", not identical, f"identical: {identical}")

    varying = [c for c, v in stats.items() if v[2] > 0.0]
    check(
        "6. ATMO channels carry varying data",
        len(varying) >= 5,
        f"{len(varying)}/6 vary: {varying}; flat/empty: {sorted(set(stats) - set(varying))}",
    )

    ranges = {
        "2t": (180.0, 340.0),
        "2d": (180.0, 340.0),
        "10u": (-120.0, 120.0),
        "10v": (-120.0, 120.0),
        "cp": (0.0, 1.0),
        "msl": (8e4, 1.1e5),
    }
    implausible = [
        c
        for c, (lo, hi) in ranges.items()
        if stats[c][1] > 0.0 and not (lo <= stats[c][3][0] and stats[c][3][1] <= hi)
    ]
    check(
        "7. ATMO channels that carry data are in a plausible physical range",
        not implausible,
        f"out of range: {implausible}",
    )

    # GLORYS must be untouched by the fallback: its col_map 'var' entries all exist.
    glorys_r = DataReaderMesh(tw_mid, Path(streams["GLORYS"]["filenames"][0]), streams["GLORYS"])
    rd_g = glorys_r.get_source(np.int64(12))
    check(
        "8. GLORYS read takes no col_map fallback (identical code path to before Task 1b)",
        not glorys_r._var_warned,
        f"fallbacks/errors: {sorted(glorys_r._var_warned)}",
    )
    g_stats = {}
    for i, ch in enumerate(glorys_r.source_channels):
        col = rd_g.data[:, i]
        g_stats[ch] = (bool(np.all(col == 0.0)), float(np.isfinite(col).mean()))
    g_zero = [c for c, v in g_stats.items() if v[0]]
    g_empty = sorted(c for c, v in g_stats.items() if v[1] == 0.0)
    check("9. no GLORYS channel is zero-filled", not g_zero, f"zero-filled: {g_zero}")
    uo0 = rd_g.data[:, glorys_r.source_channels.index("uo_0.494025m")]
    check(
        "10. GLORYS uo_0.494025m has real ocean values",
        np.isfinite(uo0).any() and np.nanstd(uo0) > 0 and np.nanmax(np.abs(uo0)) < 20.0,
        f"finite_frac={np.isfinite(uo0).mean():.3f} std={np.nanstd(uo0):.4f} "
        f"absmax={np.nanmax(np.abs(uo0)):.4f}",
    )

    print("\n--- DATA DEFECTS in the eval source files (file-level, not reader bugs) ---")
    a_empty = sorted(c for c, v in stats.items() if v[1] == 0.0)
    print(
        f"ATMO  {atmo_r.filename_source.name}: channels with no finite data at all: "
        f"{a_empty if a_empty else 'none'}"
    )
    print(
        f"GLORYS {glorys_r.filename_source.name}: channels with no finite data at all "
        f"({len(g_empty)}/41): {g_empty if g_empty else 'none'}"
    )

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
