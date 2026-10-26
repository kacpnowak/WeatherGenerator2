"""Tests for scoring runs of the per-variable stream layout (TEMP, SALT, SURFACE, U, V).

    OMP_NUM_THREADS=4 .venv/bin/python -m pytest paper_eval/test_stream_layout.py -p no:cacheprovider

Everything is synthetic and built in `tmp_path`: a 6-node / 4-element mesh file, a
results archive (`validation_chkpt00000_rank0000.zip`) with streams TEMP (node, wet-point
normalization), SURFACE (node, archive normalization) and U (elem), truth archives,
climatology zarrs split across a "shared" and an "extra" source, and the run's model
config. The stage-1 scores are recomputed by hand over the mesh-wet points only and
must match, while the prediction at dry points is garbage (the new model's dry-point
output is unconstrained) and the target/source there are NaN (masked from the mesh).
"""

import json

import numpy as np
import pytest
import xarray as xr
import zarr

from paper_eval import climatology, compute, config, meshtools, scores

NOD2, NELEM = 6, 4
NZ1 = np.array([2.5, 7.5, 450.0])
# wet layer count per node / element (n_wet_layers = nlevels - 1): node 4 and 5 are dry
# below the surface layer; element 3 is dry below the surface layer
NLEV_NOD = np.array([4, 4, 3, 4, 2, 2])
NLEV_ELEM = np.array([4, 3, 4, 2])
NOD_AREA0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
NOD_AREA1 = np.array([1.5, 2.5, 3.5, 4.5, 0.0, 0.0])
ELEM_AREA = np.array([10.0, 20.0, 30.0, 40.0])
LAT_NOD = np.array([-60.0, -30.0, -10.0, 10.0, 30.0, 60.0])
LAT_ELEM = np.array([-40.0, -5.0, 5.0, 40.0])

STREAMS = {  # stream -> (entity, channels, npts, lat)
    "TEMP": ("node", ["temp_2", "temp_7"], NOD2, LAT_NOD),
    "SURFACE": ("node", ["ssh", "a_ice"], NOD2, LAT_NOD),
    "U": ("elem", ["u_2", "u_7"], NELEM, LAT_ELEM),
}
LEADS = [1, 2]
INIT = np.datetime64("2209-01-01")
GARBAGE = 1.0e4  # prediction at dry points

# truth archive statistics (the archive normalizes every channel) and the wet-point JSON
ARCHIVE = {
    "node": (["lon", "lat", "ssh", "a_ice", "sst", "temp_2", "temp_7"], [0.1, 0.3, 15.0, 14.0, 9.0], [0.5, 0.4, 8.0, 7.0, 4.0]),
    "elem": (["lon", "lat", "u_2", "u_7"], [0.05, 0.02], [0.2, 0.1]),
}
WET_NODE = {"_meta": {}, "temp_2": {"mean": 16.0, "std": 6.0}, "temp_7": {"mean": 12.0, "std": 3.0}}
WET_ELEM = {"_meta": {}, "u_2": {"mean": 0.01, "std": 0.25}, "u_7": {"mean": 0.0, "std": 0.15}}

# climatology sources: the shared ones lack the deeper level and sea ice, which the
# "extra" ones carry; channel order differs from the streams' on purpose
CLIM_SOURCES = {
    "FESOM_NODE": ["sst", "ssh", "temp_2"],
    "FESOM_NODE_extra": ["a_ice", "temp_7"],
    "FESOM_ELEM": ["u_2"],
    "FESOM_ELEM_extra": ["u_7"],
}


def _wet(entity, channels):
    nlev = NLEV_NOD if entity == "node" else NLEV_ELEM
    out = []
    for ch in channels:
        if ch in ("ssh", "a_ice", "sst"):
            out.append(np.ones(len(nlev), dtype=bool))
        else:
            level = {"2": 0, "7": 1}[ch.rsplit("_", 1)[1]]
            out.append(nlev - 1 > level)
    return np.stack(out, axis=1)  # (npts, nchan)


def _weights(entity, channels):
    if entity == "elem":
        return np.repeat(ELEM_AREA[:, None], len(channels), axis=1)
    return np.stack([NOD_AREA1 if ch == "temp_7" else NOD_AREA0 for ch in channels], axis=1)


def _clim_value(source_channel_idx, doy, npts, source):
    """Deterministic climatology field of one source channel (point-dependent)."""
    seed = sum(map(ord, source)) + 7 * source_channel_idx
    return (np.arange(npts) * 0.3 + 0.01 * doy + seed % 11).astype(np.float32)


def _write_mesh(path):
    face_nodes = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]).T  # 1-based (3, elem)
    nod_area = np.zeros((len(NZ1), NOD2))
    nod_area[0], nod_area[1] = NOD_AREA0, NOD_AREA1
    xr.Dataset(
        {
            "lon": ("nod2", np.linspace(-150, 150, NOD2)),
            "lat": ("nod2", LAT_NOD),
            "face_nodes": (("three", "elem"), face_nodes),
            "nod_area": (("nz", "nod2"), nod_area),
            "elem_area": ("elem", ELEM_AREA),
            "nlevels_nod2D": ("nod2", NLEV_NOD),
            "nlevels": ("elem", NLEV_ELEM),
            "nz1": ("nz1", NZ1),
        }
    ).to_netcdf(path)


def _write_truth(root):
    paths = {}
    for entity, (colnames, means, stds) in ARCHIVE.items():
        d = root / f"ocean_{entity}"
        d.mkdir(parents=True)
        g = zarr.open_group(str(d / f"ocean_{entity}_2180.zarr"), mode="w")
        arr = g.create_array("data", shape=(2, len(colnames)), dtype="f4")
        arr.attrs.update({"colnames": colnames, "means": means, "std": stds})
        paths[entity] = str(d)
    return paths


def _write_clims(clim_dir):
    clim_dir.mkdir(parents=True, exist_ok=True)
    for source, channels in CLIM_SOURCES.items():
        npts = NELEM if "ELEM" in source else NOD2
        clim = np.stack(
            [
                np.stack([_clim_value(i, doy, npts, source) for i in range(len(channels))], axis=1)
                for doy in range(1, 367)
            ]
        )
        g = zarr.open_group(str(clim_dir / f"clim_{source}_2180-2209.zarr"), mode="w")
        g.create_array("clim", data=clim, chunks=(1, npts, len(channels)))
        g.attrs.update({"channels": channels, "ref_start": 2180, "ref_end": 2209})


def _expected_clim(channel, doy, npts):
    for source, channels in CLIM_SOURCES.items():
        if channel in channels and (("ELEM" in source) == (npts == NELEM)):
            return _clim_value(channels.index(channel), doy, npts, source)
    raise AssertionError(channel)


def _norm(stream, channels):
    """(mean, std) the reader normalized `source` with: wet JSON for TEMP/U, archive
    statistics for SURFACE."""
    entity = STREAMS[stream][0]
    colnames, means, stds = ARCHIVE[entity]
    wet = {"TEMP": WET_NODE, "U": WET_ELEM}.get(stream, {})
    mean = [wet[c]["mean"] if c in wet else means[colnames.index(c) - 2] for c in channels]
    std = [wet[c]["std"] if c in wet else stds[colnames.index(c) - 2] for c in channels]
    return np.array(mean), np.array(std)


#: streams written as a random, per-lead subset of their points (`max_num_targets`)
#: in the `sparse` variant of the run fixture
SUBSAMPLED = ("TEMP", "U")


def _fields(seed, sparse=False):
    """Synthetic physical truth/prediction/initial state per stream and lead; NaN at
    dry target/source points, GARBAGE at dry prediction points. With `sparse`, the
    target/prediction of SUBSAMPLED streams keep only a shuffled subset of the points
    (all but two), re-drawn per lead: `per_lead[lead][2]` is its index, else None."""
    rng = np.random.default_rng(seed)
    out = {}
    for stream, (entity, channels, npts, _lat) in STREAMS.items():
        wet = _wet(entity, channels)
        init = rng.normal(size=(npts, len(channels))).astype(np.float32) + 5.0
        per_lead = {}
        for lead in LEADS:
            tar = (init + 0.3 * lead * rng.normal(size=init.shape)).astype(np.float32)
            pred = (tar + rng.normal(scale=0.5, size=init.shape)).astype(np.float32)
            idx = rng.permutation(npts)[: npts - 2] if sparse and stream in SUBSAMPLED else None
            per_lead[lead] = (np.where(wet, tar, np.nan), np.where(wet, pred, GARBAGE), idx)
        out[stream] = (np.where(wet, init, np.nan).astype(np.float32), per_lead)
    return out


def _write_results(path, fields):
    store = zarr.storage.ZipStore(str(path), mode="w")
    root = zarr.open_group(store=store, mode="w", zarr_format=3)
    interval = {"start": str(INIT.astype("datetime64[ns]")), "end": "2209-01-02T00:00:00.000000000"}
    for stream, (entity, channels, npts, lat) in STREAMS.items():
        init, per_lead = fields[stream]
        coords = np.stack([lat, np.linspace(-170, 170, npts)], axis=1).astype(np.float32)
        mean, std = _norm(stream, channels)
        groups = {"0/source": ((init - mean) / std, coords)}
        for lead in LEADS:
            tar, pred, idx = per_lead[lead]
            sel = slice(None) if idx is None else idx
            groups[f"{lead}/target"] = (tar[sel], coords[sel])
            groups[f"{lead}/prediction"] = (pred[sel][..., None], coords[sel])
        for key, (data, pts) in groups.items():
            g = root.create_group(
                f"0/{stream}/{key}",
                attributes={"channels": channels, "geoinfo_channels": [], "source_interval": interval},
            )
            g.create_array("data", data=np.asarray(data, dtype=np.float32))
            g.create_array("coords", data=pts)
    store.close()


@pytest.fixture(params=[False, True], ids=["full", "subsampled"])
def run(tmp_path, request):
    """Synthetic per-variable-layout run (full targets, or TEMP/U subsampled per lead as
    with `max_num_targets`); returns (cfg, fields)."""
    mesh_path = tmp_path / "mesh.nc"
    _write_mesh(mesh_path)
    truth = _write_truth(tmp_path / "truth")
    _write_clims(tmp_path / "cache" / "climatology")

    run_id = "fakefv01"
    results = tmp_path / "results" / run_id
    results.mkdir(parents=True)
    fields = _fields(seed=3, sparse=request.param)
    _write_results(results / "validation_chkpt00000_rank0000.zip", fields)

    (tmp_path / "wet_node.json").write_text(json.dumps(WET_NODE))
    (tmp_path / "wet_elem.json").write_text(json.dumps(WET_ELEM))
    model_dir = tmp_path / "models" / run_id
    model_dir.mkdir(parents=True)
    (model_dir / f"model_{run_id}_chkpt00000.json").write_text(
        json.dumps(
            {
                "streams": {
                    "TEMP": {"type": "fesom", "normalization_stats": "wet_node.json"},
                    "SURFACE": {"type": "fesom"},
                    "U": {"type": "fesom", "normalization_stats": "wet_elem.json"},
                }
            }
        )
    )

    clim = {"node": ["FESOM_NODE", "FESOM_NODE_extra"], "elem": ["FESOM_ELEM", "FESOM_ELEM_extra"]}
    cfg = {
        "run_id": run_id,
        "results_dir": str(results),
        "mini_epoch": 0,
        "mesh_path": str(mesh_path),
        "streams": list(STREAMS),
        "stream_layout": {s: {"entity": e, "clim": clim[e]} for s, (e, *_r) in STREAMS.items()},
        "truth": {s: truth[e] for s, (e, *_r) in STREAMS.items()},
        "climatology": {"ref_start": 2180, "ref_end": 2209},
        "init_filter": {"start": "2209-01-01", "end": None},
        "leads": {"max": max(LEADS), "map_leads": [2]},
        "regions": {"global": None, "nh": [0, 90], "sh": [-90, 0]},
        "showcase": {"init_time": "2209-01-01", "leads": [1, 2]},
        "compute": {"n_workers": 1},
        "output": {"cache_dir": str(tmp_path / "cache"), "fig_dir": str(tmp_path / "fig")},
        "models_dir": str(tmp_path / "models"),
        "repo_root": str(tmp_path),
    }
    return cfg, fields


# ── unit level ────────────────────────────────────────────────────────────────────────
def test_stream_entity_and_clim_paths(run):
    cfg, _ = run
    assert config.stream_entity(cfg, "TEMP") == "node"
    assert config.stream_entity(cfg, "U") == "elem"
    assert [p.name for p in config.clim_paths(cfg, "SURFACE")] == [
        "clim_FESOM_NODE_2180-2209.zarr", "clim_FESOM_NODE_extra_2180-2209.zarr",
    ]
    # legacy streams: no layout entry needed, one clim zarr named after the stream
    legacy = {k: v for k, v in cfg.items() if k != "stream_layout"}
    assert config.stream_entity(legacy, "FESOM_NODE") == "node"
    assert config.stream_entity(legacy, "FESOM_ELEM") == "elem"
    assert [p.name for p in config.clim_paths(legacy, "FESOM_ELEM")] == ["clim_FESOM_ELEM_2180-2209.zarr"]
    with pytest.raises(ValueError, match="mesh entity"):
        config.stream_entity(legacy, "TEMP")


def test_masks_and_weights_entity_matches_legacy_stream(tmp_path):
    _write_mesh(tmp_path / "mesh.nc")
    mesh = meshtools.load_mesh(tmp_path / "mesh.nc")
    for legacy, entity, channels in (
        ("FESOM_NODE", "node", ["sst", "temp_2", "temp_7", "temp_450"]),
        ("FESOM_ELEM", "elem", ["u_2", "u_7", "u_450"]),
    ):
        m_old, w_old = meshtools.masks_and_weights(mesh, legacy, channels)
        m_new, w_new = meshtools.masks_and_weights(mesh, entity, channels)
        np.testing.assert_array_equal(m_old, m_new)
        np.testing.assert_array_equal(w_old, w_new)
    # sea-ice channels are surface fields: wet everywhere, surface node area
    mask, w = meshtools.masks_and_weights(mesh, "node", ["a_ice", "m_ice", "temp_7"])
    assert mask[:2].all()
    np.testing.assert_array_equal(w[0], NOD_AREA0.astype(np.float32))
    np.testing.assert_array_equal(mask[2], NLEV_NOD - 1 > 1)
    with pytest.raises(ValueError):
        meshtools.masks_and_weights(mesh, "TEMP", ["temp_2"])


def test_channel_clim_lookup_selects_by_name(run):
    cfg, _ = run
    lookup = climatology.ChannelClimLookup(config.clim_paths(cfg, "TEMP"), ["temp_7", "temp_2"])
    got = lookup.get(np.datetime64("2209-03-01"))
    doy = int(climatology.doy_index(np.datetime64("2209-03-01")))
    np.testing.assert_array_equal(got[:, 0], _expected_clim("temp_7", doy, NOD2))
    np.testing.assert_array_equal(got[:, 1], _expected_clim("temp_2", doy, NOD2))
    assert not got.flags.writeable
    with pytest.raises(ValueError, match="m_ice"):
        climatology.ChannelClimLookup(config.clim_paths(cfg, "TEMP"), ["temp_2", "m_ice"])
    # the legacy lookup stays strict about the stored channel list
    with pytest.raises(ValueError, match="channel order mismatch"):
        climatology.ClimLookup(config.clim_paths(cfg, "TEMP")[0], ["temp_2"])


def test_norm_stats_per_stream(run):
    cfg, _ = run
    for stream, (_e, channels, *_r) in STREAMS.items():
        mean, std = scores._load_norm_stats(cfg, stream, channels)
        exp_mean, exp_std = _norm(stream, channels)
        np.testing.assert_allclose(mean, exp_mean, rtol=1e-6)
        np.testing.assert_allclose(std, exp_std, rtol=1e-6)


# ── end to end: compute.main on the synthetic run ─────────────────────────────────────
def _wrmse(err, w):
    return float(np.sqrt(np.sum(w * err**2) / np.sum(w)))


def test_compute_scores_wet_points_only(run, tmp_path):
    cfg, fields = run
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg))
    compute.main(["--config", str(cfg_path), "--n-workers", "1"])

    out = tmp_path / "cache" / cfg["run_id"]
    for stream, (entity, channels, npts, _lat) in STREAMS.items():
        ds = xr.open_dataset(out / f"scores_{stream}.nc").sel(region="global", sample=0)
        assert list(ds["channel"].values) == channels
        wet = _wet(entity, channels)
        w = _weights(entity, channels)
        init, per_lead = fields[stream]
        for lead in LEADS:
            tar, pred, idx = per_lead[lead]
            sampled = np.ones(npts, dtype=bool)
            if idx is not None:
                sampled[:] = False
                sampled[idx] = True
            doy = int(climatology.doy_index(INIT + np.timedelta64(lead, "D")))
            for c, ch in enumerate(channels):
                m = wet[:, c] & sampled  # scored: wet points the target holds
                cl = _expected_clim(ch, doy, npts)[m].astype(np.float64)
                t, p, s0 = (x[m, c].astype(np.float64) for x in (tar, pred, init))
                wc = w[m, c]
                sel = ds["score"].sel(lead=lead, channel=ch)
                np.testing.assert_allclose(float(sel.sel(model="emulator", metric="rmse")), _wrmse(p - t, wc), rtol=1e-5)
                np.testing.assert_allclose(float(sel.sel(model="persistence", metric="rmse")), _wrmse(s0 - t, wc), rtol=1e-4)
                np.testing.assert_allclose(float(sel.sel(model="climatology", metric="rmse")), _wrmse(cl - t, wc), rtol=1e-5)
                fa, oa = p - cl, t - cl
                acc = np.sum(wc * fa * oa) / np.sqrt(np.sum(wc * fa**2) * np.sum(wc * oa**2))
                np.testing.assert_allclose(float(sel.sel(model="emulator", metric="acc")), acc, rtol=1e-4)
        # dry-point garbage never reaches a score
        assert np.all(np.abs(ds["score"].sel(model="emulator", metric="rmse").values) < 100)
        maps = xr.open_dataset(out / f"maps_{stream}.nc")
        assert list(maps["channel"].values) == channels
        snap = xr.open_dataset(out / f"snapshots_{stream}.nc")
        # snapshot source is the denormalized initial state (wet points)
        np.testing.assert_allclose(snap["source"].values.T[wet], init[wet], rtol=1e-5)
        # snapshot target/prediction on the full point set, NaN where not sampled
        for k, lead in enumerate(LEADS):
            tar, pred, idx = per_lead[lead]
            want = tar.copy()
            if idx is not None:
                want[np.setdiff1d(np.arange(npts), idx)] = np.nan
            np.testing.assert_array_equal(snap["target"].values[k].T, want)
        np.testing.assert_allclose(snap["lat"].values, _lat, rtol=1e-6)
