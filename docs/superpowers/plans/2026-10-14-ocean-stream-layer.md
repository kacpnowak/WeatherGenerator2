# Ocean data and stream layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Five per-variable FESOM streams (TEMP, SALT, U, V, SURFACE) with a mesh-based dry-point mask, bathymetry as geoinfo and wet-point statistics, plus a from-scratch training config verified by a smoke run.

**Architecture:** A small mesh module reads wet-layer counts, bottom depth and layer depths from the FESOM mesh file. The FESOM reader uses it, behind new stream keys, to set dry targets to NaN and dry inputs to the channel mean while the rows are still in mesh order (`get_source`/`get_target`, before the sampler shuffles them), and to return log bottom depth as its single geoinfo channel. Stream configs and a training config wire this up; a smoke run measures memory and step time.

**Tech Stack:** Python 3.12, NumPy, xarray, zarr, dask, OmegaConf, pytest, PyTorch/FSDP2, SLURM on Levante.

**Spec:** `docs/superpowers/specs/2026-10-14-ocean-stream-layer-design.md`

## Global Constraints

- Branch `kacpnowak/develop/forcings`; commit with a pathspec (`git commit -m "..." -- <paths>`), never push; trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never `uv run`; Python as `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python`; ruff as `RAYON_NUM_THREADS=1 .venv/bin/ruff`.
- Mesh file: `/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc`. Archives: `/work/ab0995/a270088/Kacper/weathergenertor/AWICM3/{ocean_node,ocean_elem,atmos_all}` (another user's data: read only).
- Wet rule: point p is wet at level k when `k < n_wet_layers(p)`, `n_wet_layers = nlevels_nod2D - 1` (nodes) / `nlevels - 1` (elements). Bottom depth `-zbar_n_bottom` / `-zbar_e_bottom`. Geoinfo value `log1p(bottom depth)`, normalised by its mean/std over the mesh entity.
- New stream keys: `mesh_path`, `wet_mask_from_mesh: true`, `bathymetry_geoinfo: true`. Without them the reader is bit-identical to today.
- Streams (`config/streams/fesom_vars/`): TEMP 2150 (temp_2..temp_2035, 31), SALT 2151 (salt_2..salt_2035, 31), U 2152 (u_2..u_2035, 31), V 2153 (v_2..v_2035, 31), SURFACE 2154 (ssh, a_ice, m_ice); IFS_ATMO via symlink to `../fesom_forced_atm/ifs_forcing.yml`. `loss_weight: 1.0` each; depth streams use `config/streams/fesom/wet_stats_{node,elem}.json`; no `sst`/`sss` anywhere.
- `src/` and `packages/` never import `paper_eval`.
- Untracked helpers (SLURM scripts) are not committed.

## Review Focus

- **Rows not in mesh order** (shuffled/subsetted before masking): the mask must be applied in `get_source`/`get_target`, never in `normalize_*_channels`. Test in Task 2 compares masked rows against the mesh.
- **Geoinfo size asked before lazy init:** the sampler calls `get_geoinfo_size()` while sizing tokens, before any data read. Expected: 1 with `bathymetry_geoinfo`, 0 without. Test in Task 2.
- **Multi-day windows:** a source/target window can hold several days; mask and geoinfo must be tiled per day. Test in Task 2 with a 2-day window.
- **Channel names that look like depths but are not** (`a_ice`, `m_ice`, `ssh`, `MLD2`): never masked. Test in Task 1.
- **Archive NaNs at dry points** (some shallow salinity levels): sources must end up finite at dry points (mean fill), targets NaN. Test in Task 2.

---

### Task 1: Mesh module

**Files:**
- Create: `packages/readers_extra/src/weathergen/readers_extra/fesom_mesh.py`
- Test: `tests/test_fesom_mesh.py`

**Interfaces:**
- Produces: `MeshEntity(n_wet_layers: np.ndarray[int64], bottom_depth: np.ndarray[float64], nz1: np.ndarray[float64])`; `read_mesh_entity(mesh_path: str | Path, n_points: int) -> MeshEntity`; `channel_level(name: str, nz1: np.ndarray) -> int | None`; `dry_mask(channels: list[str], mesh: MeshEntity, n_steps: int) -> np.ndarray[bool]` of shape `(n_steps * npts, len(channels))`; `bathymetry(mesh: MeshEntity) -> np.ndarray[float64]` (`log1p` bottom depth, shape `(npts,)`).

- [ ] **Step 1: Write the failing tests** in `tests/test_fesom_mesh.py`:

```python
"""FESOM mesh helpers: wet layers, channel levels, dry mask, bathymetry."""

from pathlib import Path

import numpy as np
import pytest

from weathergen.readers_extra.fesom_mesh import (
    MeshEntity,
    bathymetry,
    channel_level,
    dry_mask,
    read_mesh_entity,
)

MESH = Path("/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc")
NZ1 = np.array([2.5, 7.5, 15.0, 25.0, 450.0, 2035.0])


@pytest.mark.parametrize(
    "name, level",
    [("temp_2", 0), ("salt_7", 1), ("u_15", 2), ("v_450", 4), ("temp_2035", 5),
     ("ssh", None), ("a_ice", None), ("m_ice", None), ("MLD2", None), ("sst", None)],
)
def test_channel_level(name, level):
    assert channel_level(name, NZ1) == level


def test_dry_mask_follows_wet_layers_and_tiles_per_step():
    mesh = MeshEntity(np.array([1, 5, 0]), np.array([5.0, 3000.0, 1.0]), NZ1)
    mask = dry_mask(["temp_2", "temp_450", "ssh"], mesh, n_steps=2)
    expected_day = np.array([[False, True, False], [False, False, False], [True, True, False]])
    np.testing.assert_array_equal(mask, np.concatenate([expected_day, expected_day]))


def test_bathymetry_is_log1p_bottom_depth():
    mesh = MeshEntity(np.array([1, 5]), np.array([0.0, 1000.0]), NZ1)
    np.testing.assert_allclose(bathymetry(mesh), np.log1p([0.0, 1000.0]))


@pytest.mark.skipif(not MESH.is_file(), reason="FESOM mesh not available")
@pytest.mark.parametrize("n_points, wet_450", [(126858, 0.73), (244659, 0.73)])
def test_real_mesh_entities(n_points, wet_450):
    mesh = read_mesh_entity(MESH, n_points)
    assert mesh.n_wet_layers.shape == mesh.bottom_depth.shape == (n_points,)
    assert (mesh.bottom_depth > 0).all()
    k = channel_level("temp_450", mesh.nz1)
    assert abs(np.mean(mesh.n_wet_layers > k) - wet_450) < 0.01


@pytest.mark.skipif(not MESH.is_file(), reason="FESOM mesh not available")
def test_unknown_point_count_is_rejected():
    with pytest.raises(ValueError, match="neither"):
        read_mesh_entity(MESH, 12345)
```

- [ ] **Step 2: Run** `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_fesom_mesh.py` — expected FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** `fesom_mesh.py`:

```python
"""Static FESOM mesh information for the FESOM reader: wet layers, bottom depth, layer depths."""

import dataclasses
from pathlib import Path

import numpy as np
import xarray as xr


@dataclasses.dataclass(frozen=True)
class MeshEntity:
    """Nodes or elements of a FESOM mesh."""

    n_wet_layers: np.ndarray  # (npts,) number of wet layers at each point
    bottom_depth: np.ndarray  # (npts,) depth of the sea floor, m (positive)
    nz1: np.ndarray  # (nlev,) layer mid-depths, m


def read_mesh_entity(mesh_path: str | Path, n_points: int) -> MeshEntity:
    """The mesh entity (nodes or elements) with `n_points` points."""
    with xr.open_dataset(mesh_path) as ds:
        nz1 = ds["nz1"].values.astype(np.float64)
        if n_points == ds.sizes["nod2"]:
            n_wet = ds["nlevels_nod2D"].values - 1
            bottom = -ds["zbar_n_bottom"].values
        elif n_points == ds.sizes["elem"]:
            n_wet = ds["nlevels"].values - 1
            bottom = -ds["zbar_e_bottom"].values
        else:
            raise ValueError(
                f"{n_points} points are neither the mesh's nodes ({ds.sizes['nod2']}) nor its "
                f"elements ({ds.sizes['elem']}): {mesh_path}"
            )
    return MeshEntity(n_wet.astype(np.int64), bottom.astype(np.float64), nz1)


def channel_level(name: str, nz1: np.ndarray) -> int | None:
    """Level index of a `<var>_<depth>` channel (nearest layer mid-depth), None for 2-D ones."""
    head, _, tail = name.rpartition("_")
    if not head or not tail.isdigit():
        return None
    return int(np.argmin(np.abs(nz1 - float(tail))))


def dry_mask(channels: list[str], mesh: MeshEntity, n_steps: int) -> np.ndarray:
    """(n_steps * npts, nchannels): True where a channel's level lies below the sea floor."""
    day = np.zeros((mesh.n_wet_layers.shape[0], len(channels)), dtype=bool)
    for j, name in enumerate(channels):
        level = channel_level(name, mesh.nz1)
        if level is not None:
            day[:, j] = mesh.n_wet_layers <= level
    return np.tile(day, (n_steps, 1))


def bathymetry(mesh: MeshEntity) -> np.ndarray:
    """log(1 + bottom depth in m) per point."""
    return np.log1p(mesh.bottom_depth)
```

- [ ] **Step 4: Run** the tests — expected PASS (all, including the two real-mesh tests on Levante).

- [ ] **Step 5: Lint and commit**

```bash
RAYON_NUM_THREADS=1 .venv/bin/ruff check packages/readers_extra/src/weathergen/readers_extra/fesom_mesh.py tests/test_fesom_mesh.py
RAYON_NUM_THREADS=1 .venv/bin/ruff format packages/readers_extra/src/weathergen/readers_extra/fesom_mesh.py tests/test_fesom_mesh.py
git add packages/readers_extra/src/weathergen/readers_extra/fesom_mesh.py tests/test_fesom_mesh.py
git commit -m "Add FESOM mesh helpers: wet layers, channel levels, dry mask, bathymetry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- packages/readers_extra/src/weathergen/readers_extra/fesom_mesh.py tests/test_fesom_mesh.py
```

---

### Task 2: FESOM reader: mesh mask and bathymetry geoinfo

**Files:**
- Modify: `packages/readers_extra/src/weathergen/readers_extra/data_reader_fesom.py` (`__init__` near the `geoinfo_channels`/`geoinfo_idx` initialisation ~line 69; `_lazy_init` ~line 243-478, which resets `geoinfo_channels`/`geoinfo_idx` to `[]` near line 468; `get_source` ~649; `get_target` ~688)
- Test: `tests/test_fesom_reader_mesh.py`

**Interfaces:**
- Consumes: Task 1's `read_mesh_entity`, `dry_mask`, `bathymetry`.
- Produces: stream keys `mesh_path`, `wet_mask_from_mesh`, `bathymetry_geoinfo`; with `bathymetry_geoinfo`, `get_geoinfo_size() == 1` at any time (also before `_lazy_init`), `get_source/get_target(...).geoinfos` of shape `(rows, 1)`, and `normalize_geoinfos` normalising with the bathymetry mean/std over the mesh entity.

Behaviour to implement (read the reader first; keep every existing behaviour when the keys are absent):
1. In `__init__`: if `stream_info.get("bathymetry_geoinfo")`, set `self.geoinfo_channels = ["bathymetry"]` and `self.geoinfo_idx = [0]` instead of the empty lists, so `get_geoinfo_size()` is right before lazy init. If either `wet_mask_from_mesh` or `bathymetry_geoinfo` is set and `mesh_path` is missing, raise `ValueError` naming the stream and `mesh_path`.
2. In `_lazy_init`: do not reset the geoinfo lists when `bathymetry_geoinfo` is set. After the mesh sizes are known, if either key is set, `self._mesh = read_mesh_entity(stream_info["mesh_path"], self.source_mesh_size)` (source and target come from the same archive; assert `self.target_mesh_size == self.source_mesh_size`). With `wet_mask_from_mesh`, precompute the one-day masks `self._dry_src_day = dry_mask(<selected source channel names>, self._mesh, 1)` and `self._dry_trg_day = dry_mask(<selected target channel names>, self._mesh, 1)` — use the same selected channel-name lists the reader uses for `source_idx`/`target_idx` (find them in `_lazy_init`/`select`). With `bathymetry_geoinfo`: `raw = bathymetry(self._mesh)`, `self._geo_day = raw[:, None].astype(np.float32)`, `self.mean_geoinfo = np.array([raw.mean()])`, `self.stdev_geoinfo = np.array([raw.std()])`.
3. In `get_source`, after `data` is computed and `n_steps` known: with `wet_mask_from_mesh`, `mask = np.tile(self._dry_src_day, (n_steps, 1))` and `data = np.where(mask, self.src_mean_vec.astype(data.dtype), data)` (dry inputs become the channel mean, hence 0 after normalisation; archive NaNs at dry points disappear). With `bathymetry_geoinfo`, `geoinfos = np.tile(self._geo_day, (n_steps, 1))` instead of the empty array.
4. In `get_target`: the same with `self._dry_trg_day` and `np.nan` as fill (dry targets are skipped by the loss), and the same geoinfos.
5. `normalize_source_channels` / `normalize_target_channels` stay as they are (their value-based fills still work for configs that use them).

- [ ] **Step 1: Write the failing tests** in `tests/test_fesom_reader_mesh.py` (real data, skipped where absent):

```python
"""FESOM reader with the mesh-based dry mask and bathymetry geoinfo (real data)."""

from pathlib import Path

import numpy as np
import pytest

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.readers_extra.fesom_mesh import bathymetry, dry_mask, read_mesh_entity
from weathergen.readers_extra.registry import get_extra_reader

ROOT = Path("/work/ab0995/a270088/Kacper/weathergenertor/AWICM3")
MESH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
pytestmark = pytest.mark.skipif(not (ROOT / "ocean_node").is_dir(), reason="AWI-CM3 data absent")
CHANNELS = ["temp_2", "temp_450", "salt_95", "salt_450"]


def _reader(days=1, **keys):
    info = {"name": "T", "type": "fesom", "stream_id": 2150, "filenames": ["ocean_node"],
            "source": CHANNELS, "target": CHANNELS, "loss_weight": 1.0, **keys}
    tw = TimeWindowHandler(np.datetime64("2100-01-01T00:00"), np.datetime64("2100-01-20T00:00"),
                           np.timedelta64(days, "D"), np.timedelta64(days, "D"))
    return get_extra_reader("fesom")(tw_handler=tw, filename=ROOT / "ocean_node",
                                     stream_info=info, stage="train")


MESH_KEYS = {"mesh_path": MESH, "wet_mask_from_mesh": True, "bathymetry_geoinfo": True}


def test_geoinfo_size_is_known_before_any_read():
    assert _reader(**MESH_KEYS).get_geoinfo_size() == 1
    assert _reader().get_geoinfo_size() == 0


def test_mesh_keys_need_a_mesh_path():
    with pytest.raises(ValueError, match="mesh_path"):
        _reader(wet_mask_from_mesh=True)


@pytest.mark.parametrize("days", [1, 2])
def test_dry_targets_are_nan_and_wet_targets_untouched(days):
    plain, masked = _reader(days), _reader(days, **MESH_KEYS)
    raw, got = plain.get_target(1), masked.get_target(1)
    n_steps = raw.data.shape[0] // 126858
    dry = dry_mask(CHANNELS, read_mesh_entity(MESH, 126858), n_steps)
    assert np.isnan(got.data[dry]).all()
    np.testing.assert_array_equal(got.data[~dry], raw.data[~dry])
    np.testing.assert_array_equal(got.coords, raw.coords)


@pytest.mark.parametrize("days", [1, 2])
def test_dry_inputs_become_zero_after_normalisation(days):
    reader = _reader(days, **MESH_KEYS)
    src = reader.get_source(1)
    n_steps = src.data.shape[0] // 126858
    dry = dry_mask(CHANNELS, read_mesh_entity(MESH, 126858), n_steps)
    norm = reader.normalize_source_channels(src.data.copy())
    assert np.isfinite(norm).all()
    np.testing.assert_allclose(norm[dry], 0.0, atol=1e-6)


def test_bathymetry_geoinfo_matches_the_mesh_and_normalises():
    reader = _reader(**MESH_KEYS)
    src = reader.get_source(1)
    assert src.geoinfos.shape == (src.data.shape[0], 1)
    raw = bathymetry(read_mesh_entity(MESH, 126858))
    np.testing.assert_allclose(src.geoinfos[:126858, 0], raw.astype(np.float32))
    g = reader.normalize_geoinfos(src.geoinfos.astype(np.float64).copy())
    assert abs(g.mean()) < 1e-3 and abs(g.std() - 1.0) < 1e-3


def test_without_mesh_keys_the_reader_is_unchanged():
    a, b = _reader(), _reader()
    np.testing.assert_array_equal(a.get_source(1).data, b.get_source(1).data)
    assert a.get_source(1).geoinfos.shape[1] == 0
```

- [ ] **Step 2: Run** `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_fesom_reader_mesh.py` — expected FAIL (geoinfo size 0 / no NaNs / no ValueError).

- [ ] **Step 3: Implement** behaviour 1-5 above in `data_reader_fesom.py`.

- [ ] **Step 4: Run** the new tests and the existing reader tests: `tests/test_fesom_reader_mesh.py tests/test_fesom_wet_stats.py tests/test_forcing_fesom_data.py tests/test_forcing_sampler.py` (plus any `test_data_reader_fesom*.py` under `packages/readers_extra/tests/`) — expected PASS.

- [ ] **Step 5: Lint (changed lines only; the file has pre-existing findings) and commit**

```bash
git add tests/test_fesom_reader_mesh.py
git commit -m "FESOM reader: mesh-based dry mask and bathymetry geoinfo

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- packages/readers_extra/src/weathergen/readers_extra/data_reader_fesom.py tests/test_fesom_reader_mesh.py
```

---

### Task 3: Per-variable stream configs

**Files:**
- Create: `config/streams/fesom_vars/{temp,salt,u,v,surface}.yml`, symlink `config/streams/fesom_vars/ifs_forcing.yml -> ../fesom_forced_atm/ifs_forcing.yml`
- Test: `tests/test_fesom_vars_configs.py`

**Interfaces:**
- Consumes: Task 2's stream keys.
- Produces: stream set `config/streams/fesom_vars` with streams `TEMP`, `SALT`, `U`, `V`, `SURFACE`, `IFS_ATMO`.

Each ocean stream file follows `config/streams/fesom/fesom.yml` (licence header; keys `type: fesom`, `stream_id`, `filenames`, `loss_weight: 1.`, `source`, `target` (same list), `location_weight: cosine_latitude`, `masking_rate: 0.6`, `masking_rate_none: 0.05`, `token_size: 32`, `tokenize_spacetime: True`, `max_num_targets: 32768`, and the same `embed`, `embed_target_coords`, `target_readout`, `pred_head` blocks), plus `mesh_path`, `wet_mask_from_mesh: true`, `bathymetry_geoinfo: true`, and for TEMP/SALT `normalization_stats: config/streams/fesom/wet_stats_node.json`, for U/V `.../wet_stats_elem.json`. No `target_fill_value`/`source_fill_value`. Channel lists: the 31 names `<var>_<d>` for d in 2, 7, 15, 25, 35, 45, 55, 65, 75, 85, 95, 107, 125, 147, 175, 210, 255, 310, 375, 450, 535, 630, 735, 850, 975, 1110, 1255, 1415, 1600, 1810, 2035; SURFACE: ssh, a_ice, m_ice; TEMP/SALT/SURFACE read `ocean_node`, U/V `ocean_elem`. Add a header comment per file saying what the stream holds and why (per-variable stream, mesh mask, bathymetry, wet statistics).

- [ ] **Step 1: Write the failing tests** in `tests/test_fesom_vars_configs.py`:

```python
"""The per-variable FESOM stream set."""

from pathlib import Path

import pytest

from weathergen.common.config import load_streams

STREAMS = Path(__file__).resolve().parents[1] / "config" / "streams" / "fesom_vars"
DEPTHS = [2, 7, 15, 25, 35, 45, 55, 65, 75, 85, 95, 107, 125, 147, 175, 210, 255, 310, 375,
          450, 535, 630, 735, 850, 975, 1110, 1255, 1415, 1600, 1810, 2035]
EXPECTED = {
    "TEMP": (2150, "ocean_node", [f"temp_{d}" for d in DEPTHS], "wet_stats_node.json"),
    "SALT": (2151, "ocean_node", [f"salt_{d}" for d in DEPTHS], "wet_stats_node.json"),
    "U": (2152, "ocean_elem", [f"u_{d}" for d in DEPTHS], "wet_stats_elem.json"),
    "V": (2153, "ocean_elem", [f"v_{d}" for d in DEPTHS], "wet_stats_elem.json"),
    "SURFACE": (2154, "ocean_node", ["ssh", "a_ice", "m_ice"], None),
}
ARCHIVE = Path("/work/ab0995/a270088/Kacper/weathergenertor/AWICM3")


def test_stream_names():
    assert set(load_streams(STREAMS)) == set(EXPECTED) | {"IFS_ATMO"}


@pytest.mark.parametrize("name", EXPECTED)
def test_ocean_stream(name):
    s = load_streams(STREAMS)[name]
    stream_id, archive, channels, stats = EXPECTED[name]
    assert s["stream_id"] == stream_id and list(s["filenames"]) == [archive]
    assert list(s["source"]) == channels and list(s["target"]) == channels
    assert s["loss_weight"] == 1.0
    assert s["wet_mask_from_mesh"] is True and s["bathymetry_geoinfo"] is True
    assert Path(s["mesh_path"]).name == "fesom.mesh.diag.nc"
    if stats:
        assert s["normalization_stats"] == f"config/streams/fesom/{stats}"
    else:
        assert "normalization_stats" not in s
    assert "target_fill_value" not in s and "source_fill_value" not in s


def test_no_duplicate_surface_channels():
    for s in load_streams(STREAMS).values():
        assert not {"sst", "sss"} & set(s.get("source") or [])


def test_forcing_is_the_atmospheric_state_set():
    link = STREAMS / "ifs_forcing.yml"
    assert link.is_symlink() and link.resolve() == (STREAMS.parent / "fesom_forced_atm" / "ifs_forcing.yml").resolve()


@pytest.mark.skipif(not ARCHIVE.is_dir(), reason="AWI-CM3 data absent")
@pytest.mark.parametrize("name", EXPECTED)
def test_channels_exist_in_the_archive(name):
    import zarr

    _, archive, channels, _ = EXPECTED[name]
    cols = zarr.open_group(ARCHIVE / archive / f"{archive}_2100.zarr", mode="r")["data"].attrs["colnames"]
    assert set(channels) <= set(cols)
```

- [ ] **Step 2: Run** the test — expected FAIL (directory missing).
- [ ] **Step 3: Create** the five stream files and the symlink.
- [ ] **Step 4: Run** the test — expected PASS; also `tests/test_forcing_configs.py` still passes.
- [ ] **Step 5: Commit** (`git add` the new files; commit with a pathspec of `config/streams/fesom_vars tests/test_fesom_vars_configs.py`).

---

### Task 4: From-scratch training config

**Files:**
- Create: `config/config_fesom_vars.yml`
- Test: `tests/test_fesom_vars_training_config.py`

**Interfaces:**
- Consumes: Task 3's stream set.
- Produces: a config for `.venv/bin/train --config config/config_fesom_vars.yml`.

Content: the architecture and training settings of `models/frcwin01/model_frcwin01.json` (read it): all top-level model keys (`embed_*`, `ae_local_*`, `ae_adapter_*`, `ae_global_*`, `ae_aggregation_*`, `decoder_type`, `pred_*`, `num_class_tokens`, `num_register_tokens`, `fe_*`, `forecast_att_dense_rate`, `healpix_level`, `rope_2D`, `with_mixed_precision`, `with_flash_attention`, `compile_model`, `with_fsdp`, `attention_dtype`, `mixed_precision_dtype`, `mlp_norm_eps`, `norm_eps`, `norm_type`, `ae_local_max_tokens_per_cell`, `zarr_store`), `training_config`, `validation_config`, `test_config`, `data_loading`, `train_logging`, with these changes: `streams_directory: ./config/streams/fesom_vars/`, `freeze_modules: ""`, no `training_config.forecast.prefix_steps`, no run state (`general`, `from_run_id`, `world_size`, `rank`, `local_rank`, `with_ddp`, `world_size_original`, `latent_noise_*` defaults are fine to omit), no data paths (they come from the private config). Header comment: purpose, origin (frcwin01's settings), and that the retraining schedule is a separate spec.

- [ ] **Step 1: Write the failing test**:

```python
"""The from-scratch training config for the per-variable FESOM streams."""

from weathergen.common import config


def test_fesom_vars_training_config_merges():
    cf = config.load_merge_configs(None, None, None, None, "config/config_fesom_vars.yml", {})
    assert set(cf.streams) == {"TEMP", "SALT", "U", "V", "SURFACE", "IFS_ATMO"}
    assert cf.freeze_modules == ""
    assert cf.healpix_level == 5 and cf.fe_num_blocks == 16 and cf.ae_global_dim_embed == 2048
    fc = cf.training_config.forecast
    assert fc.num_steps == 15 and fc.offset == 1 and fc.get("prefix_steps") is None
    assert str(cf.training_config.start_date).startswith("2000-01-01")
    assert str(cf.validation_config.start_date).startswith("2209-01-01")
```

(Adapt the `load_merge_configs` call to its real signature — see `run_train` in `src/weathergen/run_train.py`, which passes `args.private_config, None, None, args.base_config, *args.config, cli_overwrite`.)

- [ ] **Step 2: Run** — expected FAIL (file missing).
- [ ] **Step 3: Write** `config/config_fesom_vars.yml`.
- [ ] **Step 4: Run** — expected PASS.
- [ ] **Step 5: Commit** both files with a pathspec.

---

### Task 5: Smoke run from scratch (controller-run)

- [ ] **Step 1:** Untracked `sbatch_fesom_vars_smoke.sh`, modelled on `sbatch_fesom_forced_smoke.sh` (2 nodes x 4 A100-80GB, 1 h): `srun --label .venv/bin/train --config config/config_fesom_vars.yml --run-id smkvar01 --options training_config.num_mini_epochs=1 training_config.samples_per_mini_epoch=64 validation_config.samples_per_mini_epoch=16 validation_config.output.num_samples=0 train_logging.terminal=1`, with `nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 5` running in the background into `output/smoke_vars_gpu_<job>.csv`.
- [ ] **Step 2:** Submit, wait in the background. Check: no Traceback/OOM/Watchdog; model builds (log shows the five ocean streams and IFS_ATMO); finite per-stream losses (`LossPhysical.<STREAM>.mse.avg`); peak `memory.used`; seconds per step (`s/sec`); mean GPU utilisation during training (low utilisation with long gaps = data loading limits training).
- [ ] **Step 3:** Record the numbers in the ledger; delete the smoke run's `models/`, `results/`, `logs/` folders afterwards (smoke artifact).
