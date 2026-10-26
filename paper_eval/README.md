# paper_eval

Standalone evaluation pipeline for the paper on the FESOM2 ocean-model AI emulator, run
`vdsqr1tp` (30-day daily rollouts, 364 inits, native CORE2 mesh). It runs as a two-stage
compute -> cache -> plot pipeline: stage 1 (`compute.py`) scores every sample against
persistence and climatology baselines and writes NetCDF caches; a separate spectra stage
(`compute_spectra.py`) computes native-mesh wavelength spectra on a small sample subset;
`plot.py` builds every paper figure from those caches only. The whole package is
deliberately uncommitted side work, entirely separate from the main `weathergen` package
-- never edit `packages/` or `src/` from here, and nothing in `packages/`/`src/` should
ever import from `paper_eval/`.

## Environment

Everything except the spectra stage runs in the repository's normal venv:

```
/work/ab0995/a270225/WeatherGenerator2/.venv/bin/python
```

The spectra stage additionally needs `jax` (CPU) and `implicit-filter` (the native-mesh
spectral filter package), installed additively into that same venv:

```
uv pip install jax jaxlib                                        # CPU-only; 0.10.0 in production
uv pip install --no-deps -e /work/ab0995/a270225/implicit_filter  # 1.3.1, editable, no-deps
```

`--no-deps` matters: `implicit-filter`'s own dependency pins are not relevant here, and
without it uv would try to resolve/upgrade packages the rest of the repo depends on.

To remove both again (reversible, additive-only change):

```
uv pip uninstall implicit-filter jax jaxlib
```

## Layout

```
paper_eval/
  config.py            # YAML config loading + cache_dir()/clim_dir()/fig_dir()/regrid_dir() helpers
  configs/default.yaml  # the run's config: run_id, results_dir, mesh_path, streams, init_filter, leads, spectra, ...
  merged_zarr.py        # verbatim copy of examples/merged_zarr_standalone.py: opens all per-rank result archives as one store
  meshtools.py          # CORE2 mesh loading, triangulation, wet masks/area weights, channel-name parsing
  metrics.py            # area-weighted RMSE/bias/ACC kernel (score_block)
  climatology.py        # day-of-year climatology builder (doy_index, build_climatology, ClimLookup)
  build_clim.py         # CLI: build one stream's climatology from its truth archive
  scores.py             # stage-1 scoring worker (init_worker/process_block/reduce_blocks): per-sample scores + spatial accumulators
  snapshots.py           # showcase snapshot extractor (single init, full spatial fields)
  compute.py            # CLI: stage-1 orchestrator, parallel pass over samples -> scores_*.nc, maps_*.nc, snapshots_*.nc, meta.json
  spectra.py            # implicit_filter wrapper: native-mesh scalar/velocity cumulative spectra
  compute_spectra.py    # CLI: spectra stage orchestrator -> spectra.nc
  regrid.py             # nearest-neighbour mesh->lat/lon regridding; optional visual-check utility, unused by the pipeline
  selfcheck.py          # CLI: independent re-derivation + verification of the stage-1 cache (8 checks)
  plot.py               # CLI: build every figure (f01..f11) from the caches
  plotting/             # per-figure builders (curves, depth, fesom_maps, eerie_maps, seasonal, spectra_figs,
                         # variability) + shared style.py
  cache/                # regenerable caches: climatology/, regrid/, fesom_filter_core2.npz, <run_id>/{scores,maps,snapshots}_*.nc, spectra.nc, meta.json
  figures/<run_id>/     # f01..f11, PDF+PNG
  sbatch/               # Slurm batch scripts (build_clim.sbatch, compute.sbatch, spectra.sbatch) + job .out logs
```

## Run book

All commands below are run from the repo root with
`/work/ab0995/a270225/WeatherGenerator2/.venv/bin/python`, against
`paper_eval/configs/default.yaml` unless noted. Every command line here was verified with
`--help` against the actual CLI (flags exist as documented).

### a. Build the climatology (once per reference period)

Needed before any scoring: `compute.py`'s ACC and the climatology baseline both read it.
`configs/default.yaml` already sets the production reference period
(`climatology.ref_start: 2179`, `climatology.ref_end: 2208`, disjoint from the 2209 eval
year), so no `--ref-start`/`--ref-end` override is needed for a production run.

```
.venv/bin/python -m paper_eval.build_clim --config paper_eval/configs/default.yaml --stream FESOM_NODE
.venv/bin/python -m paper_eval.build_clim --config paper_eval/configs/default.yaml --stream FESOM_ELEM
```

or, equivalently, as a 2-task array job:

```
sbatch paper_eval/sbatch/build_clim.sbatch
```

Measured on Levante (job 26713827, `--mem=96G`): FESOM_NODE 40:44 (17.7G RSS), FESOM_ELEM
46:27 (41.5G RSS) -- roughly 40-47 min/stream. Output:
`cache/climatology/clim_{stream}_{ref_start}-{ref_end}.zarr`. `build_clim` refuses to
overwrite an existing climatology file unless `--force` is passed.

### b. Stage-1 quick test (before committing to a full run)

```
.venv/bin/python -m paper_eval.compute --config paper_eval/configs/default.yaml \
    --samples 0:5 --leads 1,15,30 --n-workers 4
```

Measured on Levante (Task 9): ~77 s wall (~1.5 min) for 5 samples x 3 leads x both
streams. Writes `cache/<run_id>/{scores,maps,snapshots}_{stream}.nc` + `meta.json`
(overwritten silently -- these are regenerable caches).

### c. Selfcheck (run after any compute pass, quick test or full)

```
.venv/bin/python -m paper_eval.selfcheck --config paper_eval/configs/default.yaml
```

Independently re-derives 8 checks (a)-(h) (score recompute, fill/mask consistency,
persistence ordering, ACC decay, lead-day relation, climatology-model ACC, maps wet/dry,
meta consistency) from the raw inputs, using code paths that do not import
`meshtools.py`/`metrics.py`/`scores.py`/`climatology.py`, and compares against the
stage-1 cache. Prints one PASS/FAIL/SKIP line per check plus a summary line; exits
nonzero iff any check FAILs (a SKIP, e.g. a missing maps cache, does not fail the run).
Confirmed 8/8 PASS against the full production cache, with recompute deltas essentially
exact (<1e-9).

### d. Stage-1 full run

```
sbatch paper_eval/sbatch/compute.sbatch
```

32 cpus-per-task, 240G mem, scores every init-filtered sample (304 for `vdsqr1tp`) at
every lead 1..`leads.max` (30) for both streams. Measured on Levante (job 26718976):
1:03:13 wall.

### e. Spectra

```
sbatch paper_eval/sbatch/spectra.sbatch
```

Computes native-mesh (`implicit_filter`) cumulative power/KE spectra on a time-spread
subset of the init-filtered samples (`spectra.n_samples: 8`) at `spectra.n_scales: 20`
scales per job. Measured on Levante (job 26718977): 1:27:30 wall (~1.5 h). Output:
`cache/<run_id>/spectra.nc`.

### f. Figures

```
.venv/bin/python -m paper_eval.plot --config paper_eval/configs/default.yaml
```

Builds all of f01..f11 from the cache only (never the raw zarr archives); a figure whose
required cache file(s) are missing is skipped (`[skip]`, exit 0), while a figure builder
that raises is caught, reported, and turns the final exit code non-zero. Useful flags:

- `--only f01,f04` -- build only the listed figure ids.
- `--cache-dir PATH` -- read the stage-1/spectra cache from `PATH` instead of
  `<output.cache_dir>/<run_id>` (figures still always go to
  `<output.fig_dir>/<run_id>`).

Runs in seconds to a few minutes depending on `--only`; regenerating all 42 files
(f01..f09, PDF+PNG, some multi-channel) from the full production cache measured 4m50s on
Levante (that measurement predates f10; f10 needs the `rmse_persistence`/
`rmse_climatology`/`acc` maps variables added by Task 14, so it `[skip]`s cleanly
against any pre-Task-14 cache, including the current production cache, until Task 16's
re-run repopulates it). f11 needs the Task 20 `region` dim on `scores_*.nc`, so it
`[skip]`s cleanly against any pre-Task-20 cache the same way. Output:
`figures/<run_id>/f01_*.{pdf,png}` .. `f11_*.{pdf,png}`.

## Re-running with new inference results

To point the whole pipeline at a different inference run, change two keys in
`configs/default.yaml` (or a copy of it, passed to every CLI's `--config`):

```
run_id: <new_run_id>
results_dir: /path/to/results/<new_run_id>
```

Everything else -- `mesh_path`, the climatology (`cache/climatology/clim_*.zarr`), and
the spectra filter cache (`spectra.filter_cache`) -- is reusable as-is, as long as the
stream set (`streams`) and mesh are unchanged: none of it is keyed by `run_id`. The one
thing that MUST be revisited is `init_filter`: it excludes inits overlapping the
training period, and its `start`/`end` dates are specific to the run being evaluated --
adjust them to the new run's actual held-out period before scoring, or the sample
selection in `compute.py`/`compute_spectra.py` will be wrong (or, in the worst case,
empty -- both raise a clear `ValueError` if so).

Also revisit `showcase.init_time`: it MUST be moved into the new run's held-out period
too. `snapshots.extract` picks the *nearest available* init to `showcase.init_time`
with no error if the exact date is missing, so a stale date left over from the old run
silently produces a showcase snapshot (f04) from whatever sample happens to be
nearest -- possibly nowhere near the date the figure title implies. `snapshots.py` logs
a `[warn]` if the nearest available init is more than 15 days from the requested
`showcase.init_time`; treat that warning as a sign the config still needs updating.

All three `sbatch/` scripts (`build_clim.sbatch`, `compute.sbatch`, `spectra.sbatch`)
take an optional config path as `$1`, defaulting to `paper_eval/configs/default.yaml`
when omitted -- e.g.
`sbatch paper_eval/sbatch/compute.sbatch paper_eval/configs/<run_id>.yaml`
(array-job flags like `--array=1` still go before the script name, as usual with
`sbatch`). This is how a new run gets its own config without editing
`default.yaml` in place. Per-run `output.cache_dir` matters in particular whenever the
new run's channel set differs from the production one for some stream (e.g. run
`xjcn8rvp` adds `v_210` to `FESOM_ELEM`, making it 40 channels vs. `vdsqr1tp`'s 39):
`climatology.ClimLookup` validates the stored channel list strictly (exact list *and*
order) and raises rather than silently reusing a mismatched climatology, and the shared
`cache/climatology/clim_*.zarr` files must never be overwritten since they back the
production run's own cache/figures. The pattern is a per-run cache dir
(`output.cache_dir: paper_eval/cache_<run_id>`) containing a `climatology/` directory
populated per stream: symlink in the shared clim for any stream whose channel list is
unchanged (`ln -s ../../cache/climatology/clim_<stream>_<ref>.zarr
cache_<run_id>/climatology/clim_<stream>_<ref>.zarr`), and run `build_clim` against the
new run's own config for any stream whose channel list changed -- it reads the channel
list from the run's own results store (not the truth archive), so it produces a clim
with the new channel set automatically, written straight into the new cache dir.
`xjcn8rvp` does exactly this: `cache_xjcn8rvp/climatology/clim_FESOM_NODE_2179-2208.zarr`
is a symlink (its NODE channel list is identical to `vdsqr1tp`'s), while
`clim_FESOM_ELEM_2179-2208.zarr` is a fresh build for the 40-channel `v_210` list.

### Per-variable stream layout (fesom_vars: TEMP, SALT, SURFACE, U, V)

Models trained on `config/streams/fesom_vars/` (e.g. `fvars002`/`fvars003`; one-year
rollouts `yr1fv002`/`yr1fv003`, configs `configs/yr1fv00{2,3}.yaml`) write one stream per
variable instead of FESOM_NODE/FESOM_ELEM: TEMP/SALT (temp_2..temp_2035, 31 levels),
SURFACE (ssh, a_ice, m_ice) on nodes, U/V (31 levels) on elements. There are no sst/sss
channels; temp_2/salt_2 are what they were copies of, so configs name temp_2/salt_2
wherever old configs name sst/sss (`seasons.channels`, `figures.map_channels`,
`selfcheck.surface_channel`). The config key `stream_layout` maps each stream to its
mesh entity and climatology sources:

```
stream_layout:
  TEMP: {entity: node, clim: [FESOM_NODE, FESOM_NODE_extra]}
  U:    {entity: elem, clim: [FESOM_ELEM, FESOM_ELEM_extra]}
```

`entity` picks the wet masks/area weights (`config.stream_entity`,
`meshtools.masks_and_weights` takes "node"/"elem"); `clim` lists climatology zarrs
whose channels are picked by name (`climatology.ChannelClimLookup`, first source holding
a channel wins). The shared 2180-2209 FESOM_NODE/FESOM_ELEM climatologies are reused
as they are; they stop at 450 m and lack sea ice, so `FESOM_{NODE,ELEM}_extra` hold just
the rest (levels 535-2035 m, a_ice, m_ice), built once into the same climatology dir with
`sbatch paper_eval/sbatch/build_clim_fvars.sbatch paper_eval/configs/yr1fv002.yaml`
(`build_clim --channels`). `truth` needs an entry per stream (archive statistics for
the persistence denormalization; TEMP/SALT/U/V override them with the run's wet-point
`normalization_stats`, SURFACE keeps the archive's) and per `_extra` source. Configs
without `stream_layout` behave exactly as before (verified bitwise on yr1sal01).
`selfcheck.stream`/`selfcheck.surface_channel` (default FESOM_NODE/sst) point
`selfcheck.py` at TEMP/temp_2.

Subsampled targets: the fesom_vars stream configs set `max_num_targets: 32768`, so a run
inferred with them stores only a random subset of the points per forecast step
(re-drawn every step) in `target`/`prediction`; `source` still holds every point.
`meshtools.PointIndex` scatters such a target back onto the full point set (NaN where
not sampled) and only the sampled wet points are scored -- each daily score is then an
estimate from ~26% of the nodes (~13% of the elements). Maps carry NaN wherever
a sample did not sample a point, so with one sample they show only that lead's subset,
and snapshots likewise. Scores of a subsampled run are not exactly comparable with a
full-mesh run's: they include point-sampling noise.

## Data-contract notes

Sharp edges worth knowing before reading cache values directly:

- The fstep-0 `source` group is the model's z-score normalized input, not a physical
  field -- the pipeline denormalizes it using the truth-store's own per-channel
  mean/std (`scores._load_norm_stats` / `snapshots.py`'s reuse of it) before treating it
  as a physical initial condition; `target`/`prediction` are already physical.
- That denormalization -- and therefore the persistence baseline built from it -- is
  only correct if the evaluated model was trained with that same truth-store's
  per-channel mean/std; a run whose training used different normalization stats would
  have its `source` denormalized with the wrong mean/std and produce silently wrong
  persistence scores (no error, just incorrect numbers), so re-verify this whenever
  `results_dir`/`run_id` changes to a model trained against a different truth archive.
  Exception handled automatically: a run whose stream config sets `normalization_stats`
  (wet-point statistics JSON, path relative to the repo root) is read from
  `models/<run_id>/model_<run_id>_chkpt<mini_epoch:05d>.json` (`config.model_config_path`),
  and the JSON's mean/std replace the archive's for the channels it lists
  (`config.normalization_overrides`, applied in `scores._load_norm_stats` and
  `selfcheck._norm_stats`; `snapshots.py` and `compute_damped.py` reuse the former).
  Runs without the option get exactly the archive statistics. `pytest
  paper_eval/test_norm_stats.py` covers both cases.
- `FESOM_ELEM`'s `source` carries one extra channel (`v_210`) that `target`/`prediction`
  don't have; figures that iterate depth-suffixed channels skip it by design (it is not
  a bug if a `v_210` row/column is absent from a plot).
- Dry (masked) points are 0.0 or, occasionally, NaN in the raw data -- the mesh's
  `nlevels`/`nlevels_nod2D` is the ONLY authority for wet/dry; never infer wetness from
  a data value (`== 0.0` is not a safe test, since a real value can legitimately be
  zero).
- Node 125225 (lon=-153.99, lat=-76.99, on the Antarctic coast) is a known, harmless
  mesh/data boundary artifact: it carries a non-zero, non-NaN value at a mesh-dry depth
  in every sample/lead checked, but has zero weight in every score. `selfcheck.py`
  tolerates up to 10 such dry-point anomalies (check b) without failing; a NaN at a
  mesh-WET point remains a hard failure regardless.
- Sample ids are per-rank (each per-rank archive numbers its samples locally from 0, and
  `merged_zarr.MergedRankStore` remaps `<global_sample>` on top of that) -- never assume
  a sample id is stable or comparable across runs/configs; always key on `init_time`
  instead.
- valid date = init_time + lead (days); this is checked directly in `selfcheck.py`
  (check e) against each sample's own `target` timestamps.

## Figure index

The table below documents the default (`geometry: fesom`, e.g. `vdsqr1tp`/`xjcn8rvp`)
config's own cache file names (`scores_FESOM_NODE.nc` etc.). A `geometry: eerie`
config (e.g. `dbuzcpgf`, streams `OCEAN_2D`/`OCEAN_3D`) reads the same figure ids
from `scores_OCEAN_2D.nc`/`scores_OCEAN_3D.nc`/`maps_OCEAN_2D.nc`/... instead --
`plot.py` resolves cache file names from `cfg["streams"]`, never a hardcoded FESOM
list, so no figure id or `--only` flag changes. What *does* differ per figure under
`geometry: eerie` (`paper_eval/plotting/eerie_maps.py`, the grid-rendering
counterpart of `fesom_maps.py`; `paper_eval/plotting/curves.py`'s
`GEOMETRY_STREAMS` table for the score-based figures f01/f02/f03/f06/f07/f11):

- **f04**: maps are on a fixed 0.25 deg lat/lon grid, not the FESOM mesh --
  `target`/`prediction`/`error` render as a sparse point scatter (points are
  re-drawn every forecast step, ~9% grid coverage per sample) instead of a
  triangulated mesh field; `source` (fstep-0 initial condition) is its own extra
  row, scattered from its sparse wet-cell points onto the full grid, with its own
  norm and its own individual-panel output (`f04_{channel}_source.{pdf,png}`, no
  lead suffix -- it has no lead axis of its own).
- **f05/f10**: plain grid maps (`imshow`, row-major `(lat, lon)` reshape of the
  maps cache's already full-grid `ipoint` axis) instead of mesh triangulation;
  "dry" is `count == 0` (no sample's sparse points ever landed on that cell) in
  place of FESOM's fixed mesh mask.
- **f08**: EKE is computed at *every* depth where both `avg_uoe_X`/`avg_von_X`
  exist (including the `surf` level), not a single hardcoded shallow pair like
  FESOM's `u_2`/`v_2` -- one 3-panel (target/prediction/ratio) row per depth pair,
  stacked below the SST/SSH-equivalent (`avg_tos`/`avg_zos`) variance-ratio row.
- **f02/f06/f07**: depth-suffixed channels use the EERIE naming scheme
  (`avg_thetao_2.5m`, `avg_uoe_surf`, ...) instead of FESOM's `temp_107`-style
  suffixes; `figures.curve_depths` (FESOM's hand-picked highlight subset) does not
  apply -- EERIE has only 3-4 levels total per variable, so f02 highlights all of
  them. f07's area-weighted mean profile row uses `cos(lat)` + `count > 0` in
  place of the FESOM mesh's own per-point area weights (no mesh to read one from).
- **f09**: `[skip]`s unconditionally with reason "spectra disabled for
  sparse-sampled gridded runs" -- native-mesh spectra do not apply to gridded,
  sparsely-sampled EERIE streams (`compute_spectra.py` is never even run against
  an EERIE config).
- **f01/f03/f11**: same layout, channel-group table (`GEOMETRY_STREAMS`) picks
  the EERIE surface stream/channel names (`avg_tos`/`avg_sos`/`avg_zos`) in place
  of `sst`/`sss`/`ssh`.

| Figure | Shows | Cache file(s) |
| --- | --- | --- |
| f01 | Surface (sst, sss, ssh) RMSE/ACC vs. lead, emulator vs. persistence | `scores_FESOM_NODE.nc` |
| f02 | Depth-resolved (`figures.curve_depths`) RMSE/ACC vs. lead for temp/salt/u/v, emulator (solid) vs. persistence (dashed) | `scores_FESOM_NODE.nc`, `scores_FESOM_ELEM.nc` |
| f03 | Scorecard heatmap: emulator skill vs. persistence/climatology. Wide/short layout: all channels along the x axis (grouped Surface \| Temperature \| Salinity \| u \| v, vertical separator lines between groups, rotated channel labels), the four (lead, baseline) combinations as the y rows. Also emits one individual scorecard per (lead, baseline) row: `f03_scorecard_{lead}d_{pers\|clim}.{pdf,png}` (single horizontal strip, same x-labels, own colorbar) | `scores_FESOM_NODE.nc`, `scores_FESOM_ELEM.nc` |
| f04 | Showcase-init triptych maps (target / prediction / error = prediction - target, for the single showcase init) per `figures.map_channels`, at `leads.map_leads`. Also emits one individual panel per channel x kind x lead: `f04_{channel}_{target\|prediction\|error}_{lead}d.{pdf,png}` (own colorbar/title; target/prediction share the triptych's norm, error keeps its symmetric norm) | `snapshots_FESOM_NODE.nc`, `snapshots_FESOM_ELEM.nc` |
| f05 | Spatial mean-bias/RMSE maps (suptitle/panel labels state the N-forecast aggregation explicitly) per `figures.map_channels`, at `leads.map_leads` | `maps_FESOM_NODE.nc`, `maps_FESOM_ELEM.nc` |
| f06 | Lead x depth RMSE/ACC heatmaps | `scores_FESOM_NODE.nc`, `scores_FESOM_ELEM.nc` |
| f07 | Depth profiles, 3 rows: area-weighted mean (ground truth vs. emulator at leads 15/30, wet points only, `masks_and_weights`-weighted), then RMSE/bias at fixed leads | `scores_FESOM_NODE.nc`, `scores_FESOM_ELEM.nc`, `maps_FESOM_NODE.nc`, `maps_FESOM_ELEM.nc` |
| f08 | Variability diagnostics: one row of log2 forecast/reference variance ratios (SST, SSH, and surface EKE from u_2/v_2) per lead in `leads.map_leads`, all on one shared symmetric scale. `f08_variability.{pdf,png}` stacks every lead as a row; `f08_variability_{lead}d.{pdf,png}` is the single-row figure per lead (written only when more than one lead is available) | `maps_FESOM_NODE.nc`, `maps_FESOM_ELEM.nc` |
| f09 | Wavelength power/KE spectra, target vs. prediction, log-log spectral density vs. wavelength. Hard-cut at the mesh-resolution cutoff (200 km, `_MESH_SCALE_KM`): nothing below it is plotted and the x-axis itself ends there (no greyed/faded sub-cutoff region) | `spectra.nc` |
| f10 | Spatial skill maps (1 - RMSE/RMSE_baseline vs. persistence and vs. climatology, diverging, clipped to [-1, 1]) and ACC maps (clipped at 0) per `figures.map_channels`, at `leads.map_leads`; `[skip]`s per channel if the maps cache lacks the Task-14 `rmse_persistence`/`rmse_climatology`/`acc` variables (pre-Task-14 caches). Also emits one individual panel per channel x kind x lead: `f10_{channel}_{skill_pers\|skill_clim\|acc}_{lead}d.{pdf,png}` (same norms/labels as the combined rows) | `maps_FESOM_NODE.nc`, `maps_FESOM_ELEM.nc` |
| f11 | Seasonal skill (`seasons:` config section; NH meteorological convention of the sample's init month, local season reversed in the SH, December inits excluded). `f11_{ch}_seasonal.{pdf,png}` per `seasons.channels`: 2 rows (RMSE, ACC) x 2 cols (Northern/Southern Hemisphere) vs. lead, emulator only, 4 season lines (fixed DJF/MAM/JJA/SON palette) + a thin grey all-season (Dec excluded) reference, legend with N per season. `f11_summary_skill_{lead}d.{pdf,png}` (one per `leads.map_leads`): heatmap, x = all channels (grouped/ordered as the f03 scorecard), y = 8 rows season x hemisphere (DJF-NH, DJF-SH, ..., SON-SH), values = 1 - RMSE_emulator/RMSE_persistence. `[skip]`s cleanly if the scores cache lacks the Task 20 `region` dim (pre-Task-20 caches) | `scores_FESOM_NODE.nc`, `scores_FESOM_ELEM.nc` |

| f12 | Pointwise skill against climatology (1 - RMSE/RMSE_clim, clipped to [-1, 1]) for SST, SSH and vector velocity, in f08's layout and lead handling: `f12_skill_vs_climatology.{pdf,png}` stacks every `leads.map_leads` row, `f12_skill_vs_climatology_{lead}d.{pdf,png}` per lead. Velocity uses RMSE_|U|^2 = RMSE_u^2 + RMSE_v^2 for both forecast and climatology. `[skip]`s if the maps cache predates the Task-14 `rmse_climatology` variable | `maps_FESOM_NODE.nc`, `maps_FESOM_ELEM.nc` |

Individual panels (f03/f04/f08/f10/f12) are written in addition to, not instead of, the combined
figure, in the same builder call/pass over the data -- `plot.py --only` selects at family
granularity as before (e.g. `--only f04` builds both the combined triptych and every
individual f04 panel; there is no separate id to select only the individual outputs).
