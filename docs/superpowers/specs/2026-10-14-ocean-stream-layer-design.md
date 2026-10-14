# Ocean data and stream layer for a from-scratch retrain

Date: 2026-10-14. Branch: `kacpnowak/develop/forcings` (experimental, never merged).

## Goal

Prepare the data and stream layer for retraining the forced FESOM emulator from scratch, with no
backward compatibility to existing checkpoints. This spec covers the layer and its verification
only; the retraining schedule (pretraining stages, spin-up, daily-change prediction, ensembles) is a
separate spec.

Motivation (from the one-year evaluations, 2026-10-08 to 2026-10-14):
- the archive normalisation statistics include dry points stored as 0 (salt_450: mean 23.87,
  std 15.0 psu; wet points 34.75, 0.72), which hid deep-salinity errors from the loss;
- the lineage was trained without masking dry targets (the overwrite used wrong stream names);
- one stream holds temperature, salinity and surface fields, sharing one input embedding across
  channels whose scales differ by orders of magnitude;
- `sst`/`sss` are bit-identical to `temp_2`/`salt_2` (checked in all 210 archive years, 12 days
  per year, ~3.2e8 values each), so the top layer was predicted and weighted twice;
- the prediction heads returned bf16 under the trainer's autocast (fixed in e6be5fc9).

## Streams

Five ocean streams, each with the same channels as inputs and targets, plus the IFS forcing:

| Stream | Archive | Channels | Count |
|---|---|---|---|
| TEMP | ocean_node | temp_2 ... temp_2035 | 31 |
| SALT | ocean_node | salt_2 ... salt_2035 | 31 |
| U | ocean_elem | u_2 ... u_2035 | 31 |
| V | ocean_elem | v_2 ... v_2035 | 31 |
| SURFACE | ocean_node | ssh, a_ice, m_ice | 3 |
| IFS_ATMO | atmos_all | as `config/streams/fesom_forced_atm/ifs_forcing.yml` | 9 |

- Depth: the 31 model levels with mid-depths 2.5, 7.5, ..., 2035 m (`nz1[0:31]` of the mesh). The
  archive channel names are the rounded mid-depths (`temp_2`, `temp_7`, ..., `temp_2035`); the
  config lists them explicitly.
- Not used: `sst`, `sss` (duplicates of `temp_2`, `salt_2`), `snow`, `fw`, `evap`, `prec`, `swr`,
  `fh`, `MLD2`, `w_*`, `tx_sur`, `ty_sur`, and levels below 2035 m.
- Normalisation: TEMP, SALT, U, V use the committed wet-point statistics
  (`config/streams/fesom/wet_stats_{node,elem}.json`); SURFACE uses the archive statistics.
- Loss weight: `loss_weight: 1.0` for every ocean stream, so each variable family carries the same
  share of the loss; the IFS forcing has `loss_weight: 0`.
- Each stream has its own stream embedder (per-variable input embedding by construction).
- No shared read cache: the node archive is read by three streams and the element archive by two;
  the smoke run measures whether data loading limits training.
- Files: `config/streams/fesom_vars/` with `temp.yml`, `salt.yml`, `u.yml`, `v.yml`,
  `surface.yml`, and `ifs_forcing.yml` as a symlink to `../fesom_forced_atm/ifs_forcing.yml`.
  Stream ids (unused so far; today FESOM_NODE is 2, FESOM_ELEM 1, IFS_ATMO 2139): TEMP 2150,
  SALT 2151, U 2152, V 2153, SURFACE 2154.

## Dry points and bathymetry (FESOM reader)

All three use one wet mask built from the mesh
(`/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc`):

- Wet mask: point p is wet at level k when `k < n_wet_layers(p)`, with
  `n_wet_layers = nlevels_nod2D - 1` for nodes and `nlevels - 1` for elements (the convention of
  `paper_eval/meshtools.py`). A channel's level index comes from its name (`<var>_<depth>` matched
  to the nearest `nz1`), as `paper_eval.meshtools.parse_channel` does. Channels without a depth
  suffix (ssh, a_ice, m_ice) are wet everywhere. The archive row order equals the mesh order
  (verified for both archives when the wet-point statistics were computed).
- Targets: dry points of depth channels become NaN, so the loss skips them (`lp_loss` already
  normalises each channel by its number of valid points). This replaces `target_fill_value`
  (a value test) for streams that use the mesh mask.
- Inputs: dry points of depth channels are set to the channel mean before normalisation, so they
  are 0 after normalisation.
- Bathymetry geoinfo: one static value per point, `log(1 + bottom depth in m)`, with bottom depth
  `-zbar_n_bottom` (nodes) or `-zbar_e_bottom` (elements) from the mesh; normalised by its mean
  and std over the points of that mesh entity. Returned as the reader's single geoinfo channel for
  every ocean stream, for sources and target coordinates. The sampler and model already size
  tokens and embeddings from `get_geoinfo_size()`.
- Config (per ocean stream): `mesh_path: <mesh file>`, `wet_mask_from_mesh: true`,
  `bathymetry_geoinfo: true`. Without these keys the reader behaves exactly as today (no geoinfo,
  value-based `target_fill_value` / `source_fill_value` still available), so existing configs and
  runs are unaffected.
- The reader reads the mesh once per reader instance (small: ~127k nodes, ~245k elements).

## Training config for the smoke run

`config/config_fesom_vars.yml`: the architecture and data settings of the current lineage (from
`models/frcwin01/model_frcwin01.json`: HEALPix level 5, `ae_local_dim_embed` and
`ae_global_dim_embed` 2048, 4 global blocks, 16 forecast-engine blocks, decoder
`PerceiverIOCoordConditioning`, 1-day windows, training 2000-01-01 to 2208-12-31, validation 2209,
forecasting mode with `offset: 1`, `num_steps: 15`), with `streams_directory:
config/streams/fesom_vars`, nothing frozen, no spin-up (`prefix_steps` unset). It is trained with
the `train` entry point (from scratch). The smoke run overrides it to one short mini-epoch.

## Decoder precision

Done in e6be5fc9: the prediction heads run with autocast disabled on fp32 input, so physical
predictions are fp32; flash-attention blocks stay bf16.

## Verification

Unit tests (CPU):
- mesh wet mask: a known shallow node is dry at a deep level and wet at the surface; channels
  without depth are wet everywhere; level parsing for `temp_2`, `salt_450`, `u_2035`;
- targets: dry points of depth channels are NaN after `normalize_target_channels`, wet points are
  unchanged; SURFACE channels never NaN;
- inputs: dry points are exactly 0 after `normalize_source_channels`;
- geoinfo: shape (npoints, 1), finite, normalised (mean ~0, std ~1 over the mesh), ordering equal
  to the archive rows;
- without the new keys: output bit-identical to today (regression);
- stream configs: the five streams have the channel lists above (31/31/31/31/3, no sst/sss), the
  mesh keys, wet statistics on the depth streams, `loss_weight` 1, and the forcing link.

Real-data check (skipped where the archive is absent): build all five readers for one day; each
stream's channels as listed; per-level dry fraction equal to the mesh's (e.g. 27 % dry at 450 m
for nodes); the archive value is 0 or NaN at mesh-dry points (tolerance as in
`compute_wet_stats.py`); geoinfo finite.

Smoke run (2 nodes x 4 A100-80GB, `train` from scratch, a few dozen steps): the model builds with
five ocean streams and the forcing; losses of all five streams are finite; peak GPU memory; seconds
per step and the time the GPUs wait for data, compared with today's ~21-30 s per step. If data
loading limits training, a shared per-worker day cache (keyed by archive and day) is added in a
follow-up.

Done when the unit tests and the real-data check pass and the smoke run trains stably at an
acceptable step time.
