# v3 Benchmark Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bank the ft0808 result with a full OceanBench eval (phase 0), then train and evaluate a fresh "v3" model with benchmark-focused target channels that outranks GLONET on ≥12/20 scoreboard rows (phases 1-2).

**Architecture:** Asymmetric channels (133-channel source, 33-channel target), drift-safe training schedule (cosine LR decay, 2023-H2 held out for selection, recency + 10-step curriculum phases), aggregation-engine finetune, all evaluated through the existing v2 OceanBench eval stack.

**Tech Stack:** WeatherGenerator (this repo, branch `oceanbench-eval`), slurm on Jupiter (aarch64), oceanbench 0.4.0 venv, kerchunk parquets (already built), zarr v2 challenger stores.

**Spec:** `docs/superpowers/specs/2026-08-19-v3-beat-glonet-design.md`

## Global Constraints

- ALL sbatch jobs use `--account=e-ext-2025e01-128` (never hclimrep). Existing scripts with `#SBATCH --account=hclimrep` are overridden at submission: `sbatch -A e-ext-2025e01-128 <script>` — or the header is edited when the task touches the script anyway.
- No output-side corrections anywhere (no persistence blending, no bias correction).
- No loss channel-weighting: v3 stream configs must NOT contain `target_channel_weights` (the v2 file `config/streams/glorys_v2/glorys.yml` still carries one — do not copy it).
- Checkpoint selection and go/no-go decisions never use training or validation loss; only forecast-RMSE smokes (defined in Task 3/10).
- Compute nodes have no internet egress. Heavy CPU work (exports) runs via CPU-node sbatch, never on login nodes.
- Before re-running inference under an existing run-id: `rm -rf /e/scratch/weatherai/shared_work/{models,results}/<run_id>` (stale-run-json trap).
- Python for analysis: repo venv `/e/scratch/hclimrep/nowak2/WeatherGenerator2/.venv/bin/python`. OceanBench: `/e/scratch/hclimrep/nowak2/oceanbench/.venv/bin/python` (zarr 2.18.4 — challenger stores must stay zarr v2, `write_empty_chunks=True`).
- Working directory for all commands: `/e/scratch/hclimrep/nowak2/WeatherGenerator2` unless stated.
- Key model/run ids: ft0808 = `glory_finetuning_20260808_191038` (best v2 checkpoint, only `_latest.chkpt`); v2 base = `glorys_20260730_223127`.
- Progress ledger: append one line per completed task to `.superpowers/sdd/wondrous-coalescing-taco/progress.md`.

---

### Task 1: Parameterize the OceanBench driver for the v2/v3 stack

**Files:**
- Modify: `run_oceanbench_inference.sh` (repo root)

**Interfaces:**
- Produces: driver flags `--streams-config <path>` (path passed as `--config`, replaces the hardcoded eval config AND removes the `--options streams_directory=...` term entirely) and `--output-stream <name>` (default `GLORYS`, v2/v3 use `GLORYS2`/`GLORYS3`). Existing flags (`--model`, `--run-prefix`, `--dates`, `--dry-run`, `--eval-config`) keep working unchanged.

Background you need: the driver currently builds (line ~233) a CMD containing BOTH `--config ${eval_config}` and `--options streams_directory=./config/streams/glorys_eval/ ... "test_config.output.streams=[GLORYS]"`. Two bugs for v2: (a) config-source merge semantics UNION the stream sets when the config file and `--options` name different streams directories (documented in `config/config_forecasting_glorys_obench_v2.yml` header) — so `streams_directory` must never be passed via `--options`; (b) the output stream name is hardcoded `[GLORYS]`.

- [ ] **Step 1: Edit the driver**

In the argument-parsing block (follows the same case-statement pattern as `--eval-config`), add:

```bash
        --output-stream)
            if [[ $# -lt 2 ]]; then
                echo "--output-stream requires an argument" >&2
                exit 1
            fi
            OUTPUT_STREAM="$2"
            shift 2
            ;;
```

with `OUTPUT_STREAM="GLORYS"` initialised next to the other defaults near the top.

In `build_command()`: delete the `streams_directory=./config/streams/glorys_eval/` term from the `--options` list (v1 runs are unaffected: `config_forecasting_glorys_obench.yml` line 28 carries the same value itself — the header comment there says so), quote the stream name with the existing `printf -v ... '%q'` pattern, and replace the literal `"test_config.output.streams=[GLORYS]"` with `"test_config.output.streams=[${output_stream_q}]"`.

Change line 2 `#SBATCH --account=hclimrep` → `#SBATCH --account=e-ext-2025e01-128`.

Update the runbook header comment: document both new flags and the account.

- [ ] **Step 2: Verify with dry runs**

```bash
bash -n run_oceanbench_inference.sh
./run_oceanbench_inference.sh --dry-run --dates "2024-01-03" | grep -c "streams_directory"   # expect 0
./run_oceanbench_inference.sh --dry-run --dates "2024-01-03" | grep -c "output.streams=\[GLORYS\]"  # expect 1 (v1 default intact)
./run_oceanbench_inference.sh --dry-run --dates "2024-01-03" \
  --model glory_finetuning_20260808_191038 \
  --eval-config config/config_forecasting_glorys_obench_v2.yml \
  --output-stream GLORYS2 | grep -c "output.streams=\[GLORYS2\]"  # expect 1
```

- [ ] **Step 3: Commit**

```bash
git add run_oceanbench_inference.sh
git commit -m "Parameterize OceanBench driver for v2/v3 stacks (--output-stream, no streams_directory override)"
```

### Task 2: ft0808 full 52-date inference (phase 0)

**Files:** none modified — operational task using Task 1's driver.

- [ ] **Step 1: Submit** (skip-existing makes resubmission safe; the smoke dates under prefix `obench_ft0808_` are a DIFFERENT prefix and will not be reused — this run uses its own):

```bash
./run_oceanbench_inference.sh \
  --model glory_finetuning_20260808_191038 \
  --run-prefix obench_ft0808full_ \
  --eval-config config/config_forecasting_glorys_obench_v2.yml \
  --output-stream GLORYS2
```

- [ ] **Step 2: Monitor to completion** (background poll of `squeue`; job name `obench-inference`; self-resubmits on 8 h walls). On finish assert 52 zips:

```bash
ls -d /e/scratch/weatherai/shared_work/results/obench_ft0808full_2024*/validation_chkpt00000_rank0000.zip | wc -l   # expect 52
```

- [ ] **Step 3: Integrity sweep** — run this snippet (52 zips: fsteps 0-10 present, Tuesday `source_interval`, prediction shape (≈695383, 129, 1)):

```python
import zarr, glob, re, numpy as np
zips = sorted(glob.glob("/e/scratch/weatherai/shared_work/results/obench_ft0808full_2024*/validation_chkpt00000_rank0000.zip"))
assert len(zips) == 52, len(zips)
bad = []
for z in zips:
    date = re.search(r"_(2024\d{4})/", z).group(1)
    store = zarr.storage.ZipStore(z, mode="r"); root = zarr.open_group(store, mode="r")
    g = root["0"]["GLORYS2"]
    if sorted(g.keys(), key=int) != [str(i) for i in range(11)]: bad.append((date, "fsteps")); continue
    si = g["0"]["source"].attrs["source_interval"]
    if np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}") - np.datetime64(si["start"], "D") != np.timedelta64(1, "D"):
        bad.append((date, si["start"]))
    p = g["10"]["prediction"]; d = p["data"] if "data" in p else p
    if d.shape[1] != 129 or d.shape[0] < 600000: bad.append((date, d.shape))
    store.close()
assert not bad, bad
print("52/52 pass")
```

- [ ] **Step 4: Ledger line** (`Phase 0 inference complete, 52/52 zips verified`).

### Task 3: Extend the exporter for v2/v3 zips and export ft0808

**Files:**
- Modify: `export_glorys_nc.py` (repo root)
- Create: `/e/scratch/hclimrep/nowak2/eval_output/sbatch_export.sh` (CPU-node wrapper)

**Interfaces:**
- Produces: exporter flags `--stream-group` (zarr group name inside zips, default `GLORYS`) and `--levels` (comma-separated depth values, default the v1 10-level set). Challenger store layout unchanged: dims `(first_day_datetime, lead_day_index, depth, latitude=672, longitude=1440)`, zarr v2, `write_empty_chunks=True`, CF `standard_name` attrs, 1970 epoch.

The 14 export levels (spec): `0.494025,9.573,15.8101,25.2114,34.4342,47.3737,65.8073,92.3261,130.666,155.851,222.475,318.127,453.938,541.089`. ft0808 has thetao/so/uo/vo at ALL of these (32-level model). For the future v3 model, uo/vo exist only at 0.494025 and 15.8101: the exporter must NaN-fill a variable's missing levels rather than fail (only surface/15 m velocity rows are scored; Class-4/Lagrangian read the 15.8101 level, which exists).

- [ ] **Step 1: Modify the exporter.** Read the current channel-selection logic (it builds per-variable depth stacks by parsing `var_<depth>m` channel names from the zip's `channels` attr, hardcoding group `"GLORYS"`). Changes: (a) group name from `--stream-group`; (b) depth axis = parsed `--levels` floats, per variable select channel `f"{var}_{level}m"` — string-match against the actual channel names (they are exact strings like `thetao_0.494025m`), and where a channel is absent write NaN for that (var, level) plane; (c) `zos` handling unchanged; (d) assert at least the 0.494025 m level exists for every variable.

- [ ] **Step 2: Self-test against one existing smoke zip** (fast, on login node — reads one date only):

```bash
.venv/bin/python export_glorys_nc.py --self-test \
  --run-prefix obench_ft0808_ --stream-group GLORYS2 \
  --levels 0.494025,9.573,15.8101,25.2114,34.4342,47.3737,65.8073,92.3261,130.666,155.851,222.475,318.127,453.938,541.089 \
  --dates 2024-01-03 \
  --output /tmp/claude-23518/-e-scratch-hclimrep-nowak2-WeatherGenerator2/2ccf298b-2b79-47ef-a00c-5f8dab3dfed6/scratchpad/selftest_v2.zarr
```

Expected: exporter's own integrity checks pass; open the store and assert `depth` has 14 entries and `thetao` lead-0 surface mean is ≈ 14 °C.

- [ ] **Step 3: Create the CPU-node sbatch wrapper** `/e/scratch/hclimrep/nowak2/eval_output/sbatch_export.sh`:

```bash
#!/bin/bash
#SBATCH --account=e-ext-2025e01-128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=obench-export
#SBATCH --output=/e/scratch/hclimrep/nowak2/eval_output/logs/%j.out
#SBATCH --error=/e/scratch/hclimrep/nowak2/eval_output/logs/%j.err
# Usage: sbatch sbatch_export.sh <run_prefix> <stream_group> <levels_csv> <output_zarr>
set -u
cd /e/scratch/hclimrep/nowak2/WeatherGenerator2
.venv/bin/python -u export_glorys_nc.py \
  --run-prefix "$1" --stream-group "$2" --levels "$3" --output "$4"
```

- [ ] **Step 4: Export ft0808 (full 52 dates) on a CPU node:**

```bash
sbatch /e/scratch/hclimrep/nowak2/eval_output/sbatch_export.sh \
  obench_ft0808full_ GLORYS2 \
  0.494025,9.573,15.8101,25.2114,34.4342,47.3737,65.8073,92.3261,130.666,155.851,222.475,318.127,453.938,541.089 \
  /e/scratch/hclimrep/nowak2/eval_output/wg_ft0808_challenger.zarr
```

Monitor; on completion run the cross-venv read gate:

```python
# /e/scratch/hclimrep/nowak2/oceanbench/.venv/bin/python
import warnings; warnings.filterwarnings("ignore")
import xarray as xr, numpy as np
ds = xr.open_zarr("/e/scratch/hclimrep/nowak2/eval_output/wg_ft0808_challenger.zarr", consolidated=False)
assert dict(ds.sizes) == {"first_day_datetime": 52, "lead_day_index": 10, "depth": 14, "latitude": 672, "longitude": 1440}, dict(ds.sizes)
from oceanbench.core.resolution import get_dataset_resolution
assert get_dataset_resolution(ds) == "quarter_degree"
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
assert sorted(rename_dataset_with_standard_names(ds).data_vars) == [
 "eastward_sea_water_velocity","northward_sea_water_velocity",
 "sea_surface_height_above_geoid","sea_water_potential_temperature","sea_water_salinity"]
print("READ GATE PASSED")
```

- [ ] **Step 5: Commit the exporter change + ledger line.**

```bash
git add export_glorys_nc.py
git commit -m "Exporter: --stream-group/--levels for v2/v3 zips, NaN-fill absent levels"
```

### Task 4: ft0808 OceanBench evaluation + scoreboard (phase 0 complete)

**Files:**
- Create: `/e/scratch/hclimrep/nowak2/eval_output/challenger_ft0808.py`

- [ ] **Step 1: Challenger file** (same pattern as `challenger_glory_ft_20260731.py`):

```python
"""OceanBench challenger: WeatherGenerator ft0808 (glory_finetuning_20260808_191038)."""
import os
import xarray
CHALLENGER_ZARR = os.environ.get(
    "WG_CHALLENGER_STORE",
    "/e/scratch/hclimrep/nowak2/eval_output/wg_ft0808_challenger.zarr",
)
challenger_dataset: xarray.Dataset = xarray.open_zarr(CHALLENGER_ZARR, consolidated=False)
assert challenger_dataset.sizes["first_day_datetime"] == 52
assert challenger_dataset.sizes["lead_day_index"] == 10
assert challenger_dataset.sizes["latitude"] == 672
assert challenger_dataset.sizes["longitude"] == 1440
```

- [ ] **Step 2: Submit the offline eval** (existing script; account override at submission):

```bash
cd /e/scratch/hclimrep/nowak2/eval_output && sbatch -A e-ext-2025e01-128 -t 03:00:00 \
  run_oceanbench_eval.sh /e/scratch/hclimrep/nowak2/eval_output/challenger_ft0808.py \
  /e/scratch/hclimrep/nowak2/oceanbench_stage
```

Monitor; success = job COMPLETED exit 0 and `challenger_ft0808.global.report.ipynb` exists.

- [ ] **Step 3: Rename report single-dot** → `challenger_ft0808_global_report.ipynb` (the notebook-reader convention).

- [ ] **Step 4: Scoreboard rebuild + republish:**

```bash
cd /e/scratch/hclimrep/nowak2/eval_output && \
/e/scratch/hclimrep/nowak2/oceanbench/.venv/bin/python build_scoreboard_data.py --no-download
```

Re-embed `scoreboard_data.json` into `scoreboard.html` between the FIRST `/*DATA-START*/.../*DATA-END*/` pair only (the second occurrence is runtime JS — use the position-based replace, anchor: the match must come after `id="data"`), then republish the artifact at the existing URL `https://claude.ai/code/artifact/b1df873c-b017-4311-aabd-c708d23658df` (WebFetch it first if the publish is rejected for staleness).

- [ ] **Step 5: Produce the phase-0 baseline table**: for each of the 20 metric rows × leads {1,5,10}, ft0808 vs GLONET from `scoreboard_data.json` (version 0.4.0), marking win/loss. Save as `/e/scratch/hclimrep/nowak2/eval_output/ft0808_vs_glonet_rows.md`, append the row-win count to the ledger. This is the baseline phase 1 must beat.

### Task 5: v3 stream configs

**Files:**
- Create: `config/streams/glorys_v3/glorys.yml`, `config/streams/glorys_v3/atmo.yml`
- Create: `config/streams/glorys_v3_eval/glorys.yml`, `config/streams/glorys_v3_eval/atmo.yml`

**Interfaces:**
- Produces: stream name `GLORYS3` (MUST differ from GLORYS2 — stream names key checkpoint module names `embeds.<name>.*`); 133 source channels (order pinned identically to `config/streams/glorys_v2/glorys.yml` `channels_order`); 33 target channels in this exact order:
  `zos`, then `thetao_<L>m` for L in [0.494025, 9.573, 15.8101, 25.2114, 34.4342, 47.3737, 65.8073, 92.3261, 130.666, 155.851, 222.475, 318.127, 453.938, 541.089], then `so_<L>m` same 14, then `uo_0.494025m`, `uo_15.8101m`, `vo_0.494025m`, `vo_15.8101m`.

- [ ] **Step 1: `config/streams/glorys_v3/glorys.yml`** — copy `config/streams/glorys_v2/glorys.yml` and change: top key `GLORYS2:` → `GLORYS3:`; DELETE the whole `target_channel_weights:` block (global constraint); replace the `target:` list with the 33 channels above (source and channels_order stay the full 133). Keep `filenames` (training parquet `glorys_025_mesh_fixed.parq`), stream_id, token/embed settings unchanged.
- [ ] **Step 2: `config/streams/glorys_v3/atmo.yml`** — copy `config/streams/glorys_v2/atmo.yml` verbatim (radiation set `[2t,2d,10u,10v,ssrd,strd]`, 1979-2024 zarr). No changes needed.
- [ ] **Step 3: eval mirrors** — copy `config/streams/glorys_v2_eval/glorys.yml` → `glorys_v3_eval/glorys.yml` with: key `GLORYS3:`, the same 33-entry `target:` list (eval target MUST equal the trained pred head — the 129-vs-133 lesson), source/channels_order full 133, `filenames` v3 nowcast parquet + `target_file` training parquet (unchanged). Copy `glorys_v2_eval/atmo.yml` → `glorys_v3_eval/atmo.yml` verbatim (stats_override block included).
- [ ] **Step 4: Verify with reader instantiation:**

```python
import yaml, numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.readers_extra.data_reader_mesh import DataReaderMesh
for f, exp_targets in [("config/streams/glorys_v3/glorys.yml", 33),
                       ("config/streams/glorys_v3_eval/glorys.yml", 33)]:
    cfg = OmegaConf.create({**yaml.safe_load(open(f))["GLORYS3"], "name": "GLORYS3"})
    twh = TimeWindowHandler(np.datetime64("2020-01-01T00"), np.datetime64("2020-02-01T00"),
                            np.timedelta64(24,"h"), np.timedelta64(24,"h"))
    r = DataReaderMesh(twh, Path(cfg["filenames"][0]), cfg)
    assert len(r.source_channels) == 133 and len(r.target_channels) == exp_targets, (f, len(r.source_channels), len(r.target_channels))
    assert r.target_channels[0] == "zos" and r.target_channels[1] == "thetao_0.494025m"
    assert r.target_channels[-1] == "vo_15.8101m"
    assert r.target_channel_weights is None or r.target_channel_weights == []
print("v3 stream configs verified")
```

- [ ] **Step 5: Commit** (`git add config/streams/glorys_v3 config/streams/glorys_v3_eval && git commit -m "Add v3 stream configs: 133-source/33-target GLORYS3"`).

### Task 6: v3 training configs (three curriculum phases)

**Files:**
- Create: `config/config_glorys_v3_phaseA.yml` (epochs 0-48: full record, 4-step)
- Create: `config/config_glorys_v3_phaseB.yml` (epochs 48-56: 2010+, 4-step)
- Create: `config/config_glorys_v3_phaseC.yml` (epochs 56-64: 2010+, 10-step)

Phase A (fresh `train` run). Content — copy the license header comment style from `config/config_forecasting_glorys_finetuning.yml`, then:

```yaml
streams_directory: "./config/streams/glorys_v3/"
freeze_modules: ""

train_logging:
  terminal: 10
  metrics: 20
  checkpoint: 64

data_loading :
  num_workers: 8

general:
  istep: 0
  run_id: ???

training_config:
  training_mode: ["masking"]
  num_mini_epochs: 48
  samples_per_mini_epoch: 4096
  # 2023-H2 held out for checkpoint selection (spec); 2024 (benchmark year) never trained.
  start_date: 1993-01-01T00:00
  end_date: 2023-06-30T00:00
  learning_rate_scheduling :
    lr_start: 1e-6
    lr_max: 5e-5
    lr_final_decay: 1e-6
    lr_final: 0.0
    num_steps_warmup: 256
    num_steps_cooldown: 512
    policy_warmup: "cosine"
    policy_decay: "cosine"     # NOT constant - the constant-LR drift mode is the #1 diagnosed failure
    policy_cooldown: "linear"
    parallel_scaling_policy: "sqrt"
  forecast :
      time_step: 24:00:00
      offset: 1
      num_steps: 4
      policy: "fixed"
      pushforward: False

validation_config:
  samples_per_mini_epoch: 64
  # inside ATMO coverage, disjoint from training, benchmark year untouched
  start_date: 2022-01-01T00:00
  end_date: 2022-12-31T00:00
```

Phase B: identical except `num_mini_epochs: 8`, `start_date: 2010-01-01T00:00` (recency window; `end_date` unchanged). Phase C: as B but `forecast.num_steps: 10` and `num_mini_epochs: 8`. B and C are launched with `train_continue --from-run-id <v3 run> --reuse-run-id` so the epoch counter and LR schedule continue.

- [ ] **Step 1: Write the three files** with the exact content above (B/C as deltas described).
- [ ] **Step 2: Verify**: `yaml.safe_load` each; assert phase A `end_date == '2023-06-30T00:00'`, `policy_decay == 'cosine'`, phase C `num_steps == 10`; assert none of the three contains `target_channel_weights` or a non-empty `freeze_modules`.
- [ ] **Step 3: Commit** (`config: v3 curriculum phase configs`).

### Task 7: Generalize the smoke tooling for v3 + 2023-H2 selection metric

**Files:**
- Modify: `/e/scratch/hclimrep/nowak2/eval_output/v2_model_smoke.sh`
- Create: `config/config_forecasting_glorys_obench_v3.yml`
- Create: `/e/scratch/hclimrep/nowak2/eval_output/compare_smoke.py`

**Interfaces:**
- Produces: `v2_model_smoke.sh <model_run_id> <short_tag> [wednesday] [mini_epoch] [obench_config] [output_stream]` — args 5/6 default to the v2 values so every existing invocation still works; `compare_smoke.py --runs tagA=run_idA,tagB=run_idB --week YYYYMMDD [--ref stage|parquet]` prints the weighted-RMSE/bias table for {zos, thetao@0.494/47.37/92.33/318.13, so@0.494} × leads {1,5,10}.

- [ ] **Step 1: obench v3 config** — copy `config/config_forecasting_glorys_obench_v2.yml`, change `streams_directory` to `./config/streams/glorys_v3_eval/` and the header comment's stream name to GLORYS3.
- [ ] **Step 2: smoke script args 5/6** — in `v2_model_smoke.sh` add `OBENCH_CFG="${5:-config/config_forecasting_glorys_obench_v2.yml}"` and `OUT_STREAM="${6:-GLORYS2}"`, use them in the `--config` and `output.streams=[...]` positions. `#SBATCH --account=` line → `e-ext-2025e01-128`. `bash -n` to verify.
- [ ] **Step 3: `compare_smoke.py`** — extract the ad-hoc comparison code used throughout this project into a script. Core (complete, working — this is the same code used for every comparison so far, parameterized):

```python
#!/usr/bin/env python
"""Weighted-RMSE/bias smoke comparison of inference zips vs GLORYS reference."""
import argparse
import numpy as np, xarray as xr, zarr, fsspec, json

STAGE = "/e/scratch/hclimrep/nowak2/oceanbench_stage/reference-glorys-quarter_degree-10d"
TRAIN_PARQ = "/e/data1/climateai/hclimrep/data/glorys_025/glorys_025_mesh_fixed.parq"
REF_LAT = np.arange(-78.0, 90.0, 0.25); SRC_LAT = np.arange(-89.875, 90.0, 0.25)
CHS = [("zos","zos",None),("thetao_0.494025m","thetao",0.494025),("thetao_47.3737m","thetao",47.3737),
       ("thetao_92.3261m","thetao",92.3261),("thetao_318.127m","thetao",318.127),("so_0.494025m","so",0.494025)]

def grid_from_mesh(vals, lats, lons):
    g = np.full((720,1440), np.nan, np.float32)
    r = np.round((lats+89.875)/0.25).astype(int); c = np.round((lons%360)/0.25).astype(int)%1440
    g[r,c] = vals; g = np.roll(g, 720, axis=1)
    da = xr.DataArray(g, coords={"lat":SRC_LAT,"lon":np.arange(-180,180,0.25)},dims=("lat","lon"))
    return da.interp(lat=REF_LAT).values

def wstat(d):
    w = np.cos(np.deg2rad(REF_LAT))[:,None]*np.ones(1440); m = np.isfinite(d)
    return np.nansum(d[m]*w[m])/np.nansum(w[m]), np.sqrt(np.nansum(d[m]**2*w[m])/np.nansum(w[m]))

def ref_field_stage(week, var, dep, ld):
    ds = xr.open_zarr(f"{STAGE}/{week}.zarr", consolidated=False)
    a = ds[var].isel(lead_day_index=ld)
    if dep is not None:
        a = a.isel(depth=int(np.argmin(np.abs(ds.depth.values - dep))))
    return a.values

def ref_field_parquet(week, var, dep, ld):
    # 2023-H2 selection: reference = training reanalysis parquet (daily, 12:00 stamps).
    # NEVER use the zips' own 'target' arrays (known-broken time indexing).
    m = fsspec.get_mapper("reference://", fo=TRAIN_PARQ, remote_protocol="file")
    ds = xr.open_dataset(m, engine="zarr", chunks={}, consolidated=False)
    day = np.datetime64(f"{week[:4]}-{week[4:6]}-{week[6:]}") + np.timedelta64(ld, "D")
    ti = int(np.argmin(np.abs(ds.time.values - (day + np.timedelta64(12, "h")))))
    name = f"{var}_{dep}m".replace("_Nonem","") if dep is not None else var
    cm = ds.attrs["weathergen_col_map"]; cm = json.loads(cm) if isinstance(cm, str) else cm
    v = ds[cm.get(name, name)].isel(time=ti).values
    lats = ds["lat"].values; lons = ds["lon"].values
    return grid_from_mesh(v.ravel(), lats.ravel(), lons.ravel())

def zip_errors(run_id, week, ref_mode):
    st = zarr.storage.ZipStore(f"/e/scratch/weatherai/shared_work/results/{run_id}/validation_chkpt00000_rank0000.zip", mode="r")
    root = zarr.open_group(st, mode="r")
    grp = [k for k in root["0"].keys() if k.startswith("GLORYS")][0]
    g = root["0"][grp]
    out = {}
    for fstep, ld in [(1,0),(5,4),(10,9)]:
        p = g[str(fstep)]["prediction"]
        chs = list(p.attrs["channels"]); d = p["data"][:] if "data" in p else p[:]
        coords = p["coords"][:]
        for ch, var, dep in CHS:
            i = chs.index(ch)
            pred = grid_from_mesh(d[:,i,0] if d.ndim==3 else d[:,i], coords[:,0], coords[:,1])
            ref = (ref_field_stage if ref_mode == "stage" else ref_field_parquet)(week, var, dep, ld)
            out[(ch, ld+1)] = wstat(pred - ref)
    st.close(); return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="tagA=run_idA,tagB=run_idB,...")
    ap.add_argument("--week", required=True, help="YYYYMMDD (a Wednesday)")
    ap.add_argument("--ref", choices=["stage","parquet"], default="stage")
    a = ap.parse_args()
    runs = dict(p.split("=") for p in a.runs.split(","))
    res = {t: zip_errors(r, a.week, a.ref) for t, r in runs.items()}
    tags = list(runs)
    print(f"{'channel/lead':26s}" + "".join(f"{t+'_rmse':>12s}{t+'_bias':>12s}" for t in tags))
    for ch, var, dep in CHS:
        for ld in (1,5,10):
            row = f"{ch:24s}LD{ld:<3d}"
            for t in tags:
                b, r = res[t][(ch,ld)]
                row += f"{r:>12.4f}{b:>12.4f}"
            print(row)
```

- [ ] **Step 4: Verify** against a known result: `.venv/bin/python /e/scratch/hclimrep/nowak2/eval_output/compare_smoke.py --runs ft0808=obench_ft0808_20240103 --week 20240103` must reproduce SST LD1 ≈ 0.7596 and zos LD1 ≈ 0.0796 (the recorded ft0808 numbers).
- [ ] **Step 5: Commit** repo-side file (`config/config_forecasting_glorys_obench_v3.yml`); eval_output scripts are untracked (note in ledger).

### Task 8: Launch v3 phase A and monitor with drift gates

Operational task. The launch uses the user's standard launcher (`weathergen_slurm_local.sh` wrapper); the ONLY changes vs their normal training launch are the config file and the account.

- [ ] **Step 1: Launch** (fresh `train` run — note run id from output):

```bash
sbatch -A e-ext-2025e01-128 weathergen_slurm_local.sh \
  --config config/config_glorys_v3_phaseA.yml
```

(If the launcher's argument passing differs, mirror exactly how `glory_finetuning_20260810_160149` was launched, substituting the config and adding `-A e-ext-2025e01-128`; do NOT pass `--from-run-id` — this is a fresh run.) Record `<V3_RUN_ID>` in the ledger. Verify from the first log: `GLORYS3` stream opened, ATMO source `['10u','10v','2d','2t','ssrd','strd']` (zarr order), 33 target channels, `freeze_modules` empty, lr policy cosine.

- [ ] **Step 2: Smoke cadence** — at epochs ≈4, 8, 12, ... (whenever `chkpt000NN` for NN divisible by 4 appears):

```bash
sbatch /e/scratch/hclimrep/nowak2/eval_output/v2_model_smoke.sh <V3_RUN_ID> v3eNN 2024-01-03 NN \
  config/config_forecasting_glorys_obench_v3.yml GLORYS3
sbatch /e/scratch/hclimrep/nowak2/eval_output/v2_model_smoke.sh <V3_RUN_ID> v3eNN 2024-07-03 NN \
  config/config_forecasting_glorys_obench_v3.yml GLORYS3
.venv/bin/python /e/scratch/hclimrep/nowak2/eval_output/compare_smoke.py \
  --runs v3=obench_v3eNN_20240103,ft0808=obench_ft0808_20240103 --week 20240103
```

**Drift gate (from the spec):** if surface-SST bias deepens by ≥0.1 °C across two consecutive smokes, pause the chain and surface the finding to Kacper before continuing — do not silently keep training.

- [ ] **Step 3: Completion** — after epoch 48, ledger line with the last smoke table.

### Task 9: Phases B and C (recency + 10-step)

- [ ] **Step 1: Launch phase B** when A completes:

```bash
sbatch -A e-ext-2025e01-128 weathergen_slurm_local.sh \
  --config config/config_glorys_v3_phaseB.yml --from-run-id <V3_RUN_ID> --reuse-run-id
```

- [ ] **Step 2: Smoke at B end** (epoch ≈56) exactly as Task 8 Step 2; same drift gate.
- [ ] **Step 3: Launch phase C** (same command with `phaseC` config), smoke at end (epoch ≈64).
- [ ] **Step 4: Ledger** the A/B/C smoke progression table.

### Task 10: Checkpoint selection on 2023-H2

- [ ] **Step 1:** For each candidate checkpoint (last ~6 saved epochs spanning phases B/C), run inference smokes for three 2023-H2 Wednesdays — 2023-07-05, 2023-09-06, 2023-11-01 — with `v2_model_smoke.sh <V3_RUN_ID> v3selNN <wed> NN config/config_forecasting_glorys_obench_v3.yml GLORYS3`. NOTE: these dates have no nowcast IC; the eval stream's source file only covers 2024 Tuesdays. Therefore selection smokes use the TRAINING streams instead: pass `config/config_glorys_v3_phaseA.yml`-style config — concretely, create `config/config_glorys_v3_selection.yml` = copy of `config_forecasting_glorys_obench_v3.yml` with `streams_directory: "./config/streams/glorys_v3/"` (training parquet as source: reanalysis ICs). This measures relative checkpoint quality, which is all selection needs.
- [ ] **Step 2:** Score each with `compare_smoke.py --ref parquet --week <YYYYMMDD>`; selection score = mean over the 3 dates of (SST_LD1 + SST_LD5 + zos_LD1·10 + t47_LD1 + t92_LD1) — surface-and-upper-ocean weighted, matching the contested rows. Pick the argmin; record the full table in the ledger.
- [ ] **Step 3:** Announce the selected epoch to Kacper with the table before phase-2 finetuning (gate).

### Task 11: Aggregation-engine finetune

**Files:**
- Create: `config/config_glorys_v3_aggft.yml`

- [ ] **Step 1: Config** — copy `config/config_glorys_v3_phaseC.yml` and change:

```yaml
freeze_modules: ".*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*forecast_engine.*|.*latent.*|.*GLORYS.*|.*ATMO.*"
training_config:
  num_mini_epochs: 4          # SHORT - the ft0818 lesson: long grinds reverse their gains
```

(rest identical to phase C: 2010+ window, 10-step, cosine decay, val 2022). The regex leaves ONLY the aggregation engine trainable — the ATMO embedding is frozen too (Kacper: leaving it trainable in earlier finetunes was a mistake); `.*GLORYS.*` also matches GLORYS3.

- [ ] **Step 2: Launch** from the selected epoch:

```bash
sbatch -A e-ext-2025e01-128 weathergen_slurm_local.sh \
  --config config/config_glorys_v3_aggft.yml --from-run-id <V3_RUN_ID> --mini-epoch <SELECTED_EPOCH>
```

Record `<AGGFT_RUN_ID>`. Verify freeze blocks in the first log: `Freeze block ... GLORYS3` AND `Freeze block ... ATMO` present, no `Freeze block ... aggregation` line (aggregation is the only trainable part).

- [ ] **Step 3: Selection smoke after EVERY epoch** (4 epochs, ~10 min each): same three 2023-H2 dates + the two 2024 stage dates. Keep the finetuned epoch only if it beats the selected base epoch on the Task 10 selection score AND its 2024-date SST bias has not deepened by more than 0.05 °C; otherwise the base epoch goes to eval. Record the decision in the ledger.

### Task 12: v3 full evaluation and scoreboard (phase 2 complete)

- [ ] **Step 1: 52-date inference** of the winning checkpoint (Task 11's decision):

```bash
./run_oceanbench_inference.sh \
  --model <WINNING_RUN_ID> \
  --run-prefix obench_v3final_ \
  --eval-config config/config_forecasting_glorys_obench_v3.yml \
  --output-stream GLORYS3
```

(If the winner is a specific epoch rather than `_latest`, first copy that checkpoint to `_latest` naming in a NEW model dir — or add `--mini-epoch` plumbing to the driver in the same style as Task 1 — whichever, verify the loaded checkpoint in the log.) Integrity sweep as Task 2 Step 3 with 33 channels expected.

- [ ] **Step 2: Export** via `sbatch_export.sh` with the same 14 levels, output `wg_v3_challenger.zarr`; read gate as Task 3 Step 4 (uo/vo will be NaN below 15.8101 m — assert `uo` at depth index 0 and 2 finite fraction > 0.6 and NaN fraction == 1.0 at index 3).
- [ ] **Step 3: Eval** as Task 4 Steps 1-3 with `challenger_v3.py` / report rename.
- [ ] **Step 4: Scoreboard rebuild + republish** as Task 4 Step 4.
- [ ] **Step 5: Success-criteria verdict**: recompute the rows-vs-GLONET table (Task 4 Step 5 script) for v3. PASS = ≥12/20 rows better at majority of leads AND wins on temp_50m, temp_100m, mld. Write the verdict + table to the ledger and present to Kacper with the scoreboard link.

## Self-review notes

- Spec coverage: phase 0 = Tasks 1-4; channel design = Task 5; schedule = Task 6; selection metric + drift gates = Tasks 7, 8, 10; aggregation finetune = Task 11; phase 2 = Task 12. Recency+rollout curriculum = Task 6/9. Account constraint = global + every sbatch shown.
- The 2023-H2 selection subtlety (no nowcast ICs) is resolved concretely in Task 10 (training-stream ICs, parquet reference, relative comparison only).
- Types/names consistent: `GLORYS3`, 33-target list defined once in Task 5 and referenced; smoke script signature defined in Task 7 and used in 8-11.

> **2026-09-01 amendment (verified, commit dfce5f61):** every evaluation — smokes, Task 10 selection, Task 12 full
> eval — uses the training ERA5 zarr for the ATMO stream (IC window only; the model consumes no forcing after the
> IC window). The OceanBench IFS parquets are retired: the benchmark forcing product is daily by construction and
> cost 20–45% RMSE. Task 10 may additionally use the official 2023 GLO12 nowcasts as ICs (dataset covers 2023–2025).
