# v3 Sea-Level Finetune (seaft) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finetune v3 epoch 45 for 4 epochs with gap-derived channel weights (zos-led) on 2020–2023 data, aggregation engine + GLORYS3 decoder trainable, select the best epoch on the two 2024 smoke dates, and push the winner through the full OceanBench evaluation.

**Architecture:** No model code changes. Everything is configuration plus small scripts: a stream directory carrying `target_channel_weights`, a finetune config that forks the v3 run at epoch 45 with a narrowed `freeze_modules` regex, a selection-rule script, and the existing smoke / export / OceanBench pipeline. The trainer derives the epoch counter from `general.istep` in the run json, so the finetune config pins `istep: 0` and only the first slurm link carries the config.

**Tech Stack:** WeatherGenerator (`src/weathergen/run_train.py train_continue`), OmegaConf YAML configs, slurm on JUPITER (account `e-ext-2025e01-128`), the eval scripts under `/e/scratch/hclimrep/nowak2/eval_output/` (`v2_model_smoke.sh`, `full_compare.py`, `smoke_table.py`, `sbatch_export.sh`, `run_oceanbench_eval.sh`, `build_scoreboard_data.py`), pytest via `.venv`.

**Spec:** `docs/superpowers/specs/2026-09-03-v3-sealevel-finetune-design.md`

## Global Constraints

- Base checkpoint: `glorys_v3_20260819_chkpt00045.chkpt` (`--mini-epoch 45` of run `glorys_v3_20260819`). Never use `-1` for this run (its `_latest` is ~e56).
- Slurm account for every job: `e-ext-2025e01-128`. Wall time per job ≤ 12:00:00 (QOS rejects longer).
- No heavy work on login nodes: inference/training on GPU nodes, exports/comparisons on CPU nodes via sbatch.
- Evaluation ATMO forcing = training ERA5 zarr (already wired in `config/streams/glorys_v3_eval/atmo.yml`); never the OceanBench IFS parquets.
- No output-side corrections (no bias correction, no blending). Model/training-side changes only.
- Trainable set: `encoder.ae_aggregation_engine` + `embed_target_coords.GLORYS3` + `target_token_engines.GLORYS3` + `pred_heads.GLORYS3`. Regex (YAML double-quoted): `".*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*forecast_engine.*|.*latent.*|encoder\\..*GLORYS.*|.*ATMO.*"`.
- Loss: `mse` only (no `dynamic_loss`), 10-step horizon, 24 h windows, `pushforward: False`.
- Window 2020-01-01T00:00 → 2023-12-31T00:00; validation 2019-01-01T00:00 → 2019-12-31T00:00; 4 mini-epochs × 4096 samples; `lr_max: 2e-5`, warmup 128, cooldown 512.
- Keep rule per epoch (both 2024 dates): family ratio < 1.00, rest ratio ≤ 1.01, LD1 SST bias not colder than e45 by > 0.05 °C.
- Before any same-run-id resubmission: `rm -rf /e/scratch/weatherai/shared_work/{models,results}/<run_id>` (stale-run-json trap).
- Every decision goes to `.superpowers/sdd/2026-08-19-v3-beat-glonet/progress.md`. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: Stream directory with gap-derived channel weights

**Files:**
- Create: `config/streams/glorys_v3_seaft/glorys.yml` (copy of `config/streams/glorys_v3/glorys.yml` + weights)
- Create: `config/streams/glorys_v3_seaft/atmo.yml` (byte-identical copy of `config/streams/glorys_v3/atmo.yml`)
- Test: `tests/test_seaft_stream_config.py`

**Interfaces:**
- Consumes: `eval_output/seaft_channel_weights.json` (rule weights, computed 2026-09-03), the v3 run json's `streams.GLORYS3.train_target_channels` (33 names).
- Produces: stream directory `./config/streams/glorys_v3_seaft/` whose `GLORYS3.target_channel_weights` is a mapping of exactly the 33 target channels.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seaft_stream_config.py
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
STREAM = REPO / "config/streams/glorys_v3_seaft/glorys.yml"
BASE_STREAM = REPO / "config/streams/glorys_v3/glorys.yml"
RUN_JSON = Path(
    "/e/scratch/weatherai/shared_work/models/glorys_v3_20260819/model_glorys_v3_20260819_chkpt00045.json"
)
EXPECTED = {
    "zos": 6.91, "thetao_0.494025m": 2.44, "thetao_9.573m": 3.16, "thetao_15.8101m": 3.47,
    "thetao_25.2114m": 1.0, "thetao_34.4342m": 1.0, "thetao_47.3737m": 1.36,
    "thetao_65.8073m": 3.28, "thetao_92.3261m": 2.16, "thetao_130.666m": 1.0,
    "thetao_155.851m": 1.54, "thetao_222.475m": 1.81, "thetao_318.127m": 1.52,
    "thetao_453.938m": 1.67, "thetao_541.089m": 1.72, "so_0.494025m": 1.16,
    "so_9.573m": 1.7, "so_15.8101m": 1.82, "so_25.2114m": 1.0, "so_34.4342m": 1.0,
    "so_47.3737m": 1.26, "so_65.8073m": 2.25, "so_92.3261m": 1.0, "so_130.666m": 1.0,
    "so_155.851m": 1.11, "so_222.475m": 1.59, "so_318.127m": 1.74, "so_453.938m": 2.17,
    "so_541.089m": 2.39, "uo_0.494025m": 3.0, "uo_15.8101m": 1.11, "vo_0.494025m": 3.0,
    "vo_15.8101m": 1.02,
}


def _load(path):
    return yaml.safe_load(path.read_text())


def test_weights_mapping_matches_rule_with_family_floor():
    w = _load(STREAM)["GLORYS3"]["target_channel_weights"]
    assert w == EXPECTED


def test_weights_cover_exactly_the_target_channels():
    w = _load(STREAM)["GLORYS3"]["target_channel_weights"]
    targets = json.loads(RUN_JSON.read_text())["streams"]["GLORYS3"]["train_target_channels"]
    assert len(targets) == 33
    assert set(w) == set(targets)


def test_only_weights_differ_from_base_stream():
    seaft = _load(STREAM)["GLORYS3"]
    base = _load(BASE_STREAM)["GLORYS3"]
    seaft.pop("target_channel_weights")
    assert seaft == base


def test_atmo_stream_is_unchanged_era5():
    seaft = (REPO / "config/streams/glorys_v3_seaft/atmo.yml").read_text()
    base = (REPO / "config/streams/glorys_v3/atmo.yml").read_text()
    assert seaft == base
    assert "aifs-ea-an-oper-0001-mars-n320-1979-2024-6h-v1-for-single-v2.zarr" in seaft
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /e/scratch/hclimrep/nowak2/WeatherGenerator2 && .venv/bin/python -m pytest tests/test_seaft_stream_config.py -v`
Expected: 4 FAIL (FileNotFoundError on `config/streams/glorys_v3_seaft/glorys.yml`).

- [ ] **Step 3: Create the stream directory**

```bash
cd /e/scratch/hclimrep/nowak2/WeatherGenerator2
mkdir -p config/streams/glorys_v3_seaft
cp config/streams/glorys_v3/atmo.yml config/streams/glorys_v3_seaft/atmo.yml
python3 - <<'EOF'
import json
src = open("config/streams/glorys_v3/glorys.yml").read()
w = json.load(open("/e/scratch/hclimrep/nowak2/eval_output/seaft_channel_weights.json"))
w["uo_0.494025m"] = 3.0   # family floor (spec §4): surface currents feed geo_u/v + Lagrangian rows
w["vo_0.494025m"] = 3.0
block = ["  # seaft: gap-derived static loss weights (spec docs/superpowers/specs/2026-09-03-v3-sealevel-finetune-design.md §4)",
         "  #   weight = clip(1 + 40*(RMSE_e45/RMSE_ft0808 - 1), 1, 8), both 2024 smoke weeks x leads 1/5/10;",
         "  #   uo/vo surface floored at 3.0 (family floor). Applied on the plain-MSE path (dynamic_loss off).",
         "  target_channel_weights:"]
block += [f"    {k}: {v}" for k, v in w.items()]
# insert right after the loss_weight line so the mapping sits with the stream-level settings
assert src.count("  loss_weight: 1.0\n") == 1
src = src.replace("  loss_weight: 1.0\n", "  loss_weight: 1.0\n" + "\n".join(block) + "\n", 1)
open("config/streams/glorys_v3_seaft/glorys.yml", "w").write(src)
print("written")
EOF
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_seaft_stream_config.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add config/streams/glorys_v3_seaft tests/test_seaft_stream_config.py
git commit -m "seaft: glorys_v3_seaft stream dir with gap-derived target_channel_weights (zos 6.91, uo/vo floor 3.0)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Finetune config with narrowed freeze regex

**Files:**
- Create: `config/config_glorys_v3_seaft.yml`
- Test: `tests/test_seaft_config.py`

**Interfaces:**
- Consumes: `config/config_glorys_v3_phaseC.yml` (architecture block + phase-C recipe), Task 1 stream directory.
- Produces: the config passed as `WEATHERGEN_CONFIG_EXTRA` on the first link. Keys later tasks rely on: `training_config.num_mini_epochs == 4`, `general.istep == 0`, `freeze_modules` regex.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seaft_config.py
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "config/config_glorys_v3_seaft.yml"
PHASE_C = REPO / "config/config_glorys_v3_phaseC.yml"

ARCH_KEYS = [
    "ae_local_dim_embed", "ae_local_num_blocks", "ae_local_max_tokens_per_cell",
    "ae_global_dim_embed", "ae_global_num_blocks", "ae_aggregation_num_blocks",
    "fe_num_blocks", "decoder_type", "healpix_level", "embed_orientation",
]

# representative named_modules() names; the regex is applied with re.fullmatch
MODULES_FROZEN = [
    "encoder.ae_local_engine", "encoder.ae_local_global_engine", "encoder.ae_global_engine",
    "encoder.q_cells", "forecast_engine", "latent_heads",
    "encoder.embed_engine.GLORYS3", "encoder.embed_engine.ATMO",
]
MODULES_TRAINABLE = [
    "encoder.ae_aggregation_engine", "embed_target_coords.GLORYS3",
    "target_token_engines.GLORYS3", "pred_heads.GLORYS3",
]


def _cfg():
    return yaml.safe_load(CFG.read_text())


def test_freeze_regex_leaves_only_aggregation_and_decoder_trainable():
    rx = _cfg()["freeze_modules"]
    for name in MODULES_FROZEN:
        assert re.fullmatch(rx, name), f"expected frozen: {name}"
    for name in MODULES_TRAINABLE:
        assert re.fullmatch(rx, name) is None, f"expected trainable: {name}"


def test_run_mechanics_and_schedule():
    c = _cfg()
    assert c["streams_directory"] == "./config/streams/glorys_v3_seaft/"
    assert c["general"]["istep"] == 0
    t = c["training_config"]
    assert t["num_mini_epochs"] == 4
    assert t["samples_per_mini_epoch"] == 4096
    assert t["start_date"] == "2020-01-01T00:00" and t["end_date"] == "2023-12-31T00:00"
    assert t["forecast"]["num_steps"] == 10 and t["forecast"]["pushforward"] is False
    assert list(t["losses"]["physical"]["loss_fcts"]) == ["mse"]
    lr = t["learning_rate_scheduling"]
    assert lr["lr_max"] == 2e-5 and lr["num_steps_warmup"] == 128 and lr["num_steps_cooldown"] == 512
    assert lr["policy_decay"] == "cosine"
    v = c["validation_config"]
    assert v["start_date"] == "2019-01-01T00:00" and v["end_date"] == "2019-12-31T00:00"
    assert v["validate_with_ema"]["enabled"] is True
    assert c["train_logging"]["checkpoint"] == 64


def test_architecture_block_identical_to_phase_c():
    c, p = _cfg(), yaml.safe_load(PHASE_C.read_text())
    for k in ARCH_KEYS:
        assert c[k] == p[k], k
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seaft_config.py -v`
Expected: 3 FAIL (FileNotFoundError on `config/config_glorys_v3_seaft.yml`).

- [ ] **Step 3: Create the config from phase C**

```bash
cd /e/scratch/hclimrep/nowak2/WeatherGenerator2
python3 - <<'EOF'
src = open("config/config_glorys_v3_phaseC.yml").read()

def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (old[:60], src.count(old))
    src = src.replace(old, new)

# header
rep('# v3 GLORYS curriculum, phase C:', '# v3 sea-level finetune (seaft), spec docs/superpowers/specs/2026-09-03-v3-sealevel-finetune-design.md.\n# Forks glorys_v3_20260819 at epoch 45 (train_continue --mini-epoch 45 into a NEW run id).\n# Trainable: aggregation engine + GLORYS3 decoder stack; everything else frozen.\n# (Original phase C header follows for provenance.)\n# v3 GLORYS curriculum, phase C:')
rep('streams_directory: "./config/streams/glorys_v3/"', 'streams_directory: "./config/streams/glorys_v3_seaft/"')
rep('freeze_modules: ""', 'freeze_modules: ".*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*forecast_engine.*|.*latent.*|encoder\\\\..*GLORYS.*|.*ATMO.*"')
# istep: this config is passed ONLY on the first link (fresh counter for the forked run).
rep('  # istep: 0\n  run_id: ???', '  # seaft: fresh epoch counter for the forked run. The trainer derives the epoch from\n  # istep in the run json; the inherited v3 value (~23k) would put it at 45 > 4 and the\n  # loop would never run. Pass this config on the FIRST link only; later links pure-resume.\n  istep: 0\n  run_id: ???')
rep('  num_mini_epochs: 96', '  num_mini_epochs: 4   # short finetune (ft0818 lesson: long grinds reverse their gains)')
rep('''        mse: {}
        dynamic_loss:
          window: 128
          L: 5.0   # 2026-08-21: lowered from 20 (Kacper) - cap weight spread 5:1 so high-error T channels keep gradient''',
    '''        mse: {}
        # dynamic_loss OFF: static target_channel_weights (stream config) apply on the plain-MSE path''')
rep('  start_date: 2010-01-01T00:00\n  end_date: 2023-06-30T00:00', '  start_date: 2020-01-01T00:00\n  end_date: 2023-12-31T00:00   # recency window incl. the warm 2023-H2 regime (Kacper ruling)')
rep('    lr_max: 7e-5   # 2026-08-22: raised from 5e-5 (Kacper) with 96-epoch stretch', '    lr_max: 2e-5   # finetune from a converged state')
rep('    num_steps_warmup: 256', '    num_steps_warmup: 128')
rep('  start_date: 2022-01-01T00:00\n  end_date: 2022-12-31T00:00', '  start_date: 2019-01-01T00:00\n  end_date: 2019-12-31T00:00   # 2022 is inside the finetune window')
open("config/config_glorys_v3_seaft.yml", "w").write(src)
print("written")
EOF
grep -nE "freeze_modules|istep|num_mini_epochs|start_date|end_date|lr_max|dynamic|streams_directory" config/config_glorys_v3_seaft.yml
```

The `freeze_modules` line must read exactly (single backslash inside the YAML double quotes):
`freeze_modules: ".*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*forecast_engine.*|.*latent.*|encoder\..*GLORYS.*|.*ATMO.*"`

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_seaft_config.py tests/test_seaft_stream_config.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add config/config_glorys_v3_seaft.yml tests/test_seaft_config.py
git commit -m "seaft: finetune config - fork v3@e45, aggregation+GLORYS3 decoder trainable, mse x static weights, 2020-2023, 4 epochs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Selection-rule script `seaft_score.py`

**Files:**
- Create: `scripts/seaft_score.py`
- Test: `tests/test_seaft_score.py`

**Interfaces:**
- Consumes: a `full_compare.py` output file (header `channel/lead` + per-tag `<tag>_rmse`/`<tag>_bias` columns, rows `<channel> LD<n> ...`, weeks separated by `=== week YYYYMMDD ===`).
- Produces: CLI `python scripts/seaft_score.py <compare_file> <candidate_tag> <base_tag>`; prints one line per week plus `PASS`/`FAIL`; exit 0 on pass, 1 on fail. Function `score(path, cand, base) -> dict[week, dict(family, rest, sst_bias_delta, ok)]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seaft_score.py
import importlib.util
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/seaft_score.py"
spec = importlib.util.spec_from_file_location("seaft_score", SCRIPT)
seaft_score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seaft_score)

HEADER = "channel/lead              candA_rmse  candA_bias  base_rmse  base_bias\n"


def _file(tmp_path, rows_by_week):
    out = ""
    for week, rows in rows_by_week.items():
        out += f"=== week {week} ===\n" + HEADER
        for ch, ld, cr, cb, br, bb in rows:
            out += f"{ch:24s}LD{ld:<3d}{cr:12.4f}{cb:12.4f}{br:12.4f}{bb:12.4f}\n"
    p = tmp_path / "cmp.txt"
    p.write_text(out)
    return p


def _rows(fam_ratio, rest_ratio, sst_bias_cand, sst_bias_base=-0.20):
    rows = []
    for ld in (1, 5, 10):
        rows.append(("zos", ld, 0.06 * fam_ratio, -0.03, 0.06, -0.03))
        rows.append(("uo_0.494025m", ld, 0.11 * fam_ratio, 0.0, 0.11, 0.0))
        rows.append(("vo_0.494025m", ld, 0.11 * fam_ratio, 0.0, 0.11, 0.0))
        rows.append(("thetao_0.494025m", ld, 0.60 * rest_ratio, sst_bias_cand, 0.60, sst_bias_base))
        rows.append(("so_541.089m", ld, 0.08 * rest_ratio, 0.0, 0.08, 0.0))
        rows.append(("thetao_9.573m", ld, 5.0, -1.0, 1.0, -1.0))  # artifact depth: must be ignored
    return rows


def test_passing_epoch(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(0.95, 1.00, -0.21), "20240703": _rows(0.97, 1.005, -0.22)})
    res = seaft_score.score(p, "candA", "base")
    assert res["20240103"]["ok"] and res["20240703"]["ok"]
    assert abs(res["20240103"]["family"] - 0.95) < 1e-6
    assert abs(res["20240103"]["rest"] - 1.00) < 1e-6
    assert abs(res["20240103"]["sst_bias_delta"] - (-0.01)) < 1e-6
    assert seaft_score.overall(res) is True


def test_fails_on_family_not_improved(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(1.00, 1.00, -0.20), "20240703": _rows(0.9, 1.0, -0.2)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is False


def test_fails_on_rest_regression(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(0.9, 1.02, -0.20), "20240703": _rows(0.9, 1.0, -0.2)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is False


def test_fails_on_cold_bias(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(0.9, 1.0, -0.26), "20240703": _rows(0.9, 1.0, -0.2)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is False


def test_artifact_depths_are_ignored(tmp_path):
    # the thetao_9.573m row has ratio 5.0; if it were counted, rest would blow past 1.01
    p = _file(tmp_path, {"20240103": _rows(0.9, 1.0, -0.2)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seaft_score.py -v`
Expected: FAIL at import (`scripts/seaft_score.py` missing).

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python
"""seaft epoch selection rule (spec §7) over a full_compare.py output.

usage: seaft_score.py <compare_file> <candidate_tag> <base_tag>
exit 0 = candidate passes on every week in the file, 1 = fails.
"""
import re
import sys

FAMILY = {"zos", "uo_0.494025m", "vo_0.494025m"}
# depths absent from the staged reference (scored vs nearest scoreboard level) - never counted
ARTIFACT_DEPTHS = {"9.573", "15.8101", "25.2114", "34.4342", "65.8073", "130.666"}
LEADS = (1, 5, 10)
FAMILY_MAX = 1.00      # strictly better than base on the family
REST_MAX = 1.01        # everything else within 1%
SST_BIAS_MIN_DELTA = -0.05   # LD1 SST bias may not be colder than base by more than this


def parse(path):
    weeks, cur, cols = {}, None, None
    for line in open(path):
        if line.startswith("==="):
            cur = line.split()[2]
            weeks[cur] = {}
            continue
        if line.startswith("channel/lead"):
            cols = re.findall(r"(.+?_(?:rmse|bias))", line.split(None, 1)[1].replace(" ", ""))
            continue
        m = re.match(r"(\S+)\s+LD(\d+)\s+(.*)", line)
        if not m or cur is None:
            continue
        vals = [float(v) if v != "--" else None for v in m.group(3).split()]
        weeks[cur][(m.group(1), int(m.group(2)))] = dict(zip(cols, vals))
    return weeks


def _is_artifact(ch):
    dep = ch.split("_")[-1].rstrip("m") if "_" in ch else ""
    return dep in ARTIFACT_DEPTHS


def score(path, cand, base):
    out = {}
    for week, rows in parse(path).items():
        fam, rest = [], []
        for (ch, ld), r in rows.items():
            if ld not in LEADS or _is_artifact(ch):
                continue
            a, b = r.get(f"{cand}_rmse"), r.get(f"{base}_rmse")
            if a is None or b is None:
                continue
            (fam if ch in FAMILY else rest).append(a / b)
        sst = rows[("thetao_0.494025m", 1)]
        delta = sst[f"{cand}_bias"] - sst[f"{base}_bias"]
        fam_s, rest_s = sum(fam) / len(fam), sum(rest) / len(rest)
        ok = fam_s < FAMILY_MAX and rest_s <= REST_MAX and delta >= SST_BIAS_MIN_DELTA
        out[week] = {"family": fam_s, "rest": rest_s, "sst_bias_delta": delta, "ok": ok}
    return out


def overall(res):
    return bool(res) and all(v["ok"] for v in res.values())


if __name__ == "__main__":
    path, cand, base = sys.argv[1:4]
    res = score(path, cand, base)
    for week, v in res.items():
        print(f"{week}: family {v['family']:.4f} (<{FAMILY_MAX})  rest {v['rest']:.4f} (<={REST_MAX})  "
              f"SST-bias delta {v['sst_bias_delta']:+.3f} (>={SST_BIAS_MIN_DELTA})  -> {'ok' if v['ok'] else 'FAIL'}")
    print("PASS" if overall(res) else "FAIL")
    sys.exit(0 if overall(res) else 1)
```

Save as `scripts/seaft_score.py` and `chmod +x scripts/seaft_score.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_seaft_score.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/seaft_score.py tests/test_seaft_score.py
git commit -m "seaft: selection-rule script (family <1.00, rest <=1.01, SST bias delta >= -0.05) with tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Launch the first (test) link and verify it before committing more compute

**Files:**
- Create: `/e/scratch/hclimrep/nowak2/eval_output/verify_seaft_start.sh`
- Modify: `.superpowers/sdd/2026-08-19-v3-beat-glonet/progress.md` (append)

**Interfaces:**
- Consumes: Task 1 stream dir, Task 2 config, `weathergen_slurm_local.sh` (env: `WEATHERGEN_STAGE`, `RUN_ID`, `FROM_RUN_ID`, `WEATHERGEN_MINI_EPOCH`, `WEATHERGEN_CONFIG_EXTRA`).
- Produces: run id `glorys_v3_seaft_<YYYYMMDD>` with `logs/<RUN_ID>/output.<jobid>.txt`, `models/<RUN_ID>/model_<RUN_ID>*.json`, checkpoints `chkpt0000N`; the first-link job id `<LINK1>` for Task 5.

- [ ] **Step 1: Write the bounded verification script**

```bash
cat > /e/scratch/hclimrep/nowak2/eval_output/verify_seaft_start.sh <<'EOF'
#!/bin/bash
# usage: verify_seaft_start.sh <RUN_ID> <JOBID>  - waits (<= 6 h) for the first terminal line, then prints the acceptance items.
RUN_ID="$1"; JOB="$2"
LOG=/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/$RUN_ID/output.$JOB.txt
JSON=/e/scratch/weatherai/shared_work/models/$RUN_ID/model_${RUN_ID}.json
for i in $(seq 1 360); do
  [ -f "$LOG" ] && grep -qE "^0: 00[0-9] : [0-9]{5}" "$LOG" && break
  squeue -h -j "$JOB" >/dev/null 2>&1 || { echo "JOB GONE without training output"; break; }
  sleep 60
done
echo "--- freeze set (must include local/global/forecast_engine/ATMO; must NOT include ae_aggregation_engine or pred_heads.GLORYS3) ---"
grep -oE "Freeze block \S+" "$LOG" | sort -u
echo "--- config echo ---"
grep -oE "'num_steps': [0-9]+" "$LOG" | head -1
grep -oE "'num_mini_epochs': [0-9]+" "$LOG" | head -1
grep -oE "'start_date': '[0-9T:-]+'" "$LOG" | head -2
grep -oE "'lr_max': [0-9.e-]+" "$LOG" | head -1
grep -oE "'loss_fcts': \{[^}]*\}" "$LOG" | head -1
echo "--- weights vector from run json (33 entries; zos first = 6.91; uo/vo surface = 3.0) ---"
python3 - "$JSON" <<'PY'
import json, sys
cf = json.load(open(sys.argv[1]))
s = cf["streams"]["GLORYS3"]
w = s.get("target_channel_weights"); t = s.get("train_target_channels")
print("n_weights", len(w) if w else None, "n_targets", len(t))
print({c: x for c, x in zip(t, w) if c in ("zos", "uo_0.494025m", "vo_0.494025m", "thetao_0.494025m", "so_541.089m")})
print("istep", cf["general"]["istep"], "freeze", cf.get("freeze_modules"))
PY
echo "--- first terminal lines ---"
grep -E "^0: [0-9]{3} : [0-9]{5}/" "$LOG" | head -3
EOF
chmod +x /e/scratch/hclimrep/nowak2/eval_output/verify_seaft_start.sh
```

- [ ] **Step 2: Submit the first link (carries the config; forks v3 at epoch 45)**

```bash
cd /e/scratch/hclimrep/nowak2/WeatherGenerator2
export WEATHERGEN_HOME=/e/scratch/hclimrep/nowak2/WeatherGenerator2
export WEATHERGEN_PRIVATE_REPO_PATH=/e/scratch/hclimrep/nowak2/WeatherGenerator-private
export WEATHERGEN_BASE_CONFIG=config/default_config.yml
export WEATHERGEN_STAGE=train_continue
export RUN_ID=glorys_v3_seaft_$(date +%Y%m%d)
export FROM_RUN_ID=glorys_v3_20260819
export WEATHERGEN_MINI_EPOCH=45
export WEATHERGEN_CONFIG_EXTRA=config/config_glorys_v3_seaft.yml
rm -rf /e/scratch/weatherai/shared_work/models/$RUN_ID /e/scratch/weatherai/shared_work/results/$RUN_ID
mkdir -p logs/$RUN_ID
LINK1=$(sbatch --parsable --account=e-ext-2025e01-128 --time=12:00:00 --job-name=${RUN_ID}_L1 weathergen_slurm_local.sh)
echo "RUN_ID=$RUN_ID LINK1=$LINK1" | tee /e/scratch/hclimrep/nowak2/eval_output/seaft_run.env
```

- [ ] **Step 3: Verify (run in the background; it returns when the first terminal line appears)**

```bash
source /e/scratch/hclimrep/nowak2/eval_output/seaft_run.env
/e/scratch/hclimrep/nowak2/eval_output/verify_seaft_start.sh $RUN_ID $LINK1
```

Acceptance (all must hold, else `scancel $LINK1`, fix, `rm -rf models/$RUN_ID results/$RUN_ID`, resubmit Step 2):
- `Freeze block` lines include the local, global, forecast_engine and ATMO modules; **none** for `ae_aggregation_engine`, `target_token_engines.GLORYS3`, `pred_heads.GLORYS3`, `embed_target_coords.GLORYS3`.
- `'num_steps': 10`, `'num_mini_epochs': 4`, `'start_date': '2020-01-01T00:00'` (training) and `'2019-01-01T00:00'` (validation), `'lr_max': 2e-05`, `loss_fcts` contains only `mse`.
- Run json: `n_weights 33`, `zos 6.91`, `uo_0.494025m 3.0`, `vo_0.494025m 3.0`, `istep 0`.
- First terminal line starts with `0: 000 : 00010/00512`.

- [ ] **Step 4: Ledger**

Append to `.superpowers/sdd/2026-08-19-v3-beat-glonet/progress.md`:
```
## <date> seaft Task 4: first link <LINK1> for <RUN_ID> (fork v3@e45) - VERIFIED / FAILED (<reason>)
- freeze set: <paste the sorted Freeze block names>
- config echo: num_steps 10, epochs 4, 2020-01-01..2023-12-31, val 2019, lr_max 2e-5, mse only, weights n=33 zos 6.91 uo/vo 3.0, istep 0
```

---

### Task 5: Second link and per-epoch selection smokes

**Files:**
- Create: `/e/scratch/hclimrep/nowak2/eval_output/seaft_epoch_smoke.sh`
- Modify: `.superpowers/sdd/2026-08-19-v3-beat-glonet/progress.md` (append per epoch)

**Interfaces:**
- Consumes: `seaft_run.env` (`RUN_ID`, `LINK1`), checkpoints `models/<RUN_ID>/<RUN_ID>_chkpt0000N.chkpt` (N = 0..3), `v2_model_smoke.sh`, `full_compare.py`, `smoke_table.py`, `scripts/seaft_score.py`, baselines `obench_v3e45nowera5_<week>` and `obench_ft0808nowera5_<week>`.
- Produces: `compare_seaft_e<N>.txt`, `seaft_e<N>_score.txt` per epoch; the ledger decision; the chosen epoch `<BEST_N>` for Task 6.

- [ ] **Step 1: Queue the second link as a pure resume (after Task 4 acceptance only)**

```bash
cd /e/scratch/hclimrep/nowak2/WeatherGenerator2
source /e/scratch/hclimrep/nowak2/eval_output/seaft_run.env
export WEATHERGEN_HOME=/e/scratch/hclimrep/nowak2/WeatherGenerator2
export WEATHERGEN_PRIVATE_REPO_PATH=/e/scratch/hclimrep/nowak2/WeatherGenerator-private
export WEATHERGEN_BASE_CONFIG=config/default_config.yml
export WEATHERGEN_STAGE=train_continue
export RUN_ID FROM_RUN_ID=$RUN_ID
export WEATHERGEN_MINI_EPOCH=-1
export WEATHERGEN_CONFIG_EXTRA=""        # pure resume: keeps istep/epoch counter from the run json
LINK2=$(sbatch --parsable --account=e-ext-2025e01-128 --time=12:00:00 --job-name=${RUN_ID}_L2 --dependency=afterany:$LINK1 weathergen_slurm_local.sh)
echo "LINK2=$LINK2" >> /e/scratch/hclimrep/nowak2/eval_output/seaft_run.env
```
(Training exits on its own when epoch 3 finishes; a third link would find nothing to do.)

- [ ] **Step 2: Write the per-epoch smoke driver**

```bash
cat > /e/scratch/hclimrep/nowak2/eval_output/seaft_epoch_smoke.sh <<'EOF'
#!/bin/bash
# usage: seaft_epoch_smoke.sh <N>   (after models/<RUN_ID>/<RUN_ID>_chkpt0000N.chkpt exists)
set -u
N="$1"; source /e/scratch/hclimrep/nowak2/eval_output/seaft_run.env
E=/e/scratch/hclimrep/nowak2/eval_output; S=$E/v2_model_smoke.sh; TAG=seaft$N
CK=/e/scratch/weatherai/shared_work/models/$RUN_ID/${RUN_ID}_chkpt0000${N}.chkpt
[ -f "$CK" ] || { echo "missing $CK"; exit 2; }
cd /e/scratch/hclimrep/nowak2/WeatherGenerator2
rm -rf /e/scratch/weatherai/shared_work/{models,results}/obench_${TAG}_*
J1=$(sbatch --parsable $S $RUN_ID $TAG 2024-01-03 $N config/config_forecasting_glorys_obench_v3.yml GLORYS3)
J2=$(sbatch --parsable $S $RUN_ID $TAG 2024-07-03 $N config/config_forecasting_glorys_obench_v3.yml GLORYS3)
cat > $E/sbatch_compare_${TAG}.sh <<EOS
#!/bin/bash
#SBATCH --account=e-ext-2025e01-128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=${TAG}-compare
#SBATCH --output=$E/logs/%j.out
#SBATCH --error=$E/logs/%j.err
cd $E
P=/e/scratch/hclimrep/nowak2/WeatherGenerator2/.venv/bin/python
{ for W in 20240103 20240703; do echo "=== week \$W ==="; \$P full_compare.py --week \$W --runs ${TAG}=obench_${TAG}_\$W,v3e45=obench_v3e45nowera5_\$W,ft0808=obench_ft0808nowera5_\$W; done; } > compare_${TAG}.txt
\$P /e/scratch/hclimrep/nowak2/WeatherGenerator2/scripts/seaft_score.py compare_${TAG}.txt ${TAG} v3e45 > ${TAG}_score.txt; echo "score exit \$?" >> ${TAG}_score.txt
\$P smoke_table.py compare_${TAG}.txt ${TAG} v3e45 > ${TAG}_vs_e45.md
\$P smoke_table.py compare_${TAG}.txt ${TAG} ft0808 > ${TAG}_vs_ft0808.md
EOS
CJ=$(sbatch --parsable --dependency=afterany:$J1:$J2 $E/sbatch_compare_${TAG}.sh)
echo "$TAG: smokes $J1 $J2 -> compare $CJ (outputs: compare_${TAG}.txt, ${TAG}_score.txt, ${TAG}_vs_e45.md, ${TAG}_vs_ft0808.md)"
EOF
chmod +x /e/scratch/hclimrep/nowak2/eval_output/seaft_epoch_smoke.sh
```

- [ ] **Step 3: Run the smoke for every epoch as its checkpoint appears (N = 0, 1, 2, 3)**

```bash
/e/scratch/hclimrep/nowak2/eval_output/seaft_epoch_smoke.sh 0     # then 1, 2, 3
```
When the compare job finishes: `cat seaft<N>_score.txt` (rule verdict) and read `seaft<N>_vs_e45.md` / `seaft<N>_vs_ft0808.md`.

- [ ] **Step 4: Decide and ledger after each epoch**

Append to the ledger, per epoch: the three rule numbers per week, `PASS`/`FAIL`, and the notable rows (zos LD1/LD10, uo/vo LD10, SST bias). Rule for the retry clause (spec §10): if **epoch 0** fails only on the `rest` gate, halve the gain once (regenerate weights with `A=20` in Task 1's Step 3 — same script, `1 + 20*(r-1)` — commit, `scancel` both links, `rm -rf models/$RUN_ID results/$RUN_ID`, redo Task 4); no second retry.

- [ ] **Step 5: Choose `<BEST_N>`**

Among passing epochs, the lowest mean family score over both weeks; record `BEST_N=<n>` in `seaft_run.env` (`echo "BEST_N=<n>" >> .../seaft_run.env`). If none passes: `BEST_N=none`, e45 goes to Task 6 unchanged (Task 6 then uses `--model glorys_v3_20260819 --mini-epoch 45`, which is the already-running e45 evaluation — nothing to launch).

---

### Task 6: Full evaluation of the winner and scoreboard update

**Files:**
- Create: `/e/scratch/hclimrep/nowak2/eval_output/challenger_v3seaft.py`
- Modify: `.superpowers/sdd/2026-08-19-v3-beat-glonet/progress.md`, `/e/scratch/hclimrep/nowak2/eval_output/scoreboard.html` (data blob)

**Interfaces:**
- Consumes: `seaft_run.env` (`RUN_ID`, `BEST_N`), `run_oceanbench_inference.sh` (`--mini-epoch`), `sbatch_export.sh <prefix> <stream_group> <levels> <zarr>`, `run_oceanbench_eval.sh <challenger.py> <stage_dir>`, `build_scoreboard_data.py`.
- Produces: `wg_v3seaft_challenger.zarr`, `challenger_v3seaft.global.report.ipynb`, updated `scoreboard_data.json` + artifact.

- [ ] **Step 1: Inference as two parallel 12 h jobs (26 dates each), export and eval chained in slurm**

```bash
cd /e/scratch/hclimrep/nowak2/WeatherGenerator2
source /e/scratch/hclimrep/nowak2/eval_output/seaft_run.env; [ "$BEST_N" != "none" ] || { echo "no passing epoch: e45 eval already running"; exit 0; }
D1=""; D2=""; for k in $(seq 0 51); do W=$(date -u -d "2024-01-03 +$((7*k)) days" +%Y-%m-%d); if [ $k -lt 26 ]; then D1="$D1 $W"; else D2="$D2 $W"; fi; done
COMMON=(--model $RUN_ID --run-prefix obench_seaftfull_ --eval-config config/config_forecasting_glorys_obench_v3.yml --output-stream GLORYS3 --mini-epoch $BEST_N)
rm -rf /e/scratch/weatherai/shared_work/{models,results}/obench_seaftfull_*
J1=$(sbatch --parsable --time=12:00:00 run_oceanbench_inference.sh --dates "${D1# }" "${COMMON[@]}")
J2=$(sbatch --parsable --time=12:00:00 run_oceanbench_inference.sh --dates "${D2# }" "${COMMON[@]}")
cd /e/scratch/hclimrep/nowak2/eval_output
sed 's#wg_v3e45_challenger.zarr#wg_v3seaft_challenger.zarr#; s#v3 epoch 45 (glorys_v3_20260819, ERA5 IC-window forcing)#v3 sea-level finetune (seaft, ERA5 IC-window forcing)#' challenger_v3e45.py > challenger_v3seaft.py
LEVELS="0.494025,9.573,15.8101,25.2114,34.4342,47.3737,65.8073,92.3261,130.666,155.851,222.475,318.127,453.938,541.089"
EJ=$(sbatch --parsable --dependency=afterok:$J1:$J2 sbatch_export.sh obench_seaftfull_ GLORYS3 "$LEVELS" /e/scratch/hclimrep/nowak2/eval_output/wg_v3seaft_challenger.zarr)
VJ=$(sbatch --parsable --dependency=afterok:$EJ run_oceanbench_eval.sh /e/scratch/hclimrep/nowak2/eval_output/challenger_v3seaft.py /e/scratch/hclimrep/nowak2/oceanbench_stage)
echo "inference $J1 $J2 -> export $EJ -> eval $VJ"
```

- [ ] **Step 2: Verify the chain completed**

```bash
ls /e/scratch/weatherai/shared_work/results/obench_seaftfull_*/validation_chkpt00000_rank0000.zip | wc -l   # 52
ls -la /e/scratch/hclimrep/nowak2/eval_output/challenger_v3seaft.global.report.ipynb
```
If the eval notebook is missing, `sacct -j $EJ,$VJ` tells which stage failed; rerun that stage only (all three scripts are idempotent per output).

- [ ] **Step 3: Scoreboard**

```bash
cd /e/scratch/hclimrep/nowak2/eval_output
cp challenger_v3seaft.global.report.ipynb challenger_v3seaft_global_report.ipynb
/e/scratch/hclimrep/nowak2/oceanbench/.venv/bin/python build_scoreboard_data.py --no-download
/e/scratch/hclimrep/nowak2/WeatherGenerator2/.venv/bin/python - <<'EOF'
import json
d = json.load(open('scoreboard_data.json'))
d['versions'] = {'0.4.0': d['versions']['0.4.0']}
ms = d['versions']['0.4.0']['models']
keep = {'ft0808era', 'v3seaft'}
for k in [k for k in ms if not ms[k]['official'] and k not in keep]:
    del ms[k]
blob = json.dumps(d, indent=1)
html = open('scoreboard.html').read()
s = html.index('/*DATA-START*/') + len('/*DATA-START*/'); e = html.index('/*DATA-END*/')
html = html[:s] + blob + html[e:]
html = html.replace('const NAME = { ft0808era: "WG ft0808", ', 'const NAME = { v3seaft: "WG v3 seaft", ft0808era: "WG ft0808", ')
open('scoreboard.html', 'w').write(html)
print('models:', list(ms))
EOF
```
Then republish the artifact from this session (`Artifact` tool, `file_path=/e/scratch/hclimrep/nowak2/eval_output/scoreboard.html`, `url=https://claude.ai/code/artifact/b1df873c-b017-4311-aabd-c708d23658df`) after reading the current published version.

- [ ] **Step 4: Ledger the result and the verdict against the spec's success criterion**

Append: the rows-won count vs GLONET (`build_scoreboard_data.json` → compare `v3seaft` vs `glonet` per metric at leads 1/5/10, majority rule, same script as the ft0808era readout), the medal standings, and whether `temp_50m`, `temp_100m`, `mld` are won. Commit the ledger.

---

## Self-review

- **Spec coverage:** §3 trainable scope → Task 2 (regex + test) and Task 4 (log acceptance). §4 loss/weights → Task 1 (mapping, floor) and Task 2 (mse only). §5 data/schedule/validation → Task 2. §6 run mechanics (istep, first-link-only config, pure resume) → Task 2 (`istep: 0`), Task 4 (link 1 with config), Task 5 Step 1 (link 2 pure resume). §7 selection → Task 3 (rule), Task 5 (per-epoch smokes, decision, hand-off). §8 verification → Task 4. §10 retry clause → Task 5 Step 4. §9 out of scope → nothing planned for them.
- **Placeholders:** none; every command and file is written out. `<RUN_ID>`, `<LINK1>`, `<BEST_N>` are runtime values captured in `seaft_run.env`, not placeholders.
- **Consistency:** tags `seaft<N>` ↔ run ids `obench_seaft<N>_<week>` (Task 5) match `v2_model_smoke.sh`'s `obench_<tag>_<week>` convention; baseline run ids `obench_v3e45nowera5_*` / `obench_ft0808nowera5_*` exist; `seaft_score.py`'s `score(path, cand, base)` / `overall(res)` names match the tests; weight values in Task 1's test equal `seaft_channel_weights.json` with the uo/vo floor applied.
