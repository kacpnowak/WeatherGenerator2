# From-scratch S2RoPE emulator on 46 levels: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository ready to train one FESOM ocean emulator from scratch on Levante with Wigner-D RoPE in every attention module and all 46 depth levels, and prove it with a GPU smoke test before stage 1 is submitted.

**Architecture:** Four small code fixes (learning-rate decay, atomic config write, validation stride, persistent loader workers), one stream directory and one config family for the run, a launcher with a checkpoint pruner, then a smoke test whose numbers go to the user. No model code changes: the RoPE implementation was reviewed and stays as it is.

**Tech Stack:** Python 3.12, PyTorch (FSDP, flash attention), OmegaConf configs, pytest, SLURM on Levante (2 nodes x 4 A100 80 GB).

**Spec:** `docs/superpowers/specs/2026-11-17-rope-46-levels-from-scratch-design.md`

## Global Constraints

- Run tests with `PYTHONPATH=src:packages/common/src:packages/readers_extra/src:packages/evaluate/src .venv/bin/python -m pytest -q -p no:cacheprovider <files>`. Imports take 1-2 minutes on the login node.
- Baseline that must not get worse: 13 failures in `tests/test_encoder_teacher.py`, and `tests/test_config.py` fails at collection (`pytest.fixure` typo). Both predate this work.
- Lint: `.venv/bin/ruff check <files>` and `.venv/bin/ruff format --check <files>` on files you touch. Never run `ruff format` on a file you did not write: it rewrites whole files.
- One-letter file names (`u.yml`, `v.yml`) are ignored by `.gitignore`: add them with `git add -f`.
- Commit after each task, locally, on `kacpnowak/develop/forcings`. Do not push (the session has no credentials; the user pushes). Do not merge `upstream/develop`.
- Do not edit `sbatch_fesom_vars_train.sh`, the 31-level `config/streams/fesom_vars/`, or anything under `../WeatherGenerator-private/`.
- Nothing is submitted to the `gpu` partition except the smoke test of Task 8. Stage 1 is submitted only after the user has seen the smoke results.
- Job states cannot be trusted (failed runs exit 0): judge every job by its log.
- Scratch space is on `/work` (for example `smoke46/`), never `/tmp`.
- Mesh on Levante: `/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc`. Mesh on JUPITER: `/e/data1/climateai/hclimrep/data/ifs-fesom/fesom.mesh.diag.nc`.
- The 46 depths, in order: 2, 7, 15, 25, 35, 45, 55, 65, 75, 85, 95, 107, 125, 147, 175, 210, 255, 310, 375, 450, 535, 630, 735, 850, 975, 1110, 1255, 1415, 1600, 1810, 2035, 2275, 2525, 2775, 3025, 3275, 3525, 3775, 4025, 4275, 4525, 4775, 5025, 5275, 5525, 5825.

## Review Focus

1. **A fork whose parent has not finished.** Expected: the launcher refuses and exits non-zero; it never starts a stage from the wrong weights. Pinned in Task 7.
2. **A resume in the middle of a mini-epoch with persistent workers.** Expected: the remaining batches are those of the uninterrupted run, and the next mini-epoch is the right one. Pinned in Task 4.
3. **A validation stride that leaves fewer start dates than requested samples.** Expected: a clear error at start-up, not a silently smaller or repeated validation set. Pinned in Task 3.
4. **The pruner running while a job writes a checkpoint.** Expected: it never touches `_latest`, temporary files, the two newest mini-epoch checkpoints or the final one. Pinned in Task 7.
5. **A loader with fewer samples than workers and persistent workers on** (validation: 64 samples over 8 ranks and 8 workers). Expected: every sample is delivered exactly once per mini-epoch. Pinned in Task 4.

---

### Task 1: Learning-rate decay ends at the configured value

**Files:**
- Modify: `src/weathergen/train/lr_scheduler.py:112`
- Test: `tests/test_lr_scheduler_final_decay.py` (new)

**Interfaces:**
- Consumes: `LearningRateScheduler(optimizer, batch_size, world_size, istep, lr_steps, lr_cfg)`, `.step()`; the learning rate used by a step is `optimizer.param_groups[0]["lr"]` read before `.step()`.
- Produces: nothing new. Behaviour: with `policy_warmup: cosine` the cosine decay ends at `lr_final_decay * kappa`.

- [ ] **Step 1: Write the failing test**

```python
def make(lr_max, lr_steps, istep=0):   # world size 8, batch 1, the recipe's schedule
    cfg = OmegaConf.create(dict(lr_start=1e-6, lr_max=lr_max, lr_final_decay=2e-6, lr_final=0.0,
        num_steps_warmup=256, num_steps_cooldown=512, policy_warmup="cosine",
        policy_decay="cosine", policy_cooldown="linear", parallel_scaling_policy="sqrt"))
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(3))], lr=1e-6)
    return opt, LearningRateScheduler(opt, 1, 8, istep, lr_steps, cfg)

def lrs(lr_max, lr_steps):
    opt, sch = make(lr_max, lr_steps); out = []
    for _ in range(lr_steps):
        out.append(opt.param_groups[0]["lr"]); sch.step()
    return out

@pytest.mark.parametrize("lr_max,lr_steps", [(5e-5, 32768), (1e-5, 3072), (1e-5, 2560)])
def test_decay_ends_at_lr_final_decay(lr_max, lr_steps):
    v = lrs(lr_max, lr_steps); end_decay = lr_steps - 512
    assert max(v) == pytest.approx(lr_max * 8**0.5, rel=1e-3)
    assert v[end_decay - 1] == pytest.approx(2e-6 * 8**0.5, rel=2e-2)   # 5.66e-6, was 1.77e-7
    assert all(a >= b - 1e-15 for a, b in zip(v[256:], v[257:]))        # never rises after the peak
    assert v[-1] < 2e-6 * 8**0.5 / 100                                  # cool-down reaches ~0

def test_resume_matches_continuous_schedule():
    v = lrs(1e-5, 2560)
    for k in (65, 256, 1281, 2048, 2300):
        opt, _ = make(1e-5, 2560, istep=k)
        assert opt.param_groups[0]["lr"] == pytest.approx(v[k], rel=1e-6)
```

- [ ] **Step 2: Run it and see `test_decay_ends_at_lr_final_decay` fail** (value 1.77e-7 instead of 5.66e-6).
- [ ] **Step 3: Fix the factor.** In the `policy_warmup == "cosine"` branch pass `final_div_factor=lr_cfg.lr_start / lr_final_decay_scaled` (OneCycleLR's lowest rate is `initial_lr / final_div_factor`, and `initial_lr` is `lr_start`). If the cool-down start then no longer joins the end of the decay, make it start from the decay's last value.
- [ ] **Step 4: Run the new tests and `tests/test_train_state.py tests/test_resume_position.py`.** All pass.
- [ ] **Step 5: Commit** (`Trainer: cosine decay ends at lr_final_decay (final_div_factor was inverted)`).

---

### Task 2: Atomic run-config write

**Files:**
- Modify: `packages/common/src/weathergen/common/config.py` (`save`, lines 198-209)
- Test: `tests/test_config_save_atomic.py` (new)

**Interfaces:**
- Consumes: `config.save(config: Config, mini_epoch: int | None)`, `_get_model_config_file_write_name`.
- Produces: same signature. The JSON is written to `<name>.tmp` in the same directory and moved over the target with `os.replace`.

- [ ] **Step 1: Write the failing tests.** `test_save_leaves_no_partial_file`: patch `json.dumps` (or the file write) to raise midway on the second `save` of the same run; the existing file must still parse and equal the first save. `test_save_writes_readable_json`: after a normal save the file parses, and no `*.tmp` file is left in the directory.
- [ ] **Step 2: Run, see the first fail.**
- [ ] **Step 3: Implement** the temporary file and `os.replace`; remove the temporary file if the write raises.
- [ ] **Step 4: Run the tests and `tests/test_train_state.py`.** Pass.
- [ ] **Step 5: Commit** (`Config: write the run config atomically`).

---

### Task 3: Validation dates spread over the year (`sample_stride`)

**Files:**
- Modify: `src/weathergen/datasets/multi_stream_data_sampler.py` (`__init__` near line 377, `check_samples` near 410, `reset` near 713)
- Modify: `config/default_config.yml` (document `sample_stride: 1` under `validation_config`)
- Test: `tests/test_sample_stride.py` (new)

**Interfaces:**
- Produces: `stride_perms(perms: np.ndarray, stride: int, shuffle: bool) -> np.ndarray` (module level, pure): returns `perms[::stride]`; raises `ValueError` if `stride < 1`, or if `stride > 1 and shuffle`. Config key `sample_stride` (int, default 1) read from the stage config (`mode_cfg.get("sample_stride", 1)`), stored as `self.sample_stride`.
- Behaviour: `reset()` applies `stride_perms` to the base permutation before any repeat or shuffle. `check_samples` counts the strided dates as available and raises `ValueError` naming `sample_stride` when `samples_per_mini_epoch` exceeds them (instead of reducing `samples_per_mini_epoch`), but only when `sample_stride > 1`; stride 1 keeps today's behaviour exactly.

- [ ] **Step 1: Write the failing tests**

```python
def test_stride_takes_every_nth():
    assert stride_perms(np.arange(350), 5, False).tolist() == list(range(0, 350, 5))   # 70 dates
def test_stride_one_is_identity():
    p = np.arange(17); assert stride_perms(p, 1, True) is p
def test_stride_rejects_shuffle_and_bad_values():
    with pytest.raises(ValueError): stride_perms(np.arange(9), 2, True)
    with pytest.raises(ValueError): stride_perms(np.arange(9), 0, False)
```

Plus, in the style of `tests/test_resume_data_order.py` (recording stand-ins, skipped when the Levante mesh file is absent, config `config/config_fvars46.yml` from Task 6 or, until it exists, `config/config_fesom_vars.yml` with `streams_directory=./smoke46/streams/`): a validation sampler with `samples_per_mini_epoch: 64`, `sample_stride: 5`, `shuffle: False` yields the start indices 0, 5, ..., 315 in two consecutive mini-epochs; with `sample_stride: 6` and 64 samples it raises `ValueError` (Review Focus 3).

- [ ] **Step 2: Run, see them fail** (`stride_perms` not defined).
- [ ] **Step 3: Implement** as described under Interfaces.
- [ ] **Step 4: Run the new tests and `tests/test_resume_skip.py tests/test_resume_position.py tests/test_prefix_steps.py`.** Pass.
- [ ] **Step 5: Commit** (`Sampler: sample_stride thins unshuffled start dates`).

---

### Task 4: Persistent loader workers with an unchanged data order

**Files:**
- Modify: `src/weathergen/datasets/multi_stream_data_sampler.py` (`__init__` near 400-402, `worker_workset` near 1709-1721, `__iter__` near 1730)
- Modify: `src/weathergen/train/trainer.py:684-693` (training and validation loaders)
- Modify: `config/default_config.yml` (`data_loading.persistent_workers: False`, documented)
- Test: `tests/test_persistent_workers_order.py` (new)

**Interfaces:**
- Consumes: `MultiStreamDataSampler.advance()`, `.set_resume_position(mini_epoch, steps_done)`, `.skip_steps`, `.mini_epoch`, `.data_loader_rng_seed`; the recording harness of `tests/test_resume_data_order.py` (`_RecordingBatch`, stand-ins for `_get_batch` and `_get_prefix_batch`).
- Produces: `data_loading.persistent_workers` (bool, default False) honoured by the training and validation loaders: `persistent_workers = num_workers > 0 and cf.data_loading.get("persistent_workers", False)`.

Background the implementer needs: today every `iter(loader)` forks fresh workers. Each worker is a copy of the main-process sampler at that moment, so it sees the `mini_epoch` and `skip_steps` that `advance()` / `set_resume_position()` set on the main copy, and `worker_workset` multiplies `self.data_loader_rng_seed` in place. A persistent worker keeps its copy from the first fork, so it must do both itself:

1. Keep the unmodified seed (`self._base_rng_seed`, set where `data_loader_rng_seed` is first assigned) and compute the per-iteration seed from it instead of multiplying in place.
2. Count its own iterations: in a worker process (`torch.utils.data.get_worker_info() is not None`), every `__iter__` after the first calls `self.advance()` before the work set is computed. The main process never does this (with `num_workers == 0` the trainer's own `advance()` already applies).

Check for other per-mini-epoch state that the main process changes on `self.dataset` / `self.dataset_val` between iterations (`grep -n "self.dataset\w*\.\w* =\|self.dataset\w*\.set_" src/weathergen/train/trainer.py`) and handle it the same way.

- [ ] **Step 1: Write the failing tests** (skipped when the Levante mesh file is absent). With `run_epochs(sampler, epochs, persistent)` building `DataLoader(sampler, batch_size=None, num_workers=2, persistent_workers=persistent)` and calling `sampler.advance()` after each mini-epoch:

```python
def test_three_mini_epochs_match_fresh_workers():
    assert run_epochs(new_sampler(), 3, True) == run_epochs(new_sampler(), 3, False)

def test_mini_epochs_differ_from_each_other():
    a = run_epochs(new_sampler(), 3, True)
    assert a[0] != a[1] and a[1] != a[2]

def test_resume_mid_epoch_then_next_epoch():        # Review Focus 2
    full = run_epochs(new_sampler(), 3, False)
    s = new_sampler(); s.set_resume_position(1, 5)
    got = run_epochs(s, 2, True)
    assert got[0] == full[1][5:] and got[1] == full[2]

def test_fewer_samples_than_workers():              # Review Focus 5
    # 2 samples, 4 workers: each sample exactly once per mini-epoch, same as fresh workers
    ...
```

`new_sampler()` uses `samples_per_mini_epoch: 32` and `forecast.prefix_steps: [0, 90]`, seed 7, rank 0, world size 1, as in `tests/test_resume_data_order.py`, but on a stream directory that resolves on Levante.

- [ ] **Step 2: Run, see `test_three_mini_epochs_match_fresh_workers` fail** (mini-epochs 2 and 3 differ).
- [ ] **Step 3: Implement** the two changes above and the loader option.
- [ ] **Step 4: Run the new tests and `tests/test_resume_skip.py tests/test_resume_position.py tests/test_prefix_steps.py tests/test_sample_stride.py`.** Pass. Make `tests/test_resume_data_order.py` run on Levante too by letting its data root be the first existing of the JUPITER and Levante paths and its stream directory follow; it must pass.
- [ ] **Step 5: Commit** (`Loader: persistent workers keep the data order of fresh workers`).

---

### Task 5: Stream files

**Files:**
- Create: `config/streams/fvars46_levante/{temp,salt,u,v,surface,ifs_forcing}.yml`
- Create: `config/streams/fesom_all_levels/{u,v}.yml`, `config/streams/fesom_all_levels_forced/{u,v}.yml`
- Test: `tests/test_fvars46_streams.py` (new)

**Interfaces:**
- Produces: stream directory `./config/streams/fvars46_levante/` for Task 6.

Content:
- `fvars46_levante/`: `temp.yml`, `salt.yml`, `surface.yml`, `ifs_forcing.yml` from `config/streams/fesom_all_levels_forced/`, and `u.yml`, `v.yml` from `config/streams/fesom_vars/` with `source` and `target` replaced by the 46 depths (`u_2` ... `u_5825`). All with the Levante `mesh_path`. In TEMP, SALT, U and V only: `embed.dim_embed: 512` and `target_readout.num_layers: 4`. Everything else unchanged (stream ids TEMP 2150, SALT 2151, U 2152, V 2153, SURFACE 2154, IFS_ATMO 2139; `max_num_targets: 32768`). A verified 46-level set without the size changes is in `smoke46/streams/`.
- The two JUPITER directories get `u.yml` and `v.yml` with 46 levels, the JUPITER `mesh_path` and no size changes.

- [ ] **Step 1: Write the failing test.** For each of the three directories and each of TEMP, SALT, U, V: `source == target`, exactly the 46 depths of Global Constraints with the right prefix and order; every channel has an entry in `config/streams/fesom/wet_stats_node.json` (TEMP, SALT) or `wet_stats_elem.json` (U, V); no `*_6125`. Stream ids as above and distinct. In `fvars46_levante/`: `embed.dim_embed == 512` and `target_readout.num_layers == 4` for TEMP, SALT, U, V; 256 and 2 for SURFACE; `mesh_path` is the Levante path. Compared key by key with `fesom_vars/` (ignoring `source`, `target`, `mesh_path`, `embed.dim_embed`, `target_readout.num_layers`) nothing else differs.
- [ ] **Step 2: Run, see it fail** (directories missing).
- [ ] **Step 3: Generate the files** with a throwaway script; keep the header comments of the source files and add one line saying what differs.
- [ ] **Step 4: Run the test.** Pass. `git ls-files config/streams/fvars46_levante config/streams/fesom_all_levels config/streams/fesom_all_levels_forced` lists `u.yml` and `v.yml` in all three after `git add -f`.
- [ ] **Step 5: Commit** (`Config: 46-level streams for Levante (fvars46_levante); u and v for the all-levels sets`).

---

### Task 6: Configs of the run

**Files:**
- Create: `config/config_fvars46.yml`, `config/fvars46_stage2.yml`, `config/fvars46_stage3.yml`, `config/fvars46_stage3_mix.yml`
- Test: `tests/test_fvars46_configs.py` (new; model it on `tests/test_fesom_vars_r1_configs.py`)

**Interfaces:**
- Consumes: Task 3 (`sample_stride`), Task 4 (`persistent_workers`), Task 5 (stream directory).
- Produces: stage 1 is `train --config config/config_fvars46.yml`; later stages are `train_continue` with the overlays.

`config/config_fvars46.yml` is a copy of `config/config_fesom_vars.yml` with exactly these differences:

| Key | Value |
|---|---|
| `streams_directory` | `./config/streams/fvars46_levante/` |
| `ae_global_num_heads`, `ae_aggregation_num_heads` | 16 |
| `fe_dropout_rate` | 0.0 |
| `fe_impute_latent_noise_std` | 0.0 |
| `rope_2D` | removed |
| `pe_global_enable` | True |
| `rope_mode` | `spherical_unitary` |
| `rope_modules` | `all` |
| `rope_unitary_max_band` | 10 |
| `data_loading.persistent_workers` | True |
| `training_config.num_mini_epochs` / `samples_per_mini_epoch` | 32 / 8192 |
| `training_config.learning_rate_scheduling.lr_max` | 5e-5 (warm-up 256, cool-down 512, `lr_start` 1e-6, `lr_final_decay` 2e-6 as before) |
| `training_config.forecast.num_steps` | 4 |
| `validation_config.samples_per_mini_epoch` | 64 |
| `validation_config.sample_stride` | 5 |

Overlays (no `streams:` and no `streams_directory` key in any of them):
- `fvars46_stage2.yml`: 12 x 2048 samples, `lr_max` 1e-5, `forecast.num_steps: 15`, `freeze_modules: ""`.
- `fvars46_stage3.yml`: 10 x 2048 samples, `lr_max` 1e-5, `forecast.num_steps: 15`, `forecast.prefix_steps: [0, 90]`, `freeze_modules: ""`. No channel weights (the surface weight is decided after stage 1).
- `fvars46_stage3_mix.yml`: `forecast.prefix_steps: [0, 365]`, `forecast.prefix_zero_fraction: 0.25`; applied after `fvars46_stage3.yml`.

Each file starts with a comment saying which stage it is and how it differs from the recipe (`docs/fesom-vars-training-recipe.md`).

- [ ] **Step 1: Write the failing tests.** Load with `config.load_merge_configs(None, None, None, None, "config/config_fvars46.yml")` and assert every row of the table; `get_rope_unitary_bands(cf) == list(range(1, 11))`; `get_rope_mode(cf, module=m) == "spherical_unitary"` for `aggregation`, `global`, `forecast`; the six streams load with 46 channels for TEMP, SALT, U, V; steps of stage 1 = 32 * 8192 / 8 = 32768. Merging each overlay on top gives the values above; `fvars46_stage3.yml` followed by `fvars46_stage3_mix.yml` gives `prefix_steps == [0, 365]` and `prefix_zero_fraction == 0.25`; no overlay contains `streams` or `streams_directory`. Diffing `config_fvars46.yml` against `config_fesom_vars.yml` as dictionaries shows only the keys of the table.
- [ ] **Step 2: Run, see them fail.**
- [ ] **Step 3: Write the four files.**
- [ ] **Step 4: Run the tests.** Pass.
- [ ] **Step 5: Commit** (`Config: fvars46, the from-scratch RoPE model on 46 levels, and its stage overlays`).

---

### Task 7: Launcher and checkpoint pruner

**Files:**
- Create: `sbatch_fvars46_train.sh` (copy the SLURM header and environment block of `sbatch_fesom_vars_train.sh`; account `ab0995_gpu`)
- Create: `scripts/prune_checkpoints.py`
- Test: `tests/test_prune_checkpoints.py`, `tests/test_fvars46_launcher.py` (new)

**Interfaces:**
- Launcher environment: `WG_RUN_ID` (required), `WG_CONFIGS` (colon-joined, default `config/config_fvars46.yml`, used only for a fresh start), `WG_FORK_FROM="<run> <mini_epoch>"`, `WG_OVERLAY` (colon-joined, required with `WG_FORK_FROM`), `WG_EXTRA_OPTIONS`, `WG_DRY_RUN=1` (print the command that would run and exit 0, no `srun`).
- Launcher decision order:
  1. `models/$RUN/${RUN}_chkpt<num_mini_epochs>.chkpt` exists: print "complete", exit 0.
  2. `models/$RUN/${RUN}_latest.chkpt` exists: `train_continue --from-run-id $RUN -e -1 --reuse-run-id`.
  3. `WG_FORK_FROM` set: the parent's `models/<run>/<run>_chkpt<mini_epoch 5 digits>.chkpt` must exist, else exit 2. Then `train_continue --from-run-id <run> -e <mini_epoch> --run-id $RUN --config <overlays as separate arguments> --options general.istep=0 data_loading.rng_seed=<seconds since epoch> $WG_EXTRA_OPTIONS`.
  4. Otherwise: `train --config <WG_CONFIGS as separate arguments> --run-id $RUN`.
- After `srun`: exit 1 if the run's log `output/fvars46_${RUN}_${SLURM_JOBID}.txt` holds a Python traceback that does not come from the virtualenv start-up noise (`_virtualenv`); then run the pruner for `$RUN`.
- Pruner: `checkpoints_to_delete(present: list[int], num_mini_epochs: int, keep_last: int = 2, keep_every: int = 8, final_only: bool = False) -> list[int]`. Kept: the final index `num_mini_epochs`; indices divisible by `keep_every` (if `keep_every > 0`); and, unless `final_only`, the `keep_last` largest indices below the final one. Command line: `scripts/prune_checkpoints.py <run_id> [--keep-last N] [--keep-every N] [--final-only] [--dry-run]`. For each index it removes `<run>_chkptNNNNN.chkpt`, `<run>_chkptNNNNN_train_state.pt` and `model_<run>_chkptNNNNN.json` in `models/<run>/`. It never touches a name containing `latest` or `tmp`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prune_while_running():
    assert checkpoints_to_delete(list(range(0, 13)), 32) == [1, 2, 3, 4, 5, 6, 7, 9, 10]
def test_prune_keeps_final():
    assert 32 not in checkpoints_to_delete(list(range(0, 33)), 32)
    assert checkpoints_to_delete(list(range(0, 33)), 32, final_only=True) == [
        i for i in range(33) if i % 8 and i != 32]
def test_prune_files(tmp_path):      # Review Focus 4
    # models/<run>/ with chkpt00000..00004 (+ train state, json), _latest files and a *_tmp.chkpt:
    # after pruning with keep_last=2, keep_every=8, num_mini_epochs=10 only indices 1 and 2 are
    # gone, in all three file kinds; _latest and _tmp files are untouched
    ...
```

Launcher tests run the script with `WG_DRY_RUN=1` in a temporary directory holding fake `models/` trees: missing `WG_RUN_ID` exits non-zero; a fresh start prints `train --config config/config_fvars46.yml --run-id r`; an existing `_latest` prints the in-place `train_continue`; a completed run prints "complete"; a fork with the parent checkpoint present prints `--from-run-id p -e 32 --run-id r`, both overlays as separate arguments, `general.istep=0` and a `data_loading.rng_seed=`; a fork whose parent checkpoint is missing exits 2 and prints no command (Review Focus 1); a fork without `WG_OVERLAY` exits non-zero.

- [ ] **Step 2: Run, see them fail.**
- [ ] **Step 3: Write the pruner and the launcher.** `bash -n sbatch_fvars46_train.sh` passes.
- [ ] **Step 4: Run the tests.** Pass. Check with `load_merge_configs` that `data_loading.rng_seed=<n>` given as an option to a continued run replaces the saved seed; if it does not, say so and stop (the fork semantics depend on it).
- [ ] **Step 5: Commit** (`Launcher for the fvars46 runs and a checkpoint pruner`).

---

### Task 8: Smoke test on GPUs

**Files:**
- Modify: `smoke46/sbatch_smoke46.sh`, `smoke46/smoke_stage1.yml`, `smoke46/smoke_stage3.yml` (scratch, not committed)

**Interfaces:**
- Consumes: Tasks 1-7. Before submitting, cancel the stale smoke job 29060247 (it uses the pre-change configs) if it is still pending.

The job (2 nodes, `gpu` partition, account `ab0995_gpu`, 1 h 45 min) runs four steps with `config/config_fvars46.yml` shrunk by options (`training_config.num_mini_epochs=2`, `samples_per_mini_epoch=64`, `validation_config.samples_per_mini_epoch=16`, `train_logging.terminal=1`, `train_logging.checkpoint=4`):
- A: train from scratch; cancel the step once mini-epoch 1 has logged three steps.
- B: resume in place.
- C: fork into the stage-3 settings (`fvars46_stage3.yml`, `fvars46_stage3_mix.yml`, 1 x 48 samples, `prefix_steps=[0,8]`), fresh schedule.
- D: 6-day inference rollout in chunks of 3 on one GPU, all five streams, `max_num_targets=-1`.

- [ ] **Step 1: Update the smoke files and submit.**
- [ ] **Step 2: Judge from the logs.** Pass means all of: no traceback in A (before the kill), B, C, D; finite training and validation losses; B logs the resume step and the restored optimizer and dynamic-loss state, and its first step number follows A's last `_latest`; mini-epoch 1 of B starts without the loader start-up pause (persistent workers); peak GPU memory in C below 75 GB; D's store holds TEMP, SALT, U, V with 46 channels and SURFACE with 3, finite at wet points. If C exceeds 75 GB: set `target_readout.num_layers` to 3 (then 2) in `fvars46_levante/`, rerun, and say so.
- [ ] **Step 3: Record** seconds per step in A and C, loader start time at the first and second mini-epoch, validation time, peak memory. Compare with the 31-level model (10 s per step in stage 1, 38 s in stage 3). More than 30 % slower: stop and ask the user.
- [ ] **Step 4: Delete the smoke runs** (`models/`, `results/` of the smoke run ids) and report the numbers and an updated cost estimate to the user. Do not submit stage 1 before the user has answered.

---

### Task 9: Run book (after the user's go)

**Files:**
- Create: `docs/fvars46-runbook.md`

Content, as commands and checks, no prose beyond what is needed:
- Stage 1: ten chained jobs, `WG_RUN_ID=rw46s1 sbatch --parsable -A ab0995_gpu --job-name=rw46s1 sbatch_fvars46_train.sh`, each next one with `--dependency=afterany:<previous>`.
- After the first job: the dynamic loss weights of the last logged training record (`results/rw46s1/rw46s1_train_metrics.json`, keys `LossPhysical.<stream>.mse_ema_weight.*`), step time, training loss against `fvars001` at the same step; report to the user.
- After stage 1: per-channel validation losses on the 31 shared levels against `fvars001` (stream means TEMP 0.00095, SALT 0.00202, SURFACE 0.00172, U 0.0452, V 0.0650); no-go if any stream is more than 15 % worse.
- Stage 2: `WG_RUN_ID=rw46s2 WG_FORK_FROM="rw46s1 32" WG_OVERLAY=config/fvars46_stage2.yml`, four links.
- Stage 3: `rw46c1` from `"rw46s2 12"` with `config/fvars46_stage3.yml`; `rw46c2` from `"rw46c1 10"` with the same overlay; `rw46c3` from `"rw46c2 10"` with `config/fvars46_stage3.yml:config/fvars46_stage3_mix.yml`; four links each.
- When a stage's successor has trained its first mini-epoch: `scripts/prune_checkpoints.py <finished run> --final-only`.

- [ ] **Step 1: Write the run book and commit it.**
- [ ] **Step 2: Submit stage 1** only on the user's word.
