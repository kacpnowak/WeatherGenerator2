# Long-rollout round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix training resume so chained jobs train on the same data as one uninterrupted job, add run D's mean-error loss, then train four stage-3 variants of fvars on JUPITER and score 9-year rollouts of them and of fvars003.

**Architecture:** The resume fix is split between the data sampler (start a mini-epoch at a given step without loading the skipped samples) and the trainer (compute the resume position, save and restore optimizer and dynamic-loss state in a sidecar file next to each model checkpoint). Run D is one new loss function selected by name in `loss_fcts`. Runs are config overlays on `config/fesom_vars_stage3.yml`; job scripts live in the private repo; scoring reuses `paper_eval/` from the Levante commits.

**Tech Stack:** Python 3.12, PyTorch 2.9 (FSDP2, `torch.distributed.checkpoint.state_dict`), uv, pytest, Slurm on JUPITER (GH200), xarray/netCDF4 for scoring.

**Spec:** `docs/superpowers/specs/2026-10-26-long-rollout-round1-design.md`

## Global Constraints

- Branch base `kacpnowak/develop/forcings`; never rebase onto develop.
- New branch for this work: `kacpnowak/develop/long-rollout-r1`, created from the local base and published with `git push -u origin kacpnowak/develop/long-rollout-r1`; verify with `git branch -vv` that it tracks `origin/kacpnowak/develop/long-rollout-r1`.
- No upstream issue or PR.
- New runs and runs without a resume (`general.istep == 0`) must behave exactly as before every change in this plan.
- Run ids: `fvdrop01` (A), `fvspin01` (B), `fvctrl01` (C), `fvbias01` (D).
- All four runs start with `train_continue --from-run-id fvars002 -e -1 --run-id <id> --config config/fesom_vars_stage3.yml config/fesom_vars_r1_<x>.yml --options general.istep=0`, i.e. from `fvars002_latest.chkpt` in `/e/scratch/weatherai/shared_work/models/fvars002/`.
- Data seed `data_loading.rng_seed` stays the inherited 1792056342 for all runs.
- JUPITER: 2 nodes x 4 GH200 (world size 8), account `e-ext-2025e01-128`, partition `booster`, 12 h job limit. Launch from the repo root.
- Data `/e/data1/climateai/hclimrep/data/ifs-fesom`, mesh `/e/data1/climateai/hclimrep/data/ifs-fesom/fesom.mesh.diag.nc`.
- 9-year rollout: start 2201-01-01, 3,285 daily steps (last step 2209-12-30), `test_config.end_date=2210-01-02T00:00`; year-end leads 365, 730, 1095, 1461, 1826, 2191, 2556, 2922, final 3285.
- fvars003 rollout uses `fvars003_chkpt00010.chkpt` (`-e 10`), not `_latest`.
- Climatology reference 2180-2209.
- Heavy work (tests reading real data, scoring, smoke runs) goes through Slurm, never on a login node.
- `./scripts/actions.sh lint` passes before each commit that touches Python.

## Review Focus

- Mid-epoch resume with a different world size than the run started with: the skip arithmetic no longer maps steps to samples. Expected: a warning, and the resumed job starts the current mini-epoch from its first step with the aligned counter (no crash). Test in Task 2.
- Resume from a checkpoint that has no train-state sidecar (every existing checkpoint): expected warning, fresh optimizer and dynamic-loss state, training continues. Test in Task 3.
- Starting a new run from another run's checkpoint (`general.istep=0`) while that checkpoint has a sidecar: expected to ignore it (fvars003 started with a fresh optimizer and fresh dynamic weights). Test in Task 3.
- Resume exactly at a mini-epoch boundary and resume after the last mini-epoch: expected no skip / no training and a clean exit. Test in Task 2.
- Run D's term on targets with NaN (dry points) and on a channel whose targets are all NaN: expected NaNs ignored, an all-NaN channel contributes 0 and never NaN. Test in Task 4.

---

### Task 0: Branch, Levante commits and JUPITER paths

**Files:**
- Modify (by fetch, not by hand): `config/config_fesom_vars.yml`, `config/fesom_vars_stage{1,2,3}.yml`, `tests/test_fesom_vars_training_config.py`, `paper_eval/**`
- Commit: `docs/superpowers/specs/2026-10-26-long-rollout-round1-design.md`, `docs/superpowers/plans/2026-10-26-long-rollout-round1.md`, the JUPITER path edits in `config/streams/fesom_vars/{temp,salt,u,v,surface}.yml`, `compute_wet_stats.py`, `tests/test_fesom_{mesh,reader_mesh,vars_configs,wet_stats}.py`, `tests/test_forcing_{configs,fesom_data}.py`

**Interfaces:**
- Produces: branch `kacpnowak/develop/long-rollout-r1` containing Levante commits `fbf57a86`, `e9b4b90f`, `d762f3dd`, `1fa55d17`, `1f8433ed` with their original hashes.

- [ ] **Step 1: Bundle the Levante commits.** On Levante (`ssh -o BatchMode=yes levante`), with git at `/sw/spack-levante/git-2.31.1-25ve7r/bin/git`, in `/work/ab0995/a270225/WeatherGenerator2`: `git bundle create /work/ab0995/a270225/forcings-r1.bundle f0a4f7f6..kacpnowak/develop/forcings`. Copy it with scp into the scratchpad.
- [ ] **Step 2: Discard the working-tree copies of the Levante files** (they are byte-identical to the commits): `git checkout -- config/config_fesom_vars.yml tests/test_fesom_vars_training_config.py` and `rm config/fesom_vars_stage{1,2,3}.yml`. Keep every other working-tree change.
- [ ] **Step 3: Fast-forward the base.** `git fetch <bundle> kacpnowak/develop/forcings:refs/remotes/levante/forcings`, then `git merge --ff-only levante/forcings` on `kacpnowak/develop/forcings`. Expected: `git log --oneline -5` shows `1f8433ed` on top and the hashes match Levante.
- [ ] **Step 4: Create and publish the branch** as in Global Constraints; check `git branch -vv`.
- [ ] **Step 5: Commit** the spec and plan (one commit), then the JUPITER path edits (one commit, message says they point the FESOM stream files and data tests at the JUPITER copy, and that the branch is experiment-only). Delete the Levante bundle file afterwards.

### Task 1: Sampler starts a mini-epoch at a given step

**Files:**
- Modify: `src/weathergen/datasets/multi_stream_data_sampler.py` (`__init__` near 188, `advance` near 634, `__iter__` near 1420-1500, `worker_workset` near 1503)
- Test: `tests/test_resume_skip.py` (no data), `tests/test_resume_data_order.py` (real data, skips when `/e/data1/climateai/hclimrep/data/ifs-fesom/ocean_node` is absent)

**Interfaces:**
- Produces:
  - `MultiStreamDataSampler.set_resume_position(mini_epoch: int, steps_done: int) -> None`: sets `self.mini_epoch = mini_epoch` and `self.skip_steps = steps_done`.
  - `advance()` also resets `self.skip_steps = 0`.
  - module function `worker_skip_counts(steps_to_skip: int, worker_lengths: list[int]) -> list[int]`: how many of its own batches each DataLoader worker must skip so that the first `steps_to_skip` batches the trainer receives are skipped. Simulates the DataLoader's round-robin over workers that still have batches.
  - module function `worker_ranges(rank: int, len_per_rank: int, num_workers: int) -> list[tuple[int, int]]`: the `[iter_start, iter_end)` of every worker, the same arithmetic `worker_workset` uses today (last worker takes the remainder); `worker_workset` calls it.

- [ ] **Step 1: Write the failing unit tests** in `tests/test_resume_skip.py`:

```python
def test_round_robin_equal_workers():
    assert worker_skip_counts(5, [4, 4]) == [3, 2]
def test_skip_nothing():
    assert worker_skip_counts(0, [32] * 8) == [0] * 8
def test_skip_past_short_worker():
    # worker 0 has 2 batches, worker 1 has 5: order w0 w1 w0 w1 w1 w1 w1
    assert worker_skip_counts(5, [2, 5]) == [2, 3]
def test_skip_whole_epoch():
    assert worker_skip_counts(64, [32, 32]) == [32, 32]
def test_worker_ranges_match_workset_arithmetic():
    assert worker_ranges(rank=1, len_per_rank=10, num_workers=3) == [(10, 13), (13, 16), (16, 20)]
def test_advance_resets_skip():
    # sampler built with __new__ and the two attributes set by hand
    s = MultiStreamDataSampler.__new__(MultiStreamDataSampler); s.mini_epoch = 0
    s.set_resume_position(3, 17); assert (s.mini_epoch, s.skip_steps) == (3, 17)
    s.advance(); assert (s.mini_epoch, s.skip_steps) == (4, 0)
```

- [ ] **Step 2: Run** `uv run --extra cpu pytest tests/test_resume_skip.py -q`. Expected: FAIL (names not defined).
- [ ] **Step 3: Implement** the three interfaces. In `__iter__`, compute this worker's skip count from `worker_skip_counts(self.skip_steps, [batches per worker from worker_ranges(...)])` (batches per worker = `ceil((end - start) / self.batch_size)`; single-process case: the worker is the only one). For the first `k` iterations of the batch loop, keep `i`, `num_forecast_steps` and the `draw_prefix_steps(...)` call position unchanged, advance `idx_raw` by one, and `continue` without building or yielding a batch. Initialise `self.skip_steps = 0` in `__init__`. Log once per worker how many batches it skipped.
- [ ] **Step 4: Run** the unit tests. Expected: PASS.
- [ ] **Step 5: Write the real-data test** `tests/test_resume_data_order.py`: build `MultiStreamDataSampler` from `config.load_merge_configs(None, None, None, None, "config/config_fesom_vars.yml", {})` with `training_config.samples_per_mini_epoch=32` and `forecast.prefix_steps=[0, 90]`; monkeypatch `_get_batch` and `_get_prefix_batch` on the instance to return a stub whose `sources_empty()`, `is_nan()` and `targets_empty()` are False and that records `(idx, num_forecast_steps, prefix_steps)`. Iterate through `torch.utils.data.DataLoader(sampler, batch_size=None, num_workers=2)`.

```python
def test_resumed_mid_epoch_sees_the_uninterrupted_sequence():
    full = run_epochs(new_sampler(), epochs=3)            # list of 3 lists of records
    s = new_sampler(); s.set_resume_position(1, 5)
    resumed = run_epochs(s, epochs=2)
    assert resumed[0] == full[1][5:]
    assert resumed[1] == full[2]
def test_without_resume_position_nothing_changes():
    assert run_epochs(new_sampler(), 1) == run_epochs(new_sampler(), 1)
```

- [ ] **Step 6: Run** it on a booster node: `sbatch --account=e-ext-2025e01-128 --partition=booster --nodes=1 --gres=gpu:1 --time=00:30:00 --wrap ".venv/bin/python -m pytest -rs -q tests/test_resume_data_order.py tests/test_resume_skip.py tests/test_prefix_steps.py"`. Expected: all pass, 0 skipped.
- [ ] **Step 7: Lint and commit** (`src/weathergen/datasets/multi_stream_data_sampler.py`, both tests).

### Task 2: Trainer computes the resume position and saves `_latest` at the right step

**Files:**
- Modify: `src/weathergen/train/trainer.py` (`run()` near 642-655, `train()` near 715-870, `save_model` near 1021)
- Test: `tests/test_resume_position.py`

**Interfaces:**
- Consumes: `MultiStreamDataSampler.set_resume_position` (Task 1).
- Produces: module function `resume_position(istep: int, steps_per_mini_epoch: int, num_mini_epochs: int, world_size_changed: bool = False) -> tuple[int, int]` returning `(mini_epoch_base, steps_done)`, with `steps_done` the optimizer steps already done in `mini_epoch_base`; `(num_mini_epochs, 0)` once training is complete; `steps_done` forced to 0 when `world_size_changed`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_fresh_run(): assert resume_position(0, 256, 10) == (0, 0)
def test_mid_epoch(): assert resume_position(960, 256, 10) == (3, 192)
def test_epoch_boundary(): assert resume_position(768, 256, 10) == (3, 0)
def test_complete(): assert resume_position(2560, 256, 10) == (10, 0)
def test_beyond_complete(): assert resume_position(2688, 256, 10) == (10, 0)
def test_world_size_changed_restarts_the_epoch():
    assert resume_position(960, 256, 10, world_size_changed=True) == (3, 0)
```

- [ ] **Step 2: Run** `uv run --extra cpu pytest tests/test_resume_position.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `resume_position` and use it in `run()` in place of the current `mini_epoch_base` block, with `steps_per_mini_epoch = len(self.dataset) // self.batch_size_per_gpu` and `world_size_changed = cf.world_size != self.world_size_original` (log a warning when it is True and `istep > 0`). Then call `self.dataset.set_resume_position(mini_epoch_base, steps_done)` (the #2954 change plus the skip). Log the resume position on root.
- [ ] **Step 4: Fix the `_latest` step count.** Move the `_latest` save in `train()` to after `self.cf.general.istep += 1`, so the saved `istep` equals the number of completed optimizer steps. In `run()`, after `self.save_model(mini_epoch)` at the end of each mini-epoch, also call `self.save_model(-1)`, so `_latest` is never older than the last completed mini-epoch.
- [ ] **Step 5: Run** the tests plus `tests/test_train_step_frees_batch.py tests/test_prefix_logging.py`. Expected: PASS.
- [ ] **Step 6: Lint and commit.**

### Task 3: Optimizer and dynamic-loss state survive a resume

**Files:**
- Create: `src/weathergen/train/train_state.py`
- Modify: `src/weathergen/train/loss_modules/loss_module_physical.py` (`DynamicLossEMA`), `src/weathergen/train/trainer.py` (`save_model`, `run()` after the optimizer and loss calculators are built)
- Test: `tests/test_train_state.py`

**Interfaces:**
- Produces:
  - `DynamicLossEMA.state_dict() -> dict[str, torch.Tensor]` (CPU copies of `channel_weights_ema`; `{}` when disabled) and `DynamicLossEMA.load_state_dict(state: dict[str, torch.Tensor]) -> None` (moves to its device; ignores unknown streams; keeps ones for missing ones).
  - in `train_state.py`: `train_state_path(model_path: Path, run_id: str, mini_epoch: int) -> Path` giving `<run_id>_latest_train_state.pt` for `-1`, else `<run_id>_chkpt{mini_epoch:05d}_train_state.pt`; `save_train_state(path: Path, optimizer_state: dict, dynamic_loss: dict, istep: int) -> None` (rank 0 only, write to a `_tmp` file then rename, as `save_model` does); `load_train_state(path: Path) -> dict | None` (None and a warning when the file is missing); `should_restore_train_state(istep: int) -> bool` (True iff `istep > 0`).
- The optimizer state is gathered with `get_optimizer_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))` and restored with `set_optimizer_state_dict(model, optimizer, optim_state_dict=..., options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))` from `torch.distributed.checkpoint.state_dict`. The dynamic-loss state is saved from rank 0 and restored on every rank (the other ranks' EMAs become rank 0's; accepted).

- [ ] **Step 1: Write the failing tests** (CPU, no distributed init):

```python
def test_dynamic_loss_round_trip():
    ema = DynamicLossEMA({"window": 4, "L": 20.0}, streams, "cpu"); ema.update("TEMP", torch.tensor([0.5, 2.0]))
    fresh = DynamicLossEMA({"window": 4, "L": 20.0}, streams, "cpu"); fresh.load_state_dict(ema.state_dict())
    assert torch.equal(fresh.channel_weights_ema["TEMP"], ema.channel_weights_ema["TEMP"])
def test_disabled_dynamic_loss_has_empty_state():
    assert DynamicLossEMA(None, streams, "cpu").state_dict() == {}
def test_optimizer_round_trip(tmp_path):
    # tiny nn.Linear, one AdamW step, save_train_state -> load_train_state -> set into a fresh optimizer;
    # exp_avg, exp_avg_sq and step equal
def test_missing_file_returns_none(tmp_path):
    assert load_train_state(tmp_path / "x_latest_train_state.pt") is None
def test_new_run_does_not_restore():
    assert should_restore_train_state(0) is False and should_restore_train_state(1) is True
def test_paths():
    assert train_state_path(Path("m"), "fvctrl01", -1).name == "fvctrl01_latest_train_state.pt"
    assert train_state_path(Path("m"), "fvctrl01", 3).name == "fvctrl01_chkpt00003_train_state.pt"
```

- [ ] **Step 2: Run** `uv run --extra cpu pytest tests/test_train_state.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `train_state.py` and the two `DynamicLossEMA` methods. In `save_model`, after the model file is written, gather the optimizer state on all ranks (collective) and have rank 0 write the sidecar for the same `mini_epoch` with `self.loss_calculator`'s physical loss module's `dynamic_loss_ema.state_dict()` and `self.cf.general.istep`. In `run()`, after the loss calculators exist: if `should_restore_train_state(cf.general.istep)`, load the sidecar of `(run_id_contd, mini_epoch_contd)` from that run's model directory, restore the optimizer and the dynamic-loss state, and log what was restored; on `None` continue as today.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Lint and commit.**

### Task 4: Run D's mean-error loss

**Files:**
- Modify: `src/weathergen/train/loss_modules/loss_functions.py` (new function after `mse`)
- Test: `tests/test_mean_error_loss.py`

**Interfaces:**
- Produces: `mean_error_sq(target: torch.Tensor, pred: torch.Tensor, weights_channels: torch.Tensor | None, weights_points: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]`, selected in config as `loss_fcts.mean_error_sq: {weight: <lambda>}`. Same argument conventions as `lp_loss` (pred `[ens, points, channels]`, ensemble mean taken, NaN targets masked). Per channel: `loss_chs[c] = (sum_i w_i * (pred_ic - target_ic) over valid i / max(1, n_valid_c)) ** 2`; `loss = mean(loss_chs * weights_channels)` (or plain mean when `weights_channels` is None). The existing `LossPhysical` passes it the same channel weights as `mse` and updates the dynamic EMA only for `mse`, so no change there.

- [ ] **Step 1: Write the failing tests:**

```python
def test_uniform_offset_is_penalised():   # pred = target + 0.3 everywhere -> loss_chs == 0.09
def test_zero_mean_error_is_free():       # errors +1 and -1 at equal weight -> loss_chs == 0
def test_nan_targets_are_ignored():       # NaN rows excluded from sum and count
def test_all_nan_channel_contributes_zero():  # loss_chs[c] == 0, loss finite
def test_channel_weights_scale_the_mean():    # weights [1, 3] -> loss == mean([a, 3b])
def test_selectable_by_name():
    assert getattr(loss_functions, "mean_error_sq") is mean_error_sq
```

- [ ] **Step 2: Run** `uv run --extra cpu pytest tests/test_mean_error_loss.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `mean_error_sq`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Lint and commit.**

### Task 5: JUPITER job scripts and a split-run smoke test

**Files:**
- Create (private repo, not committed here): `../WeatherGenerator-private/hpc/jupiter/sbatch_fvars_r1_train.sh`, `../WeatherGenerator-private/hpc/jupiter/sbatch_fvars_rollout.sh`

**Interfaces:**
- `sbatch_fvars_r1_train.sh` takes `WG_RUN_ID`, optional `WG_OVERLAY` (path of the run overlay, appended after `config/fesom_vars_stage3.yml`) and optional `WG_EXTRA_OPTIONS` (appended to `--options`) from the environment. If `models/$WG_RUN_ID/${WG_RUN_ID}_latest.chkpt` exists under `/e/scratch/weatherai/shared_work`, it runs `.venv/bin/train_continue --from-run-id $WG_RUN_ID -e -1 --reuse-run-id`; otherwise the start command from Global Constraints. Module setup and environment as `hpc/jupiter/weathergen_slurm.sh` (Stages/2025, GCC 13.3.0, the UCX and NCCL exports, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `MASTER_ADDR`); `#SBATCH --nodes=2 --ntasks-per-node=4 --cpus-per-task=72 --gres=gpu:4 --time=12:00:00 --account=e-ext-2025e01-128 --partition=booster`; logs to `logs/fvars-r1-%x.%j.out`; command output to `output/fvars_${WG_RUN_ID}_${SLURM_JOBID}.txt`. Chain with `--dependency=afterany`.
- `sbatch_fvars_rollout.sh FROM_RUN FROM_EPOCH RUN_ID NUM_STEPS START_DATE`: Levante's `sbatch_fesom_forced_rollout.sh` options (`samples_per_mini_epoch=1`, `output.num_samples=1`, `output.streams=[TEMP,SALT,U,V,SURFACE]`, `streams.<S>.max_num_targets=-1`, `forecast.chunk_size=30`, `forecast.resume=true`, `compute_loss=false`), with `test_config.start_date=$START_DATE` and `test_config.end_date=2210-01-02T00:00`; 1 node, 1 GPU, 12 h; resubmitting with the same arguments continues from `results/<run_id>/rollout_state.pt`.

- [ ] **Step 1: Write both scripts.** Get the Levante originals for reference with scp (`sbatch_fesom_vars_train.sh`, `sbatch_fesom_forced_rollout.sh`).
- [ ] **Step 2: Smoke-test the resume end to end.** Run id `r1smoke1`, no `WG_OVERLAY`, `WG_EXTRA_OPTIONS="training_config.num_mini_epochs=2 training_config.samples_per_mini_epoch=512 train_logging.checkpoint=8 validation_config.samples_per_mini_epoch=16"`, 2 nodes (world size 8: it must equal the inherited `world_size_original` of 8 or the skip is disabled by design), `--time=00:25:00`, so jobs are killed mid-epoch; chain four jobs. Expected in a resumed job's output: the logged resume position with `steps_done > 0`, "restored optimizer state", the per-worker skip log; and a final `istep` of exactly `2 * 512 / 8 = 128` in `model_r1smoke1_chkpt00002.json`; losses finite.
- [ ] **Step 3: Smoke-test the rollout script** with `fvars003 10 r1smokero 3 2201-01-01T00:00`. Expected: 3 steps of output under `results/r1smokero`, exit 0. Delete both smoke runs' outputs afterwards.

### Task 6: Run overlays and run D's weight

**Files:**
- Create: `config/fesom_vars_r1_drop.yml`, `config/fesom_vars_r1_spin.yml`, `config/fesom_vars_r1_ctrl.yml`, `config/fesom_vars_r1_bias.yml`
- Test: `tests/test_fesom_vars_r1_configs.py`

**Interfaces:**
- Overlay contents (each with a header comment saying what it changes and why, as the stage overlays do):
  - drop: `fe_dropout_rate: 0.0`
  - spin: `training_config.forecast.prefix_steps: [0, 365]`
  - ctrl: no keys besides the comment
  - bias: `training_config.losses.physical.loss_fcts` = `mse: {}`, `dynamic_loss: {window: 128, L: 20.0}`, `mean_error_sq: {weight: <lambda>}`

- [ ] **Step 1: Write the failing test:** merge `config/config_fesom_vars.yml` + `config/fesom_vars_stage3.yml` with and without each overlay via `config.load_merge_configs`, flatten both, and assert the set of differing keys is exactly `{"fe_dropout_rate"}`, `{"training_config.forecast.prefix_steps"}`, `set()`, `{"training_config.losses.physical.loss_fcts.mean_error_sq.weight"}` respectively.
- [ ] **Step 2: Run** `uv run --extra cpu pytest tests/test_fesom_vars_r1_configs.py -q`. Expected: FAIL.
- [ ] **Step 3: Write the four overlays**, with `weight: 1.0` in bias for now.
- [ ] **Step 4: Probe lambda.** 1-node job, run id `r1probe1`, the bias overlay with `weight: 1.0`, `training_config.num_mini_epochs=1 training_config.samples_per_mini_epoch=80` (20 steps at world size 4). From the logged per-channel training losses of `mse` and `mean_error_sq`, compute `R = sum_c w_c * mean_error_sq_c / sum_c w_c * mse_c` over all streams and steps, with `w_c` the static channel weights (5 for temp_2 and salt_2, else 1; the dynamic weights are still ones this early). Set `weight = round(0.1 / R, 2 significant digits)` in `config/fesom_vars_r1_bias.yml` and note `R` in its header comment. Delete the probe run.
- [ ] **Step 5: Run** the config test. Expected: PASS.
- [ ] **Step 6: Commit** the overlays and the test.

### Task 7: Launch training and the fvars003 rollout

- [ ] **Step 1: Submit the four training chains** (`WG_RUN_ID`/`WG_OVERLAY` per run), three linked jobs each to start with; add links while a chain is short of `num_mini_epochs`. Record job ids in the plan's progress notes.
- [ ] **Step 2: Check the first hour of each run:** `target_channel_weights [5.0, …]` on TEMP and SALT; finite losses; `fvdrop01` logs `fe_dropout_rate: 0.0`; `fvspin01` logs a mean `forecast_prefix_steps` well above 45; `fvbias01` logs a `mean_error_sq` loss.
- [ ] **Step 3: Submit the fvars003 9-year rollout:** `sbatch_fvars_rollout.sh fvars003 10 r9fv003 3285 2201-01-01T00:00`, chained until `done_steps == 3285` in `rollout_state.pt`.
- [ ] **Step 4: When each run ends, verify** `general.istep == 2560` in its final `model_<id>_chkpt00010.json` and a monotone `run_history`.

### Task 8: Scoring on JUPITER

**Files:**
- Create: `paper_eval/configs/r9fv003.yaml`, `paper_eval/configs/r9{drop,spin,ctrl,bias}.yaml`, `paper_eval/r9_table.py`
- Test: `paper_eval/test_r9_table.py`

**Interfaces:**
- Configs: copies of `paper_eval/configs/yr1fv003.yaml` with `run_id`/`results_dir` of the rollout, `mesh_path` and every `truth` entry on the JUPITER data, `init_filter.start: "2201-01-01"`, `showcase.init_time: "2201-01-01"`, `leads.max: 3285`, `leads.map_leads: [365, 730, 1095, 1461, 1826, 2191, 2556, 2922, 3285]`, `output.cache_dir` and `output.fig_dir` under `/e/scratch/hclimrep/nowak2/paper_eval_r1/`.
- `r9_table.py`: `build_table(cache_root: Path, runs: list[str], leads: list[int]) -> str` (Markdown) with, per run and year-end lead, the mean over channels of RMSE_emulator / RMSE_climatology for node and elem streams, and the emulator bias of temp/salt at 107, 450 and 975 m; reads each `scores_*.nc` fully and closes it before the next (opening many lazily segfaults netCDF4 on both machines).

- [ ] **Step 1: Copy the climatology zarrs** `clim_FESOM_{NODE,ELEM}_2180-2209.zarr` and `clim_FESOM_{NODE,ELEM}_extra_2180-2209.zarr` from Levante `paper_eval/cache_oe3i7whz/climatology/` into `/e/scratch/hclimrep/nowak2/paper_eval_r1/climatology/` (scp -r, login node; it is I/O only).
- [ ] **Step 2: Write the test** for `build_table` on two tiny synthetic score files (two runs, two leads, known ratios), asserting the table contains the expected ratio values to 3 decimals. Run: FAIL. Implement. Run: PASS.
- [ ] **Step 3: Write the five configs**; score `r9fv003` once its rollout is complete: `sbatch` a 1-node booster job running `.venv/bin/python -m paper_eval.compute --config paper_eval/configs/r9fv003.yaml` then `paper_eval.selfcheck`. Expected: selfcheck all PASS or SKIP.
- [ ] **Step 4: Commit** configs, `r9_table.py` and its test.

### Task 9: Rollouts of A-D, scores and the comparison

- [ ] **Step 1: Roll out** each finished run: `sbatch_fvars_rollout.sh <run> 10 r9<x> 3285 2201-01-01T00:00` (`r9drop`, `r9spin`, `r9ctrl`, `r9bias`), chained to completion.
- [ ] **Step 2: Score** each as in Task 8 Step 3.
- [ ] **Step 3: Build the table** for `r9ctrl, r9drop, r9spin, r9bias, r9fv003` at all year-end leads, plus the six old one-year runs at lead 363 from the Levante caches as a separate look-back block. State in the table header that the climatology includes the test years and that the only noise estimate is C vs fvars003 (confounded, likely an underestimate).
- [ ] **Step 4: Publish** the table as a short report (Artifact) for the user, with the reading rule from the spec: differences not clearly larger than the C vs fvars003 gap are inconclusive.
