# Long-horizon stability training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the IFS-forced FESOM emulator from its own drifted states (a random unsupervised spin-up before the 15 supervised steps) and judge it with a one-year free-running forced forecast.

**Architecture:** A training-only `forecast.prefix_steps: [min, max]` setting. Per training step the sampler draws a spin-up length P that is the same on every GPU. It builds inputs as today, forcing for all P + 15 steps, and targets only for the last 15 steps, using the step-range builders written for lazy inference. The model's rollout loop runs the steps before the batch's `first_supervised_step` without gradients and without decoding, like the existing pushforward steps. The loss already skips steps without predictions. A one-year inference script, paper_eval configs and a drift-diagnostics script form the acceptance test.

**Tech Stack:** Python 3.12, PyTorch (FSDP2), NumPy, OmegaConf, pytest, xarray/zarr (paper_eval), matplotlib, SLURM on Levante.

**Spec:** `docs/superpowers/specs/2026-10-08-long-horizon-stability-design.md`

## Global Constraints

- Branch `kacpnowak/develop/forcings` is experimental and never merged or pushed.
- Never edit, stage or commit the user's 13 uncommitted files (listed in `.git/forcings-wip-files.txt`). Task 5 is the only exception, and only with the user's explicit permission. Even then the file is never staged or committed.
- Commit only with a pathspec: `git commit -m "..." -- <paths>`. End every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Keep stash `3e9383a7`.
- Never use `uv run`. Use `.venv/bin/python`, `.venv/bin/train_continue` and `.venv/bin/inference`. Prefix Python commands with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`; run ruff as `RAYON_NUM_THREADS=1 .venv/bin/ruff`.
- Config spelling, verbatim from the spec: `training_config.forecast.prefix_steps: [0, 90]` (window run), `[0, 0]` (control), `num_steps: 15`, `policy: fixed`.
- `prefix_steps` applies to the training stage only. Validation and test stages ignore it.
- One-year test: one forecast from 2209-01-01 for 363 daily steps (through 2209-12-30); `chunk_size` 30, `resume: true`, `compute_loss: false`, output streams FESOM_NODE and FESOM_ELEM.
- Never use the IFS archive's `sst` as forcing (it is the ocean model's own SST).
- Delete broken runs (their `models/`, `results/` and `logs/` folders and output files) without asking.
- `paper_eval/` and the helper scripts at the repo root (`sbatch_*.sh`, `compare_*.py`, `diag_*.py`, `test_diag_*.py`) stay untracked: never `git add` them.
- Training jobs: 2 nodes × 4 A100-80GB, 12 h limit. Inference: 1 GPU. paper_eval scoring: `shared` partition.
- `run_continue` can report COMPLETED in SLURM after a crash: always check the output log for `Traceback`.
- Run ids: smoke `smkwin01`; training `frcwin01` (window) and `frcctl01` (control); one-year runs `yr1add04`, `yr1ykwx7`, `yr1win01`, `yr1ctl01`.
- Record every job id and decision in `.superpowers/sdd/2026-10-05-ifs-fesom-forcing/progress.md` (git-ignored ledger).

## Review Focus

- **Validation or test configs inherit `prefix_steps`** from `training_config` (through `get_active_stage_config`). Expected: they still roll out from the initial condition with no prefix. Test in Task 1.
- **Malformed `prefix_steps`** (`[90, 0]`, `[-1, 5]`, `[5]`, `[0, 1.5]`, a bare int). Expected: a `ValueError` naming `prefix_steps` at sampler construction, not a crash deep in a worker. Test in Task 1.
- **Forcing missing inside a prefix.** Expected: the sample is skipped and logged exactly like an eager sample, and the run continues. Test in Task 3.
- **GPUs or loader workers drawing different P for the same training step.** Expected: identical P on all GPUs for the same step, so no GPU waits on another's longer prefix. Tests in Tasks 1 and 3.
- **Start dates late in the training period.** Expected: no sample's prefix plus 15 supervised windows reaches past the training period. Test in Task 3.

---

### Task 1: Prefix-range parsing and the per-step draw

**Files:**
- Modify: `src/weathergen/datasets/multi_stream_data_sampler.py` (imports at lines 20-39; new module-level functions after `forcing_time_window_handler`, which ends near line 135)
- Test: `tests/test_prefix_steps.py` (new)

**Interfaces:**
- Produces: `resolve_prefix_range(forecast_cfg: dict, stage: Stage) -> tuple[int, int]` and `draw_prefix_steps(prefix_range: tuple[int, int], seed: int, mini_epoch: int, position: int) -> int`, both in `weathergen.datasets.multi_stream_data_sampler`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prefix_steps.py`:

```python
"""Training spin-up (forecast.prefix_steps): range parsing and the per-step draw."""

import pytest

from weathergen.datasets.multi_stream_data_sampler import draw_prefix_steps, resolve_prefix_range


def test_training_reads_the_prefix_range():
    assert resolve_prefix_range({"prefix_steps": [0, 90]}, "train") == (0, 90)
    assert resolve_prefix_range({"prefix_steps": [5, 5]}, "train") == (5, 5)


@pytest.mark.parametrize("forecast_cfg", [{}, {"prefix_steps": None}])
def test_no_prefix_steps_means_no_prefix(forecast_cfg):
    assert resolve_prefix_range(forecast_cfg, "train") == (0, 0)


@pytest.mark.parametrize("stage", ["val", "test"])
def test_validation_and_test_ignore_the_inherited_prefix(stage):
    # validation/test configs inherit the training forecast settings
    assert resolve_prefix_range({"prefix_steps": [0, 90]}, stage) == (0, 0)


@pytest.mark.parametrize("bad", [[90, 0], [-1, 5], [5], [0, 1.5], 10, "0,90"])
def test_malformed_prefix_steps_are_rejected(bad):
    with pytest.raises(ValueError, match="prefix_steps"):
        resolve_prefix_range({"prefix_steps": bad}, "train")


def test_draw_depends_only_on_seed_mini_epoch_and_position():
    first = [draw_prefix_steps((0, 90), 7, 2, p) for p in range(50)]
    again = [draw_prefix_steps((0, 90), 7, 2, p) for p in range(50)]
    assert first == again
    assert all(0 <= p <= 90 for p in first)
    assert len(set(first)) > 10
    assert first != [draw_prefix_steps((0, 90), 7, 3, p) for p in range(50)]


def test_draw_covers_both_bounds():
    assert {draw_prefix_steps((5, 7), 1, 0, p) for p in range(200)} == {5, 6, 7}


def test_zero_range_always_draws_zero():
    assert {draw_prefix_steps((0, 0), 1, 0, p) for p in range(20)} == {0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_prefix_steps.py`
Expected: FAIL with `ImportError: cannot import name 'draw_prefix_steps'`.

- [ ] **Step 3: Implement**

In `src/weathergen/datasets/multi_stream_data_sampler.py`, change the import `from weathergen.train.utils import Stage, get_batch_size_from_config` to:

```python
from weathergen.train.utils import TRAIN, Stage, get_batch_size_from_config
```

Add after `forcing_time_window_handler`:

```python
def resolve_prefix_range(forecast_cfg: dict, stage: Stage) -> tuple[int, int]:
    """
    Range [min, max] of the unsupervised spin-up steps drawn per training step.

    `forecast.prefix_steps` lets a training rollout first advance the latent state for P steps
    without gradients or targets, so the supervised steps start from the model's own drifted
    state. Only the training stage uses it: validation and test configs inherit the training
    forecast settings, and must keep rolling out from the initial condition.
    """
    prefix = forecast_cfg.get("prefix_steps")
    if stage != TRAIN or prefix is None:
        return (0, 0)
    if (
        not isinstance(prefix, list | tuple)
        or len(prefix) != 2
        or not all(isinstance(p, int) for p in prefix)
        or not 0 <= prefix[0] <= prefix[1]
    ):
        raise ValueError(
            f"forecast.prefix_steps must be [min, max] with integers 0 <= min <= max, "
            f"got {prefix!r}"
        )
    return (prefix[0], prefix[1])


def draw_prefix_steps(
    prefix_range: tuple[int, int], seed: int, mini_epoch: int, position: int
) -> int:
    """
    Spin-up length of the training step at `position` (sample index in the rank's share).

    Seeded only by the run seed, the mini-epoch and the position, never by the rank or the
    loader worker, so every rank draws the same length for the same training step: no rank
    waits on another's longer spin-up, and all ranks run the same forecast-engine calls.
    """
    low, high = prefix_range
    if high == 0:
        return 0
    rng = np.random.default_rng([seed, mini_epoch, position])
    return int(rng.integers(low, high + 1))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_prefix_steps.py`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
RAYON_NUM_THREADS=1 .venv/bin/ruff check src/weathergen/datasets/multi_stream_data_sampler.py tests/test_prefix_steps.py
RAYON_NUM_THREADS=1 .venv/bin/ruff format --check tests/test_prefix_steps.py
git add tests/test_prefix_steps.py
git commit -m "Parse forecast.prefix_steps and draw the spin-up length per training step

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/weathergen/datasets/multi_stream_data_sampler.py tests/test_prefix_steps.py
```

---

### Task 2: Prefix batch builder and the batch's first supervised step

**Files:**
- Modify: `src/weathergen/datasets/batch.py` (`BatchSamples.__init__`, near line 171)
- Modify: `src/weathergen/datasets/multi_stream_data_sampler.py` (new method `_get_prefix_batch`, placed right after `_get_batch`)
- Test: `tests/test_forcing_sampler.py` (append)

**Interfaces:**
- Consumes: the existing `_build_batch_skeleton(idx, num_forecast_steps)`, `_materialize_target_chunk(batch, step_range)`, `_build_forcing_data(batch, ds, base_idx, num_output_steps, step_range)`, `_build_condition_data(batch, ds, base_idx, num_output_steps, step_range)` and `_get_output_length(n)`.
- Produces: the attribute `BatchSamples.first_supervised_step: int | None`, which defaults to `None` and is the absolute batch step of the first decoded and supervised step. Also `MultiStreamDataSampler._get_prefix_batch(idx, num_forecast_steps: int, prefix_steps: int) -> ModelBatch`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_forcing_sampler.py`:

```python
def test_first_supervised_step_defaults_to_none():
    batch = ModelBatch(["s"], 1, 1, 1, 4)
    assert batch.get_source_samples().first_supervised_step is None
    assert batch.get_target_samples().first_supervised_step is None


def _prefix_stub(output_offset=1):
    """Stub `self` for `_get_prefix_batch`: records the skeleton and chunk calls."""
    stub = _sampler()
    stub.output_offset = output_offset
    stub._get_output_length = lambda n: MultiStreamDataSampler._get_output_length(stub, n)
    calls = types.SimpleNamespace(skeleton=None, materialized=None, conditions=[])
    reader = _FakeReader()

    def skeleton(idx, num_forecast_steps):
        calls.skeleton = (int(idx), num_forecast_steps)
        return ModelBatch([], 0, 0, output_offset, stub._get_output_length(num_forecast_steps))

    def materialize(batch, step_range):
        calls.materialized = tuple(step_range)
        # like the real chunk builder, which also builds the chunk's forcing
        MultiStreamDataSampler._build_forcing_data(
            stub, batch, reader, 10, batch.output_steps, step_range=step_range
        )
        return batch

    def get_condition(idx):
        calls.conditions.append(int(idx))
        return [float(idx)]

    stub._build_batch_skeleton = skeleton
    stub._materialize_target_chunk = materialize
    stub.forcing_datasets = {"F": types.SimpleNamespace(readers=[reader])}
    stub._build_forcing_data = lambda *a, **k: MultiStreamDataSampler._build_forcing_data(
        stub, *a, **k
    )
    stub.condition_datasets = {
        "C": types.SimpleNamespace(readers=[types.SimpleNamespace(get_condition=get_condition)])
    }
    stub._build_condition_data = lambda *a, **k: MultiStreamDataSampler._build_condition_data(
        stub, *a, **k
    )
    return stub, reader, calls


def test_prefix_batch_targets_only_the_supervised_steps_and_forces_every_step():
    stub, reader, calls = _prefix_stub(output_offset=1)
    batch = MultiStreamDataSampler._get_prefix_batch(stub, 10, 2, 3)

    assert calls.skeleton == (10, 5)  # 3 spin-up + 2 supervised forecast steps
    assert calls.materialized == (4, 6)  # targets for absolute steps 4 and 5 only
    assert sorted(reader.requested) == [11, 12, 13, 14, 15]  # forcing for every forecast step
    forcing = batch.get_source_samples().forcing
    assert [len(f) > 0 for f in forcing] == [False, True, True, True, True, True]
    assert calls.conditions == [10, 11, 12, 13]  # steps before the supervised chunk
    assert batch.get_source_samples().first_supervised_step == 4


def test_prefix_batch_with_offset_zero():
    stub, reader, calls = _prefix_stub(output_offset=0)
    batch = MultiStreamDataSampler._get_prefix_batch(stub, 10, 2, 3)

    assert calls.materialized == (3, 5)
    assert sorted(reader.requested) == [10, 11, 12, 13, 14]
    assert batch.get_source_samples().first_supervised_step == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_forcing_sampler.py -k "first_supervised or prefix_batch"`
Expected: FAIL with `AttributeError: ... 'first_supervised_step'` and `AttributeError: ... '_get_prefix_batch'`.

- [ ] **Step 3: Implement**

In `src/weathergen/datasets/batch.py`, `BatchSamples.__init__`, after `self.forcing = [[] for i in range(output_steps)]`:

```python
        # absolute output step of the first decoded and supervised step; the steps before it
        # are a training spin-up (forecast.prefix_steps) that only advances the latent state.
        # None: every output step is supervised.
        self.first_supervised_step: int | None = None
```

In `src/weathergen/datasets/multi_stream_data_sampler.py`, add after `_get_batch`:

```python
    def _get_prefix_batch(self, idx: int, num_forecast_steps: int, prefix_steps: int):
        """
        Assemble a training batch whose rollout starts with `prefix_steps` spin-up steps.

        The sample spans prefix_steps + num_forecast_steps forecast steps. The input side is
        built as in `_get_batch`; targets only for the last `num_forecast_steps` steps, with the
        step-range builders of lazy inference, so the spin-up steps' target slots stay empty
        (the model does not decode them and the loss skips them). Forcing and conditions are
        built for every step, since the model consumes them at each step.
        """
        total_steps = prefix_steps + num_forecast_steps
        num_output_steps = self._get_output_length(total_steps)
        first_supervised = self.output_offset + prefix_steps

        batch = self._build_batch_skeleton(idx, total_steps)
        # targets (and that chunk's conditions and forcing) of the supervised steps
        self._materialize_target_chunk(batch, (first_supervised, num_output_steps))

        # conditions and forcing of the steps before them; as in `_get_batch`, conditions
        # start at step 0 and forcing at the first forecast step
        for _, condition_data in self.condition_datasets.items():
            self._build_condition_data(
                batch,
                condition_data.readers[0],
                idx,
                num_output_steps,
                step_range=(0, first_supervised),
            )
        for _, forcing_data in self.forcing_datasets.items():
            self._build_forcing_data(
                batch,
                forcing_data.readers[0],
                idx,
                num_output_steps,
                step_range=(self.output_offset, first_supervised),
            )

        batch.get_source_samples().first_supervised_step = first_supervised
        return batch
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_forcing_sampler.py`
Expected: all PASS, including the existing tests.

- [ ] **Step 5: Lint and commit**

```bash
RAYON_NUM_THREADS=1 .venv/bin/ruff check src/weathergen/datasets/batch.py src/weathergen/datasets/multi_stream_data_sampler.py tests/test_forcing_sampler.py
git commit -m "Build training batches with an unsupervised spin-up before the supervised steps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/weathergen/datasets/batch.py src/weathergen/datasets/multi_stream_data_sampler.py tests/test_forcing_sampler.py
```

---

### Task 3: Sampler iteration: room for the prefix, one draw per step, dispatch

**Files:**
- Modify: `src/weathergen/datasets/multi_stream_data_sampler.py`: in `__init__`, near line 158 (forecast config) and line 205 (`check_samples` call) and line 207 (RNG seed); in `reset` (near line 530); in `__iter__` (near line 1328).
- Test: `tests/test_forcing_sampler.py` (update `test_iter_skips_sample_without_forcing` and `_iter_stub`; append new tests)

**Interfaces:**
- Consumes: `resolve_prefix_range`, `draw_prefix_steps` (Task 1); `_get_prefix_batch` (Task 2).
- Produces: sampler attributes `_prefix_range: tuple[int, int]` and `_prefix_seed: int`. Training batches with P > 0 come from `_get_prefix_batch`; P = 0 still uses `_get_batch` unchanged.

- [ ] **Step 1: Write the failing tests and update the existing stubs**

In `tests/test_forcing_sampler.py`, add these keyword arguments to the `types.SimpleNamespace(...)` stub in `test_iter_skips_sample_without_forcing` and to the one in `_iter_stub`:

```python
        rank=0,
        mini_epoch=0,
        _prefix_range=(0, 0),
        _prefix_seed=0,
```

Append:

```python
_GOOD = types.SimpleNamespace(
    sources_empty=lambda: False, is_nan=lambda: False, targets_empty=lambda: False
)


def _prefix_iter_stub(rank, prefix_range=(0, 90), get_prefix_batch=None):
    seen = []

    def get_batch(idx, num_forecast_steps):
        seen.append(("eager", 0))
        return _GOOD

    def prefix_batch(idx, num_forecast_steps, prefix_steps):
        seen.append(("prefix", prefix_steps))
        return _GOOD

    stub = types.SimpleNamespace(
        worker_workset=lambda: (rank * 4, rank * 4 + 4),
        len=4,
        rank=rank,
        mini_epoch=0,
        reset=lambda: (np.arange(8), np.full(4, 15)),
        batch_size=1,
        mode_cfg={"training_mode": ["forecast"]},
        _lazy_targets=False,
        _get_batch=get_batch,
        _get_prefix_batch=get_prefix_batch or prefix_batch,
        _prefix_range=prefix_range,
        _prefix_seed=7,
    )
    return stub, seen


def test_every_rank_draws_the_same_prefix_for_the_same_step():
    stub0, seen0 = _prefix_iter_stub(rank=0)
    stub1, seen1 = _prefix_iter_stub(rank=1)
    list(MultiStreamDataSampler.__iter__(stub0))
    list(MultiStreamDataSampler.__iter__(stub1))
    assert seen0 == seen1
    assert len(seen0) == 4
    assert all(0 <= p <= 90 for _, p in seen0)


def test_zero_prefix_range_keeps_the_eager_batch():
    stub, seen = _prefix_iter_stub(rank=0, prefix_range=(0, 0))
    list(MultiStreamDataSampler.__iter__(stub))
    assert seen == [("eager", 0)] * 4


def test_missing_forcing_in_a_prefix_skips_the_sample():
    tried = []

    def prefix_batch(idx, num_forecast_steps, prefix_steps):
        tried.append(int(idx))
        if idx == 0:
            raise ForcingUnavailableError("forcing stream 'F': window 3: no data")
        return _GOOD

    stub, _ = _prefix_iter_stub(rank=0, prefix_range=(1, 1), get_prefix_batch=prefix_batch)
    batches = list(MultiStreamDataSampler.__iter__(stub))
    assert tried == [0, 1, 2, 3, 4]
    assert batches == [_GOOD] * 4


class _ResetStub:
    """Stub `self` for `reset`: records the step count start dates must leave room for."""

    def __init__(self, prefix_range):
        self.room = []
        self.data_loader_rng_seed = 1
        self._prefix_range = prefix_range
        self._get_fsm = lambda: 15
        self.check_samples = lambda n: self.room.append(("check", n))
        self._calc_baseperms = lambda n: self.room.append(("perms", n)) or np.arange(10)
        self.samples_per_mini_epoch = 4
        self.batch_size = 1
        self.repeat_data = False
        self.shuffle = False
        self.forecast_policy = "fixed"
        self.list_num_forecast_steps = np.array([15])
        self.tokenizer = types.SimpleNamespace(reset_rng=lambda rng: None)

    def __len__(self):
        return 4


@pytest.mark.parametrize("prefix_range, room", [((0, 90), 105), ((0, 0), 15)])
def test_start_dates_leave_room_for_the_longest_prefix(prefix_range, room):
    stub = _ResetStub(prefix_range)
    perms, steps = MultiStreamDataSampler.reset(stub)
    assert stub.room == [("check", room), ("perms", room)]
    assert list(steps) == [15] * 4  # the supervised steps stay num_steps
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_forcing_sampler.py`
Expected: the new prefix tests FAIL: `__iter__` never calls `_get_prefix_batch`, and `reset` passes 15 instead of 105.

- [ ] **Step 3: Implement**

In `__init__`, right after `self.list_num_forecast_steps = np.array(steps, dtype=np.int32)`:

```python
        # unsupervised spin-up steps drawn per training step (training stage only)
        self._prefix_range = resolve_prefix_range(forecast_cfg, stage)
```

Replace `self.check_samples(self._get_fsm())` in `__init__` with:

```python
        self.check_samples(self._get_fsm() + self._prefix_range[1])
```

After `self.data_loader_rng_seed = rs if rs > nw else rs * 97`, add:

```python
        # spin-up lengths are seeded by the run seed alone (data_loader_rng_seed is made
        # unique per rank and worker), so every rank draws the same length for a step
        self._prefix_seed = int(rs)
```

In `reset`, replace

```python
        fsm = self._get_fsm()
        self.check_samples(fsm)
        perms = self._calc_baseperms(fsm)
```

with

```python
        fsm = self._get_fsm()
        # start dates must leave room for the longest spin-up before the supervised steps
        room = fsm + self._prefix_range[1]
        self.check_samples(room)
        perms = self._calc_baseperms(room)
```

In `__iter__`, after `iter_start, iter_end = self.worker_workset()`, add:

```python
        # position of this worker's first sample in the rank's share: the same on every rank
        rank_position = iter_start - self.rank * self.len
```

Inside the `for i, _bidx in enumerate(...)` loop, after `num_forecast_steps = perms_num_forecast_steps[i]`, add:

```python
            # unsupervised spin-up of this training step, identical on every rank
            prefix_steps = draw_prefix_steps(
                self._prefix_range,
                self._prefix_seed,
                self.mini_epoch,
                rank_position + i * self.batch_size,
            )
```

In the eager branch, replace

```python
                    try:
                        batch = self._get_batch(idx, num_forecast_steps)
```

with

```python
                    try:
                        if prefix_steps > 0:
                            batch = self._get_prefix_batch(idx, num_forecast_steps, prefix_steps)
                        else:
                            batch = self._get_batch(idx, num_forecast_steps)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_forcing_sampler.py tests/test_prefix_steps.py tests/test_forecast_chunks.py`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
RAYON_NUM_THREADS=1 .venv/bin/ruff check src/weathergen/datasets/multi_stream_data_sampler.py tests/test_forcing_sampler.py
git commit -m "Draw a spin-up per training step and build those batches with the prefix builder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/weathergen/datasets/multi_stream_data_sampler.py tests/test_forcing_sampler.py
```

---

### Task 4: Model: spin-up steps without gradients and without decoding

**Files:**
- Modify: `src/weathergen/model/model.py`, the rollout loop in `Model.forward` (near lines 818-864)
- Test: `tests/test_model_rollout_steps.py` (append)

**Interfaces:**
- Consumes: `BatchSamples.first_supervised_step` (Task 2).
- Produces: in `Model.forward`, batch steps before `first_supervised_step` run the forcing module and forecast engine under `torch.no_grad()`, with no decoder or latent predictions (the `output.physical` entry stays empty).

This task carries the spec's loss regression check. `LossPhysical.compute_loss` skips every step whose `preds.physical[step]` has no entry for the stream (`if not preds_batch: continue`), so the test pins that spin-up steps leave those entries empty. Calling `compute_loss` itself needs a full config; the smoke run (Task 10) checks the loss on real batches.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model_rollout_steps.py`:

```python
def _recording_grad(model):
    """Wrap the stub's forecast engine and forcing module to record whether grad is on."""
    grad = {"fe": [], "forcing": []}
    fe, forcing = model.forecast_engine, model.forcing_module

    def forecast_engine(tokens, condition, coords=None):
        grad["fe"].append(torch.is_grad_enabled())
        return fe(tokens, condition, coords)

    def forcing_module(tokens, condition, field, num_aux):
        grad["forcing"].append(torch.is_grad_enabled())
        return forcing(tokens, condition, field, num_aux)

    model.forecast_engine, model.forcing_module = forecast_engine, forcing_module
    return grad


def test_spin_up_steps_advance_without_gradients_and_are_not_decoded():
    model, log = _stub_model()
    grad = _recording_grad(model)
    batch = _batch(offset=1, num_steps=4)
    batch.first_supervised_step = 3
    output = Model.forward(model, MODEL_PARAMS, batch, batch.get_output_len())

    assert log.fe_calls == 4
    assert log.forcing == [1, 2, 3, 4]  # forcing still drives every step
    assert grad["fe"] == [False, False, True, True]
    assert grad["forcing"] == [False, False, True, True]
    assert [(d["batch_step"], d["fe_calls"]) for d in log.decodes] == [(3, 3), (4, 4)]
    assert all(not output.physical[i] for i in (1, 2))  # the loss skips these steps
    assert all("latent_state" not in output.latent[i] for i in (1, 2))


def test_first_supervised_step_at_the_first_forecast_step_changes_nothing():
    model, log = _stub_model()
    batch = _batch(offset=1, num_steps=2)
    batch.first_supervised_step = 1
    Model.forward(model, MODEL_PARAMS, batch, batch.get_output_len())

    assert log.fe_calls == 2
    assert log.forcing == [1, 2]
    assert [(d["batch_step"], d["fe_calls"]) for d in log.decodes] == [(1, 1), (2, 2)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_model_rollout_steps.py`
Expected: `test_spin_up_steps_advance_without_gradients_and_are_not_decoded` FAILS, because steps 1 and 2 are decoded with grad on.

- [ ] **Step 3: Implement**

In `Model.forward`, replace the block from `# Allow for pushforward trick` down to the pushforward `continue` with:

```python
        # Allow for pushforward trick
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        # first batch step that is decoded and supervised; the steps before it are a training
        # spin-up (forecast.prefix_steps) that only advances the latent state
        first_supervised = getattr(batch_ctx, "first_supervised_step", None)
        # roll-out in latent space, iterate and generate output over requested output steps
        for step in range(first_step, rollout_steps):
            # apply forecasting engine (if present)
            if self.forecast_engine:
                in_spin_up = (
                    first_supervised is not None and step + batch_step_offset < first_supervised
                )
                without_grad = in_spin_up or (
                    p_fwd and self.training and step != rollout_steps - 1
                )
                condition = batch_ctx.conditions[step + batch_step_offset]
                # inject the per-step spatial forcing (e.g. SST) into the FE latent state /
                # conditioning before advancing; no-op when no forcing module or field.
                if self.forcing_module is not None:
                    forcing_list = getattr(batch_ctx, "forcing", None)
                    fidx = step + batch_step_offset
                    forcing_field = (
                        forcing_list[fidx]
                        if forcing_list is not None and fidx < len(forcing_list)
                        else None
                    )
                    if forcing_field_is_empty(forcing_field) and not self._warned_missing_forcing:
                        logger.warning(
                            f"No forcing field for forecast step {fidx}: the forecast step runs "
                            "without forcing (e.g. a teacher's target batch). Warned once per "
                            "model."
                        )
                        self._warned_missing_forcing = True
                    if without_grad:
                        # nothing is back-propagated through this step: keep no activations
                        with torch.no_grad():
                            tokens, condition = self.forcing_module(
                                tokens, condition, forcing_field, self.num_aux_tokens
                            )
                    else:
                        # checkpointed like the FE blocks: otherwise the forcing embedding keeps
                        # its activations for every rollout step (~0.3 GB per step on HEALPix
                        # level 5)
                        tokens, condition = checkpoint(
                            self.forcing_module,
                            tokens,
                            condition,
                            forcing_field,
                            self.num_aux_tokens,
                            use_reentrant=False,
                        )
                if without_grad:
                    # spin-up / pushforward: advance tokens without grad; no decoding
                    with torch.no_grad():
                        tokens = self.forecast_engine(
                            tokens, condition, coords=model_params.rope_coords
                        )
                    continue
```

Leave the rest of the loop (`prev_tokens = ...` through `predict_latent`) unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_model_rollout_steps.py tests/test_forcing_injection.py tests/test_decoder_gather.py tests/test_forecast_chunks.py`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
RAYON_NUM_THREADS=1 .venv/bin/ruff check src/weathergen/model/model.py tests/test_model_rollout_steps.py
git commit -m "Run the training spin-up steps without gradients and without decoding

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/weathergen/model/model.py tests/test_model_rollout_steps.py
```

---

### Task 5: Dynamic loss weights follow the first supervised step (needs the user's permission)

The dynamic channel weights (`DynamicLossEMA`) are updated only at `timestep_idx == self.forecast_offset`. With a spin-up, that step has no prediction, so the weights would update only on batches with P = 0 (about 1 in 91). The window and control runs would then train under different weightings. The helper and its test are new tracked files. The two-line use sits in `loss_module_physical.py`, one of the user's uncommitted files.

**Gate:** do this task only if the user has explicitly allowed editing `src/weathergen/train/loss_modules/loss_module_physical.py`. Never stage or commit that file: the edit stays in the working tree alongside the user's own changes. Without permission, skip this task and record in the ledger that the window run updates the EMA weights only on P = 0 batches.

**Files:**
- Create: `src/weathergen/train/loss_modules/supervised_steps.py`
- Modify (uncommitted, user's file): `src/weathergen/train/loss_modules/loss_module_physical.py` (imports; the EMA update condition near line 417)
- Test: `tests/test_supervised_steps.py` (new)

**Interfaces:**
- Produces: `first_supervised_timestep(preds_physical: list[dict], stream_name: str) -> int | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_supervised_steps.py`:

```python
"""The first forecast step that carries predictions for a stream."""

from weathergen.train.loss_modules.supervised_steps import first_supervised_timestep


def test_first_step_with_predictions():
    preds = [{}, {}, {}, {"S": [1]}, {"S": [2]}]
    assert first_supervised_timestep(preds, "S") == 3


def test_without_spin_up_it_is_the_first_forecast_step():
    assert first_supervised_timestep([{}, {"S": [1]}, {"S": [2]}], "S") == 1


def test_stream_without_predictions():
    assert first_supervised_timestep([{}, {"T": [1]}], "S") is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_supervised_steps.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/weathergen/train/loss_modules/supervised_steps.py`:

```python
"""Which forecast steps of a rollout carry predictions."""


def first_supervised_timestep(preds_physical: list[dict], stream_name: str) -> int | None:
    """
    Index of the first forecast step with predictions for `stream_name` (None if none).

    Without a training spin-up this is the first forecast step; with one
    (forecast.prefix_steps) it is the first step after the spin-up, whose error the dynamic
    channel weights track instead of the never-decoded first forecast step.
    """
    return next((i for i, step in enumerate(preds_physical) if step.get(stream_name)), None)
```

In `loss_module_physical.py` (uncommitted), add the import:

```python
from weathergen.train.loss_modules.supervised_steps import first_supervised_timestep
```

Before `for timestep_idx, (preds_cur, target_cur) in enumerate(`, inside the per-stream loop, add:

```python
            # the dynamic channel weights track the first supervised step's error
            ema_timestep = first_supervised_timestep(preds.physical, stream_name)
```

In the EMA update condition, replace `and timestep_idx == self.forecast_offset` with `and timestep_idx == ema_timestep`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_supervised_steps.py tests/test_loss_functions.py`
Expected: PASS (`tests/test_loss_functions.py` is the user's untracked test of the loss code).

- [ ] **Step 5: Commit only the new tracked files**

```bash
RAYON_NUM_THREADS=1 .venv/bin/ruff check src/weathergen/train/loss_modules/supervised_steps.py tests/test_supervised_steps.py
git add src/weathergen/train/loss_modules/supervised_steps.py tests/test_supervised_steps.py
git commit -m "Find the first forecast step with predictions, for the dynamic loss weights

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/weathergen/train/loss_modules/supervised_steps.py tests/test_supervised_steps.py
git status --short src/weathergen/train/loss_modules/loss_module_physical.py   # expect "MM": user's staged change + this unstaged edit
```

---

### Task 6: Log the spin-up length

**Files:**
- Modify: `src/weathergen/train/trainer.py` (new module-level `prefix_length`; `Trainer.__init__` near line 82; `train` after the forward pass; `_log_terminal` TRAIN branch; `_log` TRAIN branch)
- Modify: `src/weathergen/utils/train_logger.py` (`TrainLogger.add_logs`, near line 96)
- Test: `tests/test_prefix_logging.py` (new)

**Interfaces:**
- Consumes: `BatchSamples.first_supervised_step` (Task 2).
- Produces: `prefix_length(batch) -> int` in `weathergen.train.trainer`, and `TrainLogger.add_logs(..., extra_metrics: dict[str, float] | None = None)`. The training metrics key is `forecast_prefix_steps`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prefix_logging.py`:

```python
"""The training spin-up length reaches the terminal log and the training metrics."""

import types

from weathergen.datasets.batch import BatchSamples
from weathergen.train.trainer import prefix_length
from weathergen.utils.train_logger import TrainLogger


def test_prefix_length():
    batch = BatchSamples(["s"], 1, 6, [1, 2, 3, 4, 5])
    assert prefix_length(batch) == 0
    batch.first_supervised_step = 4
    assert prefix_length(batch) == 3


def test_add_logs_writes_extra_training_metrics():
    logged = []
    stub = types.SimpleNamespace(log_metrics=lambda stage, metrics: logged.append(metrics))
    TrainLogger.add_logs(
        stub, "train", 8, {}, {}, avg_loss=[1.0], lr=1e-5,
        extra_metrics={"forecast_prefix_steps": 42.5},
    )
    assert logged[0]["forecast_prefix_steps"] == 42.5
    assert logged[0]["loss_avg_mean"] == 1.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_prefix_logging.py`
Expected: FAIL with `ImportError: cannot import name 'prefix_length'`.

- [ ] **Step 3: Implement**

In `src/weathergen/utils/train_logger.py`, add the parameter `extra_metrics: dict[str, float] | None = None` to `add_logs` after `elapsed_training_time_seconds`. Right before `self.log_metrics(stage, metrics)` add:

```python
        metrics |= extra_metrics or {}
```

In `src/weathergen/train/trainer.py`, add at module level (after the imports):

```python
def prefix_length(batch) -> int:
    """Unsupervised spin-up steps at the start of a training batch's rollout (0 without)."""
    first = getattr(batch, "first_supervised_step", None)
    return 0 if first is None else first - batch.get_output_idxs()[0]
```

In `Trainer.__init__`, next to `self.optimizer: torch.optim.Optimizer | None = None`:

```python
        # spin-up length of each training step since the last metrics log
        self._prefix_hist: list[int] = []
```

In `train`, right after the `preds = self.model(...)` call:

```python
                self._prefix_hist.append(prefix_length(x))
```

In `_log_terminal`, TRAIN branch, after the line that adds `(lr=...)` to `pstr`:

```python
                    if self._prefix_hist:
                        pstr += f"prefix={self._prefix_hist[-1]}, "
```

In `_log`, in the `elif self.cf.general.istep >= 0:` branch, pass to `add_logs`:

```python
                    extra_metrics=(
                        {"forecast_prefix_steps": float(np.mean(self._prefix_hist))}
                        if self._prefix_hist
                        else None
                    ),
```

and after `loss_calculator.stddev_unweighted_hist = []` add:

```python
        if stage == TRAIN:
            self._prefix_hist = []
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_prefix_logging.py tests/test_rollout_state_path.py`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
RAYON_NUM_THREADS=1 .venv/bin/ruff check src/weathergen/train/trainer.py src/weathergen/utils/train_logger.py tests/test_prefix_logging.py
git add tests/test_prefix_logging.py
git commit -m "Log the training spin-up length

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- src/weathergen/train/trainer.py src/weathergen/utils/train_logger.py tests/test_prefix_logging.py
```

- [ ] **Step 6: Run the whole forcing-branch test set**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_forcing.py tests/test_forcing_sampler.py tests/test_forcing_injection.py tests/test_forcing_configs.py tests/test_forcing_fesom_data.py tests/test_forecast_chunks.py tests/test_model_rollout_steps.py tests/test_output_forecast_steps.py tests/test_decoder_gather.py tests/test_rollout_state_path.py tests/test_prefix_steps.py tests/test_prefix_logging.py`
Expected: all PASS. Add `tests/test_supervised_steps.py` if Task 5 ran.

---

### Task 7: Job scripts (untracked)

**Files:**
- Modify (untracked): `sbatch_fesom_forced_smoke.sh`, `sbatch_fesom_forced_12h.sh`
- Create (untracked): `sbatch_fesom_forced_rollout.sh`

**Interfaces:**
- Produces:
  - smoke variant `win`: from frcadd04 mini-epoch 12 into `smkwin01`, prefix `[0,90]`, with GPU memory sampling;
  - 12 h variants `window` (`frcwin01`) and `control` (`frcctl01`), with the mini-epoch count taken from `WG_MINI_EPOCHS`;
  - `sbatch_fesom_forced_rollout.sh FROM_RUN FROM_EPOCH RUN_ID NUM_STEPS`.

- [ ] **Step 1: Smoke script.** In `sbatch_fesom_forced_smoke.sh`, add `FROM_RUN=ykwx7bl2` and `FROM_EPOCH=-1` before `case "$VARIANT" in`. Add this case:

```bash
  win)
    # spin-up training from the forced model frcadd04 (forecast engine trainable via WG_TRAIN_FE=1)
    FROM_RUN=frcadd04
    FROM_EPOCH=12
    RUN_ID=smkwin01
    EXTRA=()
    OPTS+=("training_config.forecast.prefix_steps=[0,90]")
    ;;
```

Change the srun line to use `--from-run-id "$FROM_RUN" -e "$FROM_EPOCH"`. Wrap it with GPU-memory sampling on the batch node:

```bash
# memory of the batch node's GPUs every 10 s (with expandable segments: reserved memory)
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 10 \
  > "output/smoke_${VARIANT}_gpumem_${SLURM_JOBID}.csv" &
SMI_PID=$!
srun --label "$TRAIN_CONTINUE" --from-run-id "$FROM_RUN" -e "$FROM_EPOCH" --run-id "$RUN_ID" \
  "${EXTRA[@]}" --options "${OPTS[@]}" > "output/smoke_${VARIANT}_${SLURM_JOBID}.txt" 2>&1
echo "srun exit code: $?"
kill "$SMI_PID"
```

Update the usage comment with `sbatch --job-name=smk-win --export=ALL,WG_NUM_STEPS=15,WG_TRAIN_FE=1 sbatch_fesom_forced_smoke.sh win`.

- [ ] **Step 2: 12 h script.** In `sbatch_fesom_forced_12h.sh`, change the base option `training_config.num_mini_epochs=6` to `training_config.num_mini_epochs=${WG_MINI_EPOCHS:-6}`. Keep it as the second entry of `OPTS`, because `forced3` slices `"${OPTS[@]:2}"`. Add the cases:

```bash
  window)
    FROM_RUN=frcadd04
    FROM_EPOCH=12
    RUN_ID=frcwin01
    EXTRA=()
    OPTS+=("training_config.forecast.prefix_steps=[0,90]")
    ;;
  control)
    FROM_RUN=frcadd04
    FROM_EPOCH=12
    RUN_ID=frcctl01
    EXTRA=()
    OPTS+=("training_config.forecast.prefix_steps=[0,0]")
    ;;
```

Add both to the usage line and to the header comment, with this explanation: "window/control continue frcadd04's final weights with the same recipe; window trains each step after a random 0-90-day spin-up without gradients, control is the same 12 h of training without it. Set WG_MINI_EPOCHS from the smoke run's step time."

- [ ] **Step 3: One-year rollout script.** Create `sbatch_fesom_forced_rollout.sh` as a copy of `sbatch_fesom_forced_180d.sh` with these changes:
  - `#SBATCH --time=04:00:00` and `#SBATCH --output=logs/fesom-rollout-%x.%j.out` (and `--error` to match);
  - this header comment:

```bash
# One free-running forced forecast from 2209-01-01 (the held-out year).
#
#   sbatch --job-name=yr1-add04 sbatch_fesom_forced_rollout.sh frcadd04 12 yr1add04 363
#
# Arguments: source run, its mini-epoch (-1 = latest), output run id, number of daily steps.
# 363 steps end on 2209-12-30, the last day with forcing (2209-12-31's 00 UTC stamp is in
# 2210). end_date 2210-01-02 only satisfies the sampler's index arithmetic, which needs
# start + steps + 2 windows for one sample; no window after 2209-12-30 is read.
```

  - the variant `case` replaced by:

```bash
FROM_RUN=${1:?usage: sbatch sbatch_fesom_forced_rollout.sh FROM_RUN FROM_EPOCH RUN_ID NUM_STEPS}
FROM_EPOCH=${2:?missing FROM_EPOCH}
RUN_ID=${3:?missing RUN_ID}
NUM_STEPS=${4:?missing NUM_STEPS}
```

  - `test_config.end_date=2210-01-02T00:00` and `test_config.forecast.num_steps=$NUM_STEPS` in `OPTS`;
  - output file `output/fesom_rollout_${RUN_ID}_${SLURM_JOBID}.txt`.

- [ ] **Step 4: Check the scripts**

Run: `bash -n sbatch_fesom_forced_smoke.sh && bash -n sbatch_fesom_forced_12h.sh && bash -n sbatch_fesom_forced_rollout.sh && echo ok`
Expected: `ok`.

Then dry-load the window run's merged config:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python - <<'EOF'
from weathergen.common import config
opts = config.from_cli_arglist(["general.istep=0", "training_config.num_mini_epochs=6",
                                "training_config.forecast.prefix_steps=[0,90]"])
cf = config.load_merge_configs(None, "frcadd04", 12, None, {}, opts)
print(cf.training_config.forecast.prefix_steps, cf.training_config.forecast.num_steps,
      cf.general.istep, "forecast_engine" in cf.freeze_modules)
EOF
```

Expected: `[0, 90] 15 0 False`.

No commit: these scripts are untracked by design.

---

### Task 8: One-year drift diagnostics script (untracked)

**Files:**
- Create (untracked): `diag_forced_year.py`, `test_diag_forced_year.py` (repo root)

**Interfaces:**
- Consumes: `paper_eval.climatology.ClimLookup`, `paper_eval.merged_zarr.open_merged_ranks`, `paper_eval.meshtools.load_mesh` and `masks_and_weights`.
- Produces: `paper_eval/figures/year_diag/diag_<run>.nc` with `pred_mean` and `true_mean` (dims `lead, band, field`) and `variability_ratio` (dims `lead, field`), plus `regional_means.png` and `variability_ratio.png`.

- [ ] **Step 1: Write the failing tests** in `test_diag_forced_year.py`:

```python
"""Pure functions of diag_forced_year.py."""

import numpy as np

from diag_forced_year import band_mask, variability_ratio, weighted_mean


def test_weighted_mean_ignores_zero_weight_points():
    x = np.array([1.0, 3.0, np.nan])
    w = np.array([1.0, 3.0, 0.0])
    assert weighted_mean(x, w) == 2.5


def test_band_mask_follows_the_paper_eval_rule():
    lat = np.array([-90.0, -30.0, 0.0, 29.99, 30.0, 89.9])
    assert band_mask(lat, (-30.0, 30.0)).tolist() == [False, True, True, True, False, False]
    assert band_mask(lat, None).all()


def test_perfect_forecast_has_ratio_one():
    rng = np.random.default_rng(0)
    truth, clim, w = rng.normal(size=100), rng.normal(size=100), rng.uniform(1, 2, 100)
    assert np.isclose(variability_ratio(truth, truth, clim, w), 1.0)


def test_ratio_measures_spread_not_bias():
    rng = np.random.default_rng(1)
    truth, clim, w = rng.normal(size=100), rng.normal(size=100), rng.uniform(1, 2, 100)
    pred = clim + 0.5 * (truth - clim) + 3.0  # half the anomaly, plus a uniform offset
    assert np.isclose(variability_ratio(pred, truth, clim, w), 0.5)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q test_diag_forced_year.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'diag_forced_year'`.

- [ ] **Step 3: Implement `diag_forced_year.py`**

```python
"""
Drift diagnostics of one-year forced runs: regional means against the truth and the
variability ratio, per lead, for selected FESOM_NODE fields.

    .venv/bin/python diag_forced_year.py RUN [RUN ...]

For each run, reads results/<run>/validation_chkpt00000_rank*.zip (one forecast, written by
sbatch_fesom_forced_rollout.sh) and the paper_eval day-of-year climatology, and writes
paper_eval/figures/year_diag/diag_<run>.nc; then draws regional_means.png and
variability_ratio.png there for all runs given.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))  # paper_eval is a plain folder at the repo root

from paper_eval.climatology import ClimLookup  # noqa: E402
from paper_eval.merged_zarr import open_merged_ranks  # noqa: E402
from paper_eval.meshtools import load_mesh, masks_and_weights  # noqa: E402

warnings.filterwarnings("ignore")
STREAM = "FESOM_NODE"
FIELDS = ("sst", "ssh", "temp_107", "salt_107", "temp_450", "salt_450")
BANDS = {
    "global": None,
    "90S-30S": (-90.0, -30.0),
    "30S-30N": (-30.0, 30.0),
    "30N-90N": (30.0, 90.0),
}
MESH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
CLIM = REPO / "paper_eval/cache_oe3i7whz/climatology/clim_FESOM_NODE_2180-2209.zarr"
OUT = REPO / "paper_eval/figures/year_diag"
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # validated categorical slots 1-4
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def band_mask(lat, bounds):
    """Points with south <= lat < north (all points for None), paper_eval's region rule."""
    lat = np.asarray(lat)
    if bounds is None:
        return np.ones(lat.shape, dtype=bool)
    south, north = bounds
    return (lat >= south) & (lat < north)


def weighted_mean(x, w):
    """Area-weighted mean of x over the points with weight > 0 (values elsewhere ignored)."""
    keep = w > 0
    return float(np.sum(w[keep] * x[keep]) / np.sum(w[keep]))


def anomaly_std(x, clim, w):
    """Area-weighted spatial standard deviation of the anomaly x - clim."""
    a = x - clim
    m = weighted_mean(a, w)
    return float(np.sqrt(weighted_mean((a - m) ** 2, w)))


def variability_ratio(pred, truth, clim, w):
    """Spread of the forecast anomaly over that of the true anomaly (1 = realistic)."""
    return anomaly_std(pred, clim, w) / anomaly_std(truth, clim, w)


def diagnose(run):
    root = open_merged_ranks(REPO / "results" / run, 0)
    s = root.store.samples[0]
    init = root.store.init_times[s]
    target0 = root[f"{s}/{STREAM}/1/target"]
    channels = list(target0.attrs["channels"])
    cols = [channels.index(f) for f in FIELDS]
    mask, w = masks_and_weights(load_mesh(MESH), STREAM, channels)
    lat = np.asarray(target0["coords"])[:, 0]
    weights = {
        band: (w[cols] * mask[cols] * band_mask(lat, bounds)[None, :]).astype(np.float64)
        for band, bounds in BANDS.items()
    }
    clim = ClimLookup(CLIM, channels)
    leads = sorted(int(k) for k in root[f"{s}/{STREAM}"].group_keys() if int(k) > 0)

    pred_mean = np.full((len(leads), len(BANDS), len(FIELDS)), np.nan)
    true_mean = np.full_like(pred_mean, np.nan)
    ratio = np.full((len(leads), len(FIELDS)), np.nan)
    for i, lead in enumerate(leads):
        truth = np.asarray(root[f"{s}/{STREAM}/{lead}/target"]["data"])[:, cols].T
        pred = np.asarray(root[f"{s}/{STREAM}/{lead}/prediction"]["data"])[:, cols, 0].T
        c = clim.get(init + np.timedelta64(lead, "D"))[:, cols].T
        for j, wb in enumerate(weights.values()):
            for k in range(len(FIELDS)):
                pred_mean[i, j, k] = weighted_mean(pred[k], wb[k])
                true_mean[i, j, k] = weighted_mean(truth[k], wb[k])
        for k in range(len(FIELDS)):
            ratio[i, k] = variability_ratio(pred[k], truth[k], c[k], weights["global"][k])

    dims = ("lead", "band", "field")
    return xr.Dataset(
        {
            "pred_mean": (dims, pred_mean),
            "true_mean": (dims, true_mean),
            "variability_ratio": (("lead", "field"), ratio),
        },
        coords={"lead": leads, "band": list(BANDS), "field": list(FIELDS)},
        attrs={"run": run, "init_time": str(init)},
    )


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot(diags):
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    first = diags[0]
    fig, axes = plt.subplots(len(FIELDS), len(BANDS), figsize=(13, 15), constrained_layout=True)
    for r, field in enumerate(FIELDS):
        for c, band in enumerate(BANDS):
            ax = axes[r, c]
            truth = first["true_mean"].sel(field=field, band=band)
            ax.plot(first.lead, truth, color=INK, lw=1.2, label="truth")
            for ds, colour in zip(diags, COLOURS, strict=False):
                ax.plot(ds.lead, ds["pred_mean"].sel(field=field, band=band), color=colour,
                        lw=1.6, label=ds.attrs["run"])
            ax.set_title(f"{field}, {band}", loc="left")
            _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Area-weighted means against the truth, one-year forecast from 2209-01-01",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / "regional_means.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, field in zip(axes.flat, FIELDS, strict=True):
        ax.axhspan(0.7, 1.3, color=GRID, alpha=0.6, lw=0)
        ax.axhline(1.0, color=INK2, lw=1.2, ls=(0, (4, 3)))
        for ds, colour in zip(diags, COLOURS, strict=False):
            ax.plot(ds.lead, ds["variability_ratio"].sel(field=field), color=colour, lw=1.6,
                    label=ds.attrs["run"])
        ax.set_title(field, loc="left")
        ax.set_xlabel("lead time (days)")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Variability ratio: forecast anomaly spread / true anomaly spread "
                 "(shaded 0.7-1.3)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(OUT / "variability_ratio.png", dpi=150)
    plt.close(fig)


def main(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    diags = []
    for run in runs:
        ds = diagnose(run)
        ds.to_netcdf(OUT / f"diag_{run}.nc")
        diags.append(ds)
        print(f"{run}: {len(ds.lead)} leads -> {OUT / f'diag_{run}.nc'}")
    plot(diags)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
```

- [ ] **Step 4: Run the tests to verify they pass, then try the script on an existing run**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q test_diag_forced_year.py`
Expected: 4 PASS.

Check the script end to end on the existing 180-day runs, in the shared partition:

```bash
sbatch --partition=shared --account=ab0995 --time=01:00:00 --mem=32G --cpus-per-task=8 \
  --output=logs/diag-%j.out \
  --wrap "OMP_NUM_THREADS=8 .venv/bin/python diag_forced_year.py f180add1 f180bas1 pekxvi5n"
```

Expected: three `diag_*.nc` files and both PNGs in `paper_eval/figures/year_diag/`. Look at the PNGs: truth lines are shared by all runs, and ratios are near 1 at day 1. No commit (untracked).

---

### Task 9: One-year reference runs (frcadd04, ykwx7bl2)

Independent of Tasks 1-6. These runs can start as soon as Task 7 exists.

- [ ] **Step 1: Submit**

```bash
sbatch --job-name=yr1-add04 sbatch_fesom_forced_rollout.sh frcadd04 12 yr1add04 363
sbatch --job-name=yr1-ykwx7 sbatch_fesom_forced_rollout.sh ykwx7bl2 -1 yr1ykwx7 363
```

Record both job ids in the ledger.

- [ ] **Step 2: Check the first chunk.** Wait in the background, not with polling sleeps, until `results/yr1add04/rollout_state.pt` exists or the log has a `Traceback`:

```bash
until [ -f results/yr1add04/rollout_state.pt ] || grep -qs Traceback output/fesom_rollout_yr1add04_*.txt; do sleep 30; done
```

If the log shows that a reader refuses the end date 2210-01-02, delete the broken runs (`results/`, `models/` and `logs/` of `yr1add04` and `yr1ykwx7`, plus their output files). Then resubmit both with 361 steps and `test_config.end_date=2209-12-31T00:00` (edit the script's `OPTS`). Use 361 for every one-year run and paper_eval config from then on, and record that in the ledger.

- [ ] **Step 3: Confirm completion.** Check that the job ended, the log has no `Traceback`, and `done_steps` equals the step count:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -c "import torch;print(torch.load('results/yr1add04/rollout_state.pt',map_location='cpu',weights_only=False)['done_steps'])"
```

Do the same for `yr1ykwx7`.

- [ ] **Step 4: paper_eval configs and scoring**

```bash
for run in yr1add04 yr1ykwx7; do
  sed -e "1s/.*/# Paper evaluation config — one-year forced forecast from 2209-01-01, run $run./" \
      -e "s/f180add1/$run/g" \
      -e "s/^  max: 180 .*/  max: 363             # per-sample scores computed for leads 1..max/" \
      -e "s/^  map_leads: \[30, 90, 180\]/  map_leads: [90, 180, 363]/" \
      -e "s/^  leads: \[1, 30, 90, 180\]/  leads: [1, 90, 180, 363]/" \
      -e "s/^  leads: \[30, 180\]/  leads: [90, 363]/" \
      paper_eval/configs/f180add1.yaml > paper_eval/configs/$run.yaml
  sbatch --partition=shared --time=02:00:00 --mem=64G paper_eval/sbatch/compute.sbatch paper_eval/configs/$run.yaml
done
diff paper_eval/configs/f180add1.yaml paper_eval/configs/yr1add04.yaml
```

Expected diff: only the header, `run_id`, `results_dir`, `max`, `map_leads` and the two `leads` lines. If the runs used 361 steps, use 361 in place of 363.

---

### Task 10: Smoke run of spin-up training

- [ ] **Step 1: Submit**

```bash
sbatch --job-name=smk-win --export=ALL,WG_NUM_STEPS=15,WG_TRAIN_FE=1 sbatch_fesom_forced_smoke.sh win
```

- [ ] **Step 2: Check the run** once it ends (wait in the background on the job id). In `output/smoke_win_<job>.txt`:
  - no `Traceback`, `OutOfMemoryError` or `CheckpointError`;
  - training lines show varying `prefix=` values and finite losses;
  - validation ran.

In `output/smoke_win_gpumem_<job>.csv`, the peak `memory.used` should be near the 15-step runs (about 72-75 GiB, below 80 GiB).

Take the step time from the `s/sec` values in the training lines: t_step = 1 / (s/sec), with batch size 1 per GPU.

If the run failed, read the traceback, fix the cause in a new commit (red-green test first), delete `smkwin01`'s `models/`, `results/` and `logs/` folders and the output files, and resubmit.

- [ ] **Step 3: Choose the mini-epoch count.** Use

`N = min(6, floor((11.5 * 3600 - 600) / (256 * t_step + 480)))`

(256 steps per mini-epoch, about 8 min of validation each, 10 min start-up). Record t_step, the memory peak and N in the ledger. If N < 3, the spin-up is too slow: record it, and before Task 11 ask the user whether to lower `max(prefix_steps)` or use a curriculum.

---

### Task 11: The two 12 h runs

- [ ] **Step 1: Submit both** with the N from Task 10:

```bash
sbatch --job-name=frc-win --export=ALL,WG_MINI_EPOCHS=<N> sbatch_fesom_forced_12h.sh window
sbatch --job-name=frc-ctl --export=ALL,WG_MINI_EPOCHS=<N> sbatch_fesom_forced_12h.sh control
```

Record the job ids.

- [ ] **Step 2: Monitor.** Watch with background waits, not polling sleeps. Check each log for `Traceback`, `OutOfMemoryError` and `CheckpointError`, and look at the losses and validations after each mini-epoch. Delete a broken run and resubmit it after the fix.

- [ ] **Step 3: On completion,** note each run's final checkpoint, the highest `frc*_chkpt*.chkpt` index (the `_latest` checkpoint is older). If a run hit the time limit, use `-1` and note it. Record both in the ledger.

---

### Task 12: One-year runs of window and control, scoring, diagnostics

- [ ] **Step 1: Submit**

```bash
sbatch --job-name=yr1-win sbatch_fesom_forced_rollout.sh frcwin01 <final epoch> yr1win01 363
sbatch --job-name=yr1-ctl sbatch_fesom_forced_rollout.sh frcctl01 <final epoch> yr1ctl01 363
```

Use the same step count as Task 9 (361 if that fallback was needed).

- [ ] **Step 2: Confirm completion** as in Task 9, Step 3.

- [ ] **Step 3: paper_eval configs and scoring** with the loop from Task 9, Step 4, for `yr1win01` and `yr1ctl01`.

- [ ] **Step 4: Diagnostics for all four runs**

```bash
sbatch --partition=shared --account=ab0995 --time=01:30:00 --mem=32G --cpus-per-task=8 \
  --output=logs/diag-%j.out \
  --wrap "OMP_NUM_THREADS=8 .venv/bin/python diag_forced_year.py yr1win01 yr1ctl01 yr1add04 yr1ykwx7"
```

Expected: four `diag_*.nc` files and both PNGs in `paper_eval/figures/year_diag/`.

---

### Task 13: Acceptance against the success criteria

**Files:**
- Create (untracked): `compare_year.py` (repo root)

- [ ] **Step 1: Validate the four-colour palette** (run from the dataviz skill's base directory):

`node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light`

Expected: all checks pass. Aqua and yellow may warn on contrast, which the direct labels and legend cover.

- [ ] **Step 2: Write `compare_year.py`**

```python
"""
Acceptance check of the long-horizon stability runs against the spec's success criteria
(docs/superpowers/specs/2026-10-08-long-horizon-stability-design.md).

    .venv/bin/python compare_year.py

Reads the paper_eval scores of the one-year runs (paper_eval/cache_oe3i7whz/<run>/), the drift
diagnostics (paper_eval/figures/year_diag/diag_<run>.nc) and the fine-tunes' training metrics;
prints one table per criterion and writes RMSE figures to paper_eval/figures/year_compare/.
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent
CACHE = REPO / "paper_eval/cache_oe3i7whz"
DIAG = REPO / "paper_eval/figures/year_diag"
OUT = REPO / "paper_eval/figures/year_compare"
YEAR = {"window": "yr1win01", "control": "yr1ctl01", "frcadd04": "yr1add04",
        "ykwx7bl2": "yr1ykwx7"}
TRAINED = {"window": "frcwin01", "control": "frcctl01", "frcadd04": "frcadd04"}
REF = "frcadd04"
DEEP = ("temp_450", "salt_450", "salt_107")
COLOURS = dict(zip(YEAR, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"), strict=True))
CLIM_COLOUR, INK, INK2, GRID, SURFACE = "#52514e", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def scores(run, stream):
    path = CACHE / run / f"scores_FESOM_{stream}.nc"
    return xr.open_dataset(path).score.sel(region="global").isel(sample=0)


def clim_ratio(s):
    """Channel-mean RMSE / climatology RMSE, per lead."""
    rmse = s.sel(metric="rmse")
    return (rmse.sel(model="emulator") / rmse.sel(model="climatology")).mean("channel")


def last_validation(run):
    path = REPO / "results" / run / f"{run}_train_metrics.json"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    val = [r for r in rows if r.get("stage") == "val"][-1]
    return val["LossPhysical.FESOM_NODE.mse.avg"], val["LossPhysical.FESOM_ELEM.mse.avg"]


def main():
    S = {(k, st): scores(r, st) for k, r in YEAR.items() for st in ("NODE", "ELEM")}
    last = int(S[(REF, "NODE")].lead.max())
    ratio = {key: clim_ratio(s) for key, s in S.items()}

    print(f"1. Drift: channel-mean RMSE / climatology RMSE (global), day 180 and day {last}")
    for st in ("NODE", "ELEM"):
        for k in YEAR:
            r = ratio[(k, st)]
            print(f"   {st:4} {k:9} d180 {float(r.sel(lead=180)):5.2f}   "
                  f"d{last} {float(r.sel(lead=last)):5.2f}")

    print("\n2. Deep fields: RMSE, persistence, share of frcadd04's gap to persistence closed")
    for field in DEEP:
        for lead in (180, last):
            ref = float(S[(REF, "NODE")].sel(model="emulator", metric="rmse",
                                             channel=field, lead=lead))
            pers = float(S[(REF, "NODE")].sel(model="persistence", metric="rmse",
                                              channel=field, lead=lead))
            row = f"   {field:9} d{lead:<4} pers {pers:.4f} |"
            for k in YEAR:
                v = float(S[(k, "NODE")].sel(model="emulator", metric="rmse",
                                             channel=field, lead=lead))
                closed = (ref - v) / (ref - pers) if ref != pers else np.nan
                row += f" {k} {v:.4f} ({100 * closed:+.0f}%)"
            print(row)

    print("\n3. Short range (limit: at most ~5 % worse than frcadd04)")
    ref_node, ref_elem = last_validation(TRAINED[REF])
    for k, run in TRAINED.items():
        node, elem = last_validation(run)
        first30 = {st: float(ratio[(k, st)].sel(lead=slice(1, 30)).mean()
                             / ratio[(REF, st)].sel(lead=slice(1, 30)).mean())
                   for st in ("NODE", "ELEM")}
        print(f"   {k:9} 15-step val MSE / frcadd04: NODE {node / ref_node:5.3f} "
              f"ELEM {elem / ref_elem:5.3f} | days 1-30 RMSE/clim vs frcadd04: "
              f"NODE {first30['NODE']:5.3f} ELEM {first30['ELEM']:5.3f}")

    print("\n4. Variability ratio, leads 30 to the end (target about 0.7-1.3)")
    for k, run in YEAR.items():
        d = xr.open_dataset(DIAG / f"diag_{run}.nc")["variability_ratio"].sel(lead=slice(30, None))
        lows = d.min("lead")
        highs = d.max("lead")
        cells = " ".join(f"{f} {float(lows.sel(field=f)):.2f}-{float(highs.sel(field=f)):.2f}"
                         for f in d.field.values)
        print(f"   {k:9} {cells}")

    plot(S, ratio)


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(bottom=0)


def plot(S, ratio):
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, st in zip(axes, ("NODE", "ELEM"), strict=True):
        for k in YEAR:
            r = ratio[(k, st)]
            ax.plot(r.lead, r, color=COLOURS[k], lw=2, label=k)
            ax.annotate(k, (float(r.lead[-1]), float(r[-1])), xytext=(4, 0),
                        textcoords="offset points", color=INK2, fontsize=8, va="center")
        ax.axhline(1, color=CLIM_COLOUR, lw=1.5, ls=(0, (4, 3)), label="climatology (= 1)")
        ax.set_title(f"FESOM_{st}: channel mean", loc="left")
        ax.set_xlabel("lead time (days)")
        _style(ax)
    axes[0].set_ylabel("RMSE / climatology RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("One-year forecast from 2209-01-01: RMSE relative to climatology",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(OUT / "rmse_vs_clim_mean.png", dpi=160)
    plt.close(fig)

    panels = [("NODE", "sst"), ("NODE", "ssh"), ("ELEM", "u_2"), ("NODE", "temp_107"),
              ("NODE", "salt_107"), ("NODE", "temp_450"), ("NODE", "salt_450"),
              ("ELEM", "u_107"), ("NODE", "sss")]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    for ax, (st, ch) in zip(axes.flat, panels, strict=True):
        for k in YEAR:
            s = S[(k, st)].sel(model="emulator", metric="rmse", channel=ch)
            ax.plot(s.lead, s, color=COLOURS[k], lw=1.8, label=k)
        clim = S[(REF, st)].sel(model="climatology", metric="rmse", channel=ch)
        ax.plot(clim.lead, clim, color=CLIM_COLOUR, lw=1.4, ls=(0, (4, 3)), label="climatology")
        pers = S[(REF, st)].sel(model="persistence", metric="rmse", channel=ch)
        ax.plot(pers.lead, pers, color=CLIM_COLOUR, lw=1.0, ls=(0, (1, 2)), label="persistence")
        ax.set_title(ch, loc="left")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle("One-year forecast from 2209-01-01: global RMSE", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.savefig(OUT / "rmse_channels.png", dpi=160)
    plt.close(fig)
    print(f"\nfigures: {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it and look at the figures**

Run: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python compare_year.py`
Expected: four tables and two figures. Open both PNGs and check for label collisions and legend overlap.

- [ ] **Step 4: Judge and report.** Compare each table with criteria 1-4 of the spec, and note which criteria the window run meets and by how much compared with the control. Write the outcome and all job ids into the ledger, then report to the user:
  - the four criteria tables, in short form;
  - the figures' paths;
  - whether the spin-up helped beyond the control;
  - recommended next step: sub-project C or D, a longer prefix, or a held-out decade.
