"""
A resumed training run continues the sample sequence of an uninterrupted one.

Upstream ecmwf/WeatherGenerator#2953 / PR #2954: `train_continue` reuses the run's rng seed, so
the sampler has to be told the mini-epoch it resumes in (its worker seed, the forecast-step
curriculum and the spin-up draw depend on it). And a run resumed from a mid-epoch `_latest`
checkpoint has to train only the batches of the interrupted mini-epoch that are not in the
checkpoint yet, the same ones, in the same order, as an uninterrupted run.

The real sampler iteration (`reset`, `worker_workset`, `__iter__`) and the real training loop
(`Trainer.train`) run here, through a real torch DataLoader, with `_get_batch` /
`_get_prefix_batch` replaced by stubs that return the sample index, the forecast steps and the
spin-up steps instead of loading data.
"""

import dataclasses
import itertools
import logging
import types
from typing import NamedTuple

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import weathergen.train.trainer as trainer_module
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.train.trainer import Trainer


class _Item(NamedTuple):
    """What a batch would be built from; doubles as the batch in the training loop."""

    idx: int
    forecast_steps: int
    prefix_steps: int

    def sources_empty(self):
        return False

    def is_nan(self):
        return False

    def targets_empty(self):
        return False

    def to_device(self, device):
        return self

    def get_source_samples(self):
        return self

    def get_output_len(self):
        return self.forecast_steps


class _NoTokenizer:
    def reset_rng(self, rng):
        pass


@dataclasses.dataclass
class _IndexRange:
    start: int
    end: int


class _StubSampler(MultiStreamDataSampler):
    """
    The real sampler without data streams: daily windows over `num_times` days, forecast
    steps drawn per batch (`sequential_random` over a curriculum, so they depend on the
    mini-epoch and the worker seed) and a spin-up draw in [0, 90].
    """

    def __init__(
        self,
        samples_per_mini_epoch,
        num_workers,
        world_size=1,
        rank=0,
        seed=1234,
        num_times=400,
    ):
        torch.utils.data.IterableDataset.__init__(self)
        self.mode_cfg = {"training_mode": ["masking"]}
        self.mini_epoch = 0
        self.skip_batches = 0
        self.rank = rank
        self.world_size = world_size
        self.repeat_data = False
        self.tokenizer = _NoTokenizer()
        self.output_offset = 1
        self.time_step = np.timedelta64(1, "D")
        self.step_timedelta = np.timedelta64(1, "D")
        self.len_timedelta = np.timedelta64(1, "D")
        self.forecast_policy = "sequential_random"
        self.list_num_forecast_steps = np.array([2, 3, 4, 5], dtype=np.int32)
        self._prefix_range = (0, 90)
        self._lazy_targets = False
        self.batch_size = 1
        self.shuffle = True
        self.index_range = _IndexRange(0, num_times)
        self.samples_per_mini_epoch = samples_per_mini_epoch
        self.check_samples(self._get_fsm() + self._prefix_range[1])
        self.data_loader_rng_seed = seed if seed > num_workers else seed * 97
        self._prefix_seed = seed
        self.rng = None

    def _get_batch(self, idx, num_forecast_steps):
        return _Item(int(idx), int(num_forecast_steps), 0)

    def _get_prefix_batch(self, idx, num_forecast_steps, prefix_steps):
        return _Item(int(idx), int(num_forecast_steps), int(prefix_steps))


def _loader(sampler, num_workers):
    # as the trainer builds its training loader
    return torch.utils.data.DataLoader(
        sampler, batch_size=None, batch_sampler=None, shuffle=False, num_workers=num_workers
    )


def _uninterrupted(make_sampler, num_workers, num_mini_epochs):
    """Batches of each mini-epoch of an uninterrupted run."""
    sampler = make_sampler()
    mini_epochs = []
    for _ in range(num_mini_epochs):
        mini_epochs.append(list(_loader(sampler, num_workers)))
        sampler.advance()
    return mini_epochs


def _resuming_trainer(sampler, istep, samples_per_mini_epoch, world_size_original=None):
    """The state `Trainer.run` has when it works out where a continued run resumes."""
    trainer = Trainer(OmegaConf.create({"terminal": 10, "metrics": 20, "checkpoint": 64}))
    trainer.cf = OmegaConf.create(
        {"general": {"istep": istep}, "world_size": sampler.world_size}
    )
    trainer.dataset = sampler
    trainer.batch_size_per_gpu = 1
    trainer.training_cfg = OmegaConf.create({"samples_per_mini_epoch": samples_per_mini_epoch})
    trainer.world_size_original = (
        sampler.world_size if world_size_original is None else world_size_original
    )
    return trainer


# (num_workers, samples_per_mini_epoch): the batches of a rank are split over the loader
# workers, the last worker takes the remainder: 11 over 3 workers is 3, 3, 5 and 10 over 4
# workers is 2, 2, 2, 4, so the last worker runs alone at the end of the mini-epoch
LOADERS = [(0, 12), (2, 12), (3, 11), (4, 10)]


def test_the_stub_exercises_forecast_steps_and_spin_up():
    mini_epochs = _uninterrupted(lambda: _StubSampler(12, 2), 2, 3)
    items = [item for mini_epoch in mini_epochs for item in mini_epoch]
    assert len({item.forecast_steps for item in items}) > 1
    assert len({item.prefix_steps for item in items}) > 5
    # the curriculum (and with workers the seed) changes between mini-epochs
    assert mini_epochs[0] != mini_epochs[1] != mini_epochs[2]


@pytest.mark.parametrize("num_workers, samples_per_mini_epoch", LOADERS)
def test_resume_at_a_mini_epoch_boundary_continues_the_sequence(
    num_workers, samples_per_mini_epoch
):
    def make():
        return _StubSampler(samples_per_mini_epoch, num_workers)

    reference = _uninterrupted(make, num_workers, 4)
    num_batches = len(reference[0])
    for k in (1, 2, 3):
        sampler = make()
        trainer = _resuming_trainer(sampler, k * num_batches, samples_per_mini_epoch)
        assert trainer._align_sampler_with_resume() == k
        assert sampler.skip_batches == 0
        assert list(_loader(sampler, num_workers)) == reference[k]


@pytest.mark.parametrize("num_workers, samples_per_mini_epoch", LOADERS)
def test_mid_epoch_resume_trains_the_remaining_batches(num_workers, samples_per_mini_epoch):
    def make():
        return _StubSampler(samples_per_mini_epoch, num_workers)

    reference = _uninterrupted(make, num_workers, 4)
    num_batches = len(reference[0])
    k = 2
    # every position, so also positions not divisible by num_workers and positions in the
    # part of the mini-epoch where only the last worker is left
    for s in range(1, num_batches):
        sampler = make()
        trainer = _resuming_trainer(sampler, k * num_batches + s, samples_per_mini_epoch)
        assert trainer._align_sampler_with_resume() == k
        assert sampler.skip_batches == s
        assert list(_loader(sampler, num_workers)) == reference[k][s:], f"resumed after {s}"

        # the next mini-epoch is a complete one again
        sampler.advance()
        assert sampler.skip_batches == 0
        assert list(_loader(sampler, num_workers)) == reference[k + 1], f"resumed after {s}"


def test_mid_epoch_resume_on_another_rank():
    # rank 1 of 2: 24 samples per mini-epoch give each rank 12, split 4, 4, 4 over 3 workers
    def make():
        return _StubSampler(24, 3, world_size=2, rank=1)

    reference = _uninterrupted(make, 3, 3)
    assert len(reference[0]) == 12
    for s in (1, 5, 7, 11):
        sampler = make()
        trainer = _resuming_trainer(sampler, 12 + s, 24)
        assert trainer._align_sampler_with_resume() == 1
        assert list(_loader(sampler, 3)) == reference[1][s:]


@pytest.mark.parametrize("skip", [64, 65, 131])
def test_mid_epoch_resume_with_the_fvars_layout(skip):
    # the fesom_vars runs: 8 ranks, 2048 samples per mini-epoch, 8 loader workers per rank,
    # a `_latest` checkpoint every 64 steps
    def make():
        return _StubSampler(2048, 8, world_size=8, rank=5, num_times=3000)

    reference = _uninterrupted(make, 8, 3)
    assert len(reference[0]) == 256
    sampler = make()
    trainer = _resuming_trainer(sampler, 256 + skip, 2048)
    assert trainer._align_sampler_with_resume() == 1
    assert list(_loader(sampler, 8)) == reference[1][skip:]
    sampler.advance()
    assert list(_loader(sampler, 8)) == reference[2]


# ----------------------------------------------------------------------------------------------
# torch DataLoader order and the per-worker split of the skipped batches


class _WorkerTagged(torch.utils.data.IterableDataset):
    """Yields (worker id, item number) with an uneven number of items per worker."""

    def __init__(self, counts):
        super().__init__()
        self.counts = counts

    def __iter__(self):
        worker = torch.utils.data.get_worker_info().id
        for j in range(self.counts[worker]):
            yield (worker, j)


def _round_robin(counts):
    for j in range(max(counts)):
        for worker, count in enumerate(counts):
            if j < count:
                yield (worker, j)


@pytest.mark.parametrize("counts", [[3, 3, 5], [2, 2, 2, 4], [0, 0, 3], [4, 4]])
def test_torch_loader_returns_worker_batches_round_robin(counts):
    # the semantics the resume split relies on: in-order round robin over the workers from
    # worker 0, exhausted workers drop out, no reordering by worker speed
    loaded = [tuple(int(v) for v in item) for item in _loader(_WorkerTagged(counts), len(counts))]
    assert loaded == list(_round_robin(counts))


@pytest.mark.parametrize("counts", [[3, 3, 5], [2, 2, 2, 4], [0, 0, 3], [4, 4], [7]])
def test_resume_worker_skips_follow_the_round_robin(counts):
    from weathergen.datasets.multi_stream_data_sampler import resume_worker_skips

    order = list(_round_robin(counts))
    # without a skip the workers keep their roles
    assert resume_worker_skips(counts, 0) == ([0] * len(counts), 0)
    for num_skip in range(1, len(order)):
        consumed, next_worker = resume_worker_skips(counts, num_skip)
        assert consumed == [
            sum(1 for worker, _ in order[:num_skip] if worker == w) for w in range(len(counts))
        ]
        assert next_worker == order[num_skip][0]


# ----------------------------------------------------------------------------------------------
# trainer: where a continued run resumes


@pytest.mark.parametrize(
    "istep, expected",
    [
        (0, (0, 0)),
        (1, (0, 1)),
        (64, (0, 64)),
        (255, (0, 255)),
        (256, (1, 0)),
        (3 * 256 + 64, (3, 64)),
        (3 * 256 + 65, (3, 65)),
        (9 * 256 + 255, (9, 255)),
        (10 * 256, (10, 0)),
    ],
)
def test_resume_position_with_the_fvars_layout(istep, expected):
    from weathergen.train.trainer import resume_position

    # per rank 2048 / 8 = 256 batches per mini-epoch
    assert resume_position(istep, 256, 1, 2048, 8, 8) == expected


def test_resume_position_single_rank():
    from weathergen.train.trainer import resume_position

    assert resume_position(0, 12, 1, 12, 1, 1) == (0, 0)
    assert resume_position(29, 12, 1, 12, 1, 1) == (2, 5)
    # no world_size_original (not reachable: init_ddp always sets world_size)
    assert resume_position(29, 12, 1, 12, None, 1) == (2, 5)


def test_resume_position_does_not_skip_after_a_world_size_change(caplog):
    from weathergen.train.trainer import resume_position

    # the run started on 8 ranks and continues on 4: mini_epoch_base as before, no skip
    with caplog.at_level(logging.WARNING):
        assert resume_position(3 * 512 + 70, 512, 1, 2048, 8, 4) == (3, 0)
    assert "world size" in caplog.text


def test_resume_position_does_not_skip_when_mini_epoch_lengths_disagree(caplog):
    from weathergen.train.trainer import resume_position

    # 22 samples on 2 ranks: 11 batches per rank, but the mini_epoch_base formula counts
    # (11 // 2) * 2 = 10 steps per mini-epoch; the position in the mini-epoch is then unknown
    with caplog.at_level(logging.WARNING):
        mini_epoch_base, skip = resume_position(25, 11, 1, 22, 2, 2)
    assert (mini_epoch_base, skip) == (2, 0)
    assert "skip" in caplog.text


# ----------------------------------------------------------------------------------------------
# training loop: a chain of jobs, each killed after some steps, trains every step once


class _JobEndError(Exception):
    """The job reached its time limit."""


class _StepBudget:
    """Learning-rate scheduler stand-in that ends the job after `budget` optimizer steps."""

    def __init__(self, budget):
        self.budget = budget

    def step(self):
        self.budget -= 1
        if self.budget == 0:
            raise _JobEndError

    def get_lr(self):
        return 0.0


class _RecordingModel(torch.nn.Module):
    def __init__(self, trained):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.trained = trained

    def forward(self, params, x, output_len):
        self.trained.append(x)
        return self.weight * 1.0


def _training_trainer(monkeypatch, sampler, num_workers, istep, trained, checkpoint, log):
    """A Trainer whose `train` runs for real over the stub sampler, with stub model and loss."""
    spme = sampler.samples_per_mini_epoch
    trainer = _resuming_trainer(sampler, istep, spme)
    trainer.train_logging = OmegaConf.create(
        {"terminal": 1000, "metrics": 1000, "checkpoint": checkpoint}
    )
    trainer.cf = OmegaConf.create(
        {
            "general": {"istep": istep},
            "world_size": 1,
            "freeze_modules": "",
            "data_loading": {},
            "local_rank": 0,
            "with_mixed_precision": False,
        }
    )
    trainer.training_cfg = OmegaConf.create(
        {"samples_per_mini_epoch": spme, "optimizer": {"grad_clip": 1.0}}
    )
    trainer.data_loader = _loader(sampler, num_workers)
    trainer.model = _RecordingModel(trained)
    trainer.mixed_precision_dtype = torch.bfloat16
    trainer.target_and_aux_calculators = {}
    trainer.target_and_aux_calculators_val = {}
    trainer.loss_calculator = types.SimpleNamespace(
        compute_loss=lambda preds, targets_and_aux, metadata: (preds**2).sum()
    )
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
    trainer.log_grad_norms = False
    trainer.collapse_monitor = types.SimpleNamespace(
        should_compute=lambda istep: False, should_log=lambda istep: False
    )
    monkeypatch.setattr(trainer_module, "extract_batch_metadata", lambda batch: None)
    monkeypatch.setattr(
        trainer,
        "_log_terminal",
        lambda bidx, mini_epoch, stage: log.append(
            (mini_epoch, bidx, int(trainer.cf.general.istep), trained[-1])
        ),
    )
    monkeypatch.setattr(trainer, "_log", lambda stage: None)
    return trainer


@pytest.mark.parametrize("num_workers", [0, 3])
def test_chained_jobs_train_every_planned_step_once(monkeypatch, num_workers):
    num_mini_epochs, checkpoint = 3, 4

    def make():
        return _StubSampler(11, num_workers)

    reference = _uninterrupted(make, num_workers, num_mini_epochs)
    num_batches = len(reference[0])
    planned = num_mini_epochs * num_batches

    latest = (0, [])  # step counter and trained batches in the `_latest` checkpoint
    budgets = iter([6, 9, 7, 10, 1000])
    while True:
        istep, lineage = latest
        trained = list(lineage)
        log = []
        trainer = _training_trainer(
            monkeypatch, make(), num_workers, istep, trained, checkpoint, log
        )
        trainer.lr_scheduler = _StepBudget(next(budgets))
        saves = []

        def save_model(mini_epoch, name=None, trainer=trainer, trained=trained, saves=saves):
            saves.append((mini_epoch, int(trainer.cf.general.istep), list(trained)))

        monkeypatch.setattr(trainer, "save_model", save_model)

        try:
            # as Trainer.run, without validation
            mini_epoch_base = trainer._align_sampler_with_resume()
            for mini_epoch in range(mini_epoch_base, num_mini_epochs):
                trainer.train(mini_epoch)
                trainer.save_model(mini_epoch)
            done = True
        except _JobEndError:
            done = False

        # the logged position in the mini-epoch matches the step counter and the batch
        for mini_epoch, bidx, step, item in log:
            assert step == mini_epoch * num_batches + bidx
            assert item == reference[mini_epoch][bidx]

        # `_latest` checkpoints: at the usual positions, step counter = steps in the model
        for mini_epoch, step, saved in saves:
            if mini_epoch == -1:
                assert step % num_batches % checkpoint == 1
            assert len(saved) == step
        mid_epoch_saves = [save for save in saves if save[0] == -1]
        if mid_epoch_saves:
            latest = mid_epoch_saves[-1][1:]
        if done:
            break

    assert trainer.cf.general.istep == planned
    assert trained == list(itertools.chain.from_iterable(reference))
