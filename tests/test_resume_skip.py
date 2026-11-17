"""Resuming the sampler mid-mini-epoch: which batches each loader worker skips (no data)."""

import types

import pytest
import torch

from weathergen.datasets.multi_stream_data_sampler import (
    MultiStreamDataSampler,
    resumed_worker_assignments,
    worker_ranges,
    worker_skip_counts,
)


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
    s = MultiStreamDataSampler.__new__(MultiStreamDataSampler)
    s.mini_epoch = 0
    s.set_resume_position(3, 17)
    assert (s.mini_epoch, s.skip_steps) == (3, 17)
    s.advance()
    assert (s.mini_epoch, s.skip_steps) == (4, 0)


@pytest.mark.parametrize(
    "worker, workset, seed",
    [
        (None, (0, 32, 5), 1),  # single process: skips its own first 5 batches
        (0, (16, 32, 2), 2 * 37 * 13 * 7),  # takes over worker 1: its range and its seed
        (1, (0, 16, 3), 1 * 37 * 13 * 7),  # takes over worker 0
    ],
)
def test_resumed_worker_runs_the_worker_it_takes_over(monkeypatch, worker, workset, seed):
    info = None if worker is None else types.SimpleNamespace(id=worker, num_workers=2)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)
    s = MultiStreamDataSampler.__new__(MultiStreamDataSampler)
    s.rank, s.world_size, s.len, s.batch_size, s.data_loader_rng_seed = 0, 1, 32, 1, 1
    s.set_resume_position(0, 5)
    assert s.worker_workset() == workset
    assert s.data_loader_rng_seed == seed


def test_resumed_workers_take_over_in_rotation_order():
    # w0 w1 w0 w1 w0 | w1 w0 w1: step 5 comes from w1, but a fresh loader starts at worker 0
    assert resumed_worker_assignments(5, [4, 4]) == [(1, 2), (0, 3)]


def test_resume_on_a_round_boundary_keeps_the_workers():
    assert resumed_worker_assignments(4, [4, 4]) == [(0, 2), (1, 2)]


def test_resume_with_uneven_workers():
    # w0 w1 w2 w0 | w1 w2 w1 w2 w2: w0 is done, so the third resumed worker has nothing left
    assert resumed_worker_assignments(4, [2, 3, 4]) == [(1, 1), (2, 1), None]


def test_resume_with_a_worker_without_batches():
    # w1 | w2 w1 w2 w1 w2
    assert resumed_worker_assignments(1, [0, 3, 3]) == [(2, 0), (1, 1), None]
    # fewer samples than workers: only the last worker has batches
    assert resumed_worker_assignments(2, [0] * 7 + [5]) == [(7, 2)] + [None] * 7


def test_resume_past_the_end_leaves_nothing():
    assert resumed_worker_assignments(8, [4, 4]) == [None, None]


def _delivered(worker_batches: list[list]) -> list:
    """What the DataLoader returns: round-robin over the workers that still have batches."""
    out = []
    for k in range(max(map(len, worker_batches), default=0)):
        out += [batches[k] for batches in worker_batches if k < len(batches)]
    return out


@pytest.mark.parametrize("lengths", [[4, 4], [2, 5], [2, 3, 4], [0, 3, 3], [0] * 7 + [5]])
def test_resumed_loader_continues_the_uninterrupted_order(lengths):
    full = _delivered([[(w, k) for k in range(n)] for w, n in enumerate(lengths)])
    for steps in range(len(full) + 1):
        resumed = _delivered(
            [
                [] if a is None else [(a[0], k) for k in range(a[1], lengths[a[0]])]
                for a in resumed_worker_assignments(steps, lengths)
            ]
        )
        assert resumed == full[steps:], steps
