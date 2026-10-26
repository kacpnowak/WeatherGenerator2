"""Real-data check that a sampler resumed mid-mini-epoch continues the uninterrupted sequence.

Skipped where the AWI-CM3 FESOM data is absent. No sample data is loaded: the batch builders
are replaced by stand-ins that record the (idx, num_forecast_steps, prefix_steps) they were
called with, so only the sample, forecast-step and spin-up sequences are compared.
"""

from pathlib import Path

import pytest
import torch

from weathergen.common import config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.train.utils import TRAIN, cfg_keys_to_filter, filter_config_by_enabled

DATA_ROOT = Path("/e/data1/climateai/hclimrep/data/ifs-fesom")

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "ocean_node").exists(), reason="AWI-CM3 FESOM data not available"
)


class _RecordingBatch:
    """Picklable stand-in for a ModelBatch: always valid, remembers what it was built from."""

    def __init__(self, idx, num_forecast_steps, prefix_steps):
        self.record = (int(idx), int(num_forecast_steps), int(prefix_steps))

    def sources_empty(self):
        return False

    def is_nan(self):
        return False

    def targets_empty(self):
        return False


def new_sampler() -> MultiStreamDataSampler:
    cf = config.load_merge_configs(
        None,
        None,
        None,
        None,
        "config/config_fesom_vars.yml",
        {"training_config": {"samples_per_mini_epoch": 32, "forecast": {"prefix_steps": [0, 90]}}},
    )
    # runtime fields run_train (seed) and Trainer.init_ddp (rank, world size) set
    cf.data_loading.rng_seed = 7
    cf.rank, cf.world_size = 0, 1
    training_cfg = filter_config_by_enabled(cf.training_config, cfg_keys_to_filter)
    sampler = MultiStreamDataSampler(cf, training_cfg, stage=TRAIN)
    sampler._get_batch = lambda idx, n: _RecordingBatch(idx, n, 0)
    sampler._get_prefix_batch = lambda idx, n, p: _RecordingBatch(idx, n, p)
    return sampler


def run_epochs(sampler: MultiStreamDataSampler, epochs: int) -> list[list[tuple[int, int, int]]]:
    """Records of every batch the trainer receives, per mini-epoch, as the trainer loops."""
    loader = torch.utils.data.DataLoader(sampler, batch_size=None, num_workers=2)
    records = []
    for _ in range(epochs):
        records.append([batch.record for batch in loader])
        sampler.advance()
    return records


def test_resumed_mid_epoch_sees_the_uninterrupted_sequence():
    full = run_epochs(new_sampler(), epochs=3)
    s = new_sampler()
    s.set_resume_position(1, 5)
    resumed = run_epochs(s, epochs=2)
    assert resumed[0] == full[1][5:]
    assert resumed[1] == full[2]


def test_without_resume_position_nothing_changes():
    assert run_epochs(new_sampler(), 1) == run_epochs(new_sampler(), 1)
