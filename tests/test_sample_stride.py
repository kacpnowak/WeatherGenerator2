"""sample_stride thins the unshuffled start dates (validation spread over the year).

The pure helper is tested directly. The sampler tests need the Levante FESOM mesh and are
skipped without it; no sample data is loaded, the batch builders are replaced by stand-ins
that record the start index they were called with.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from weathergen.common import config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler, stride_perms
from weathergen.train.utils import VAL, cfg_keys_to_filter, get_active_stage_config

MESH = Path(
    "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
)
needs_mesh = pytest.mark.skipif(not MESH.exists(), reason="Levante FESOM mesh not available")


def test_stride_takes_every_nth():
    assert stride_perms(np.arange(350), 5, False).tolist() == list(range(0, 350, 5))


def test_stride_one_is_identity():
    p = np.arange(17)
    assert stride_perms(p, 1, True) is p


def test_stride_rejects_shuffle_and_bad_values():
    with pytest.raises(ValueError):
        stride_perms(np.arange(9), 2, True)
    with pytest.raises(ValueError):
        stride_perms(np.arange(9), 0, False)


class _RecordingBatch:
    """Picklable stand-in for a ModelBatch: always valid, remembers its start index."""

    def __init__(self, idx):
        self.record = int(idx)

    def sources_empty(self):
        return False

    def is_nan(self):
        return False

    def targets_empty(self):
        return False


def new_val_sampler(val_overlay: dict) -> MultiStreamDataSampler:
    cf = config.load_merge_configs(
        None,
        None,
        None,
        None,
        "config/config_fesom_vars.yml",
        {"streams_directory": "./config/streams/fvars46_levante/"},
    )
    cf.data_loading.rng_seed = 7
    cf.rank, cf.world_size = 0, 1
    val_cfg = get_active_stage_config(
        cf.training_config, cf.get("validation_config", {}), cfg_keys_to_filter
    )
    val_cfg.update(val_overlay)
    sampler = MultiStreamDataSampler(cf, val_cfg, stage=VAL)
    sampler._get_batch = lambda idx, n: _RecordingBatch(idx)
    sampler._get_prefix_batch = lambda idx, n, p: _RecordingBatch(idx)
    return sampler


@needs_mesh
def test_validation_stride_spreads_dates_over_the_year():
    sampler = new_val_sampler({"samples_per_mini_epoch": 64, "shuffle": False, "sample_stride": 5})
    loader = torch.utils.data.DataLoader(sampler, batch_size=None, num_workers=2)
    epochs = []
    for _ in range(2):
        epochs.append([b.record for b in loader])
        sampler.advance()
    assert sorted(epochs[0]) == list(range(0, 320, 5))
    assert epochs[1] == epochs[0]


@needs_mesh
def test_stride_leaving_too_few_dates_raises():
    with pytest.raises(ValueError, match="sample_stride"):
        new_val_sampler({"samples_per_mini_epoch": 64, "shuffle": False, "sample_stride": 6})
