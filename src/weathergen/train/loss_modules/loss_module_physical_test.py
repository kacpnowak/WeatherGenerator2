import pytest
import torch
from omegaconf import OmegaConf

from weathergen.train.loss_modules.loss_module_physical import DynamicLossEMA

_STREAMS = OmegaConf.create({"FESOM_NODES": {"train_target_channels": ["sss", "temp_deep"]}})


def _ema(window: int = 10, clamp_ratio: float = 20.0) -> DynamicLossEMA:
    return DynamicLossEMA({"window": window, "L": clamp_ratio}, _STREAMS, "cpu")


def test_update_ignores_channels_without_valid_points():
    ema = _ema()

    # temp_deep is fill-only: lp_loss reports exactly 0 for it. Feeding that into the
    # inverse-MSE EMA would pin it at a huge weight and starve every other channel.
    ema.update("FESOM_NODES", torch.tensor([0.5, 0.0]))

    assert ema.channel_weights_ema["FESOM_NODES"][1].item() == 1.0


def test_update_of_observed_channel_still_moves_the_ema():
    ema = _ema(window=10)

    ema.update("FESOM_NODES", torch.tensor([0.5, 0.0]))

    # (1 - 1/10) * 1.0 + (1/10) * (1 / 0.5)
    assert ema.channel_weights_ema["FESOM_NODES"][0].item() == pytest.approx(1.1)


def test_fill_only_channel_does_not_suppress_the_weights_of_observed_channels():
    ema = _ema(window=10)

    for _ in range(100):
        ema.update("FESOM_NODES", torch.tensor([0.5, 0.0]))
    weights = ema.get_weights("FESOM_NODES", None)

    # Without the skip, temp_deep's EMA runs away to ~1e6, the clamp pins it at
    # L * min = 20, and the mean-1 normalisation pushes sss far below 1.
    assert weights[0].item() > 0.5


def test_update_ignores_nan_channel_losses():
    ema = _ema()

    ema.update("FESOM_NODES", torch.tensor([0.5, float("nan")]))

    assert ema.channel_weights_ema["FESOM_NODES"][1].item() == 1.0
