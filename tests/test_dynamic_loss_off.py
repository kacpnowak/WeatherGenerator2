import torch
from omegaconf import OmegaConf

from weathergen.train.loss_modules.loss_module_physical import DynamicLossEMA

STREAMS = OmegaConf.create({"GLORYS3": {"train_target_channels": ["zos", "thetao_0.494025m", "so_0.494025m"]}})


def test_L_equal_one_yields_exactly_the_static_weights():
    ema = DynamicLossEMA({"window": 128, "L": 1.0}, STREAMS, "cpu")
    ema.channel_weights_ema["GLORYS3"] = torch.tensor([0.5, 3.0, 0.01])  # unequal running errors
    static = torch.tensor([6.91, 2.44, 1.16])
    w = ema.get_weights("GLORYS3", static)
    assert torch.equal(w, static)


def test_L_equal_one_without_static_is_all_ones():
    ema = DynamicLossEMA({"window": 128, "L": 1.0}, STREAMS, "cpu")
    ema.channel_weights_ema["GLORYS3"] = torch.tensor([0.5, 3.0, 0.01])
    w = ema.get_weights("GLORYS3", None)
    assert torch.equal(w, torch.ones(3))


def test_L_above_one_still_reweights():
    ema = DynamicLossEMA({"window": 128, "L": 5.0}, STREAMS, "cpu")
    ema.channel_weights_ema["GLORYS3"] = torch.tensor([0.5, 3.0, 0.01])
    w = ema.get_weights("GLORYS3", None)
    assert not torch.equal(w, torch.ones(3))
