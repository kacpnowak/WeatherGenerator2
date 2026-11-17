"""Optimizer and dynamic-loss state saved next to a model checkpoint and restored on resume."""

import inspect
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from weathergen.train.loss_modules.loss_module_physical import DynamicLossEMA
from weathergen.train.train_state import (
    gather_optimizer_state,
    load_optimizer_state,
    load_train_state,
    save_train_state,
    should_restore_train_state,
    train_state_path,
)
from weathergen.train.trainer import Trainer

streams = OmegaConf.create(
    {
        "TEMP": {"train_target_channels": ["temp_0", "temp_1"]},
        "SALT": {"train_target_channels": ["salt_0", "salt_1", "salt_2"]},
    }
)


def _stepped_optimizer(lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95)):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)
    model(torch.randn(4, 3)).square().sum().backward()
    optimizer.step()
    return model, optimizer


def test_dynamic_loss_round_trip():
    ema = DynamicLossEMA({"window": 4, "L": 20.0}, streams, "cpu")
    ema.update("TEMP", torch.tensor([0.5, 2.0]))
    fresh = DynamicLossEMA({"window": 4, "L": 20.0}, streams, "cpu")
    fresh.load_state_dict(ema.state_dict())
    assert torch.equal(fresh.channel_weights_ema["TEMP"], ema.channel_weights_ema["TEMP"])
    assert not torch.equal(fresh.channel_weights_ema["TEMP"], torch.ones(2))


def test_disabled_dynamic_loss_has_empty_state():
    assert DynamicLossEMA(None, streams, "cpu").state_dict() == {}


def test_dynamic_loss_ignores_unknown_and_keeps_missing_streams():
    fresh = DynamicLossEMA({"window": 4, "L": 20.0}, streams, "cpu")
    fresh.load_state_dict({"TEMP": torch.tensor([3.0, 4.0]), "UNKNOWN": torch.tensor([5.0])})
    assert torch.equal(fresh.channel_weights_ema["TEMP"], torch.tensor([3.0, 4.0]))
    assert torch.equal(fresh.channel_weights_ema["SALT"], torch.ones(3))
    assert "UNKNOWN" not in fresh.channel_weights_ema


def test_dynamic_loss_keeps_ones_on_channel_mismatch():
    fresh = DynamicLossEMA({"window": 4, "L": 20.0}, streams, "cpu")
    fresh.load_state_dict({"SALT": torch.tensor([3.0, 4.0])})
    assert torch.equal(fresh.channel_weights_ema["SALT"], torch.ones(3))


def test_optimizer_round_trip(tmp_path):
    model, optimizer = _stepped_optimizer()
    path = train_state_path(tmp_path, "run", -1)
    save_train_state(path, gather_optimizer_state(model, optimizer), {"physical": {}}, 7)

    state = load_train_state(path)
    assert state["istep"] == 7
    assert state["dynamic_loss"] == {"physical": {}}
    fresh_model = torch.nn.Linear(3, 2)
    fresh = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    load_optimizer_state(fresh_model, fresh, state["optimizer"])

    for param, fresh_param in zip(model.parameters(), fresh_model.parameters(), strict=True):
        saved, restored = optimizer.state[param], fresh.state[fresh_param]
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(restored[key], saved[key]), key
    assert not list(tmp_path.glob("*_tmp*"))


def test_scheduler_numpy_hyperparameters_load(tmp_path):
    # the lr scheduler leaves numpy scalars in the param groups, and they get saved with them
    model, optimizer = _stepped_optimizer(lr=np.float64(1e-3))
    path = train_state_path(tmp_path, "run", -1)
    save_train_state(path, gather_optimizer_state(model, optimizer), {}, 7)
    assert load_train_state(path)["istep"] == 7


def test_restore_parameters_that_never_had_a_gradient(tmp_path):
    # no AdamW state to save for those: the encoder after a spin-up, parameters nothing uses
    def build():
        torch.manual_seed(0)
        layers = {"used": torch.nn.Linear(3, 2), "unused": torch.nn.Linear(3, 2)}
        model = torch.nn.ModuleDict(layers)
        return model, torch.optim.AdamW(model.parameters(), lr=1e-3)

    model, optimizer = build()
    model["used"](torch.randn(4, 3)).square().sum().backward()
    optimizer.step()
    path = train_state_path(tmp_path, "run", -1)
    save_train_state(path, gather_optimizer_state(model, optimizer), {}, 7)
    assert len(optimizer.state) == 2

    fresh_model, fresh = build()
    load_optimizer_state(fresh_model, fresh, load_train_state(path)["optimizer"])

    saved, restored = optimizer.state[model["used"].weight], fresh.state[fresh_model["used"].weight]
    for key in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(restored[key], saved[key]), key
    unused = fresh.state[fresh_model["unused"].weight]
    assert unused["step"] == 0
    assert not unused["exp_avg"].any() and not unused["exp_avg_sq"].any()


def test_restore_keeps_hyperparameters_of_the_new_run(tmp_path):
    # lr comes from the scheduler and betas from the total batch size of the continued run
    model, optimizer = _stepped_optimizer(lr=1e-3, betas=(0.9, 0.95))
    fresh_model = torch.nn.Linear(3, 2)
    fresh = torch.optim.AdamW(fresh_model.parameters(), lr=5e-4, betas=(0.8, 0.9))
    load_optimizer_state(fresh_model, fresh, gather_optimizer_state(model, optimizer))
    assert fresh.param_groups[0]["lr"] == 5e-4
    assert fresh.param_groups[0]["betas"] == (0.8, 0.9)
    assert len(fresh.state) == 2


def test_missing_file_returns_none(tmp_path):
    assert load_train_state(tmp_path / "x_latest_train_state.pt") is None


def test_new_run_does_not_restore():
    assert should_restore_train_state(0) is False and should_restore_train_state(1) is True


def test_paths():
    assert train_state_path(Path("m"), "fvctrl01", -1).name == "fvctrl01_latest_train_state.pt"
    assert train_state_path(Path("m"), "fvctrl01", 3).name == "fvctrl01_chkpt00003_train_state.pt"
    assert train_state_path(Path("m"), "fvctrl01", 3).parent == Path("m")


def test_run_restores_after_loss_calculator_and_scheduler():
    # the dynamic loss lives in the loss calculator, and the scheduler sets the lr kept on restore
    source = inspect.getsource(Trainer.run)
    restore = source.index("self._restore_train_state(")
    assert source.index("self.loss_calculator = LossCalculator(") < restore
    assert source.index("LearningRateScheduler(") < restore
    assert restore < source.index("self.validate_before_training()")


def test_run_decides_on_the_saved_step_counter():
    # a world-size change can realign a resumed run's counter to 0, the saved istep cannot
    source = inspect.getsource(Trainer.run)
    assert "should_restore_train_state(istep_saved)" in source


def test_restore_log_line():
    # the resume smoke test greps the job log for this text
    assert "restored optimizer state" in inspect.getsource(Trainer._restore_train_state)


def test_save_model_writes_train_state_before_config():
    # the config carrying istep is written last, as for the model file
    source = inspect.getsource(Trainer.save_model)
    assert source.index("file_tmp.replace(file_out)") < source.index("save_train_state(")
    assert source.index("save_train_state(") < source.index("config.save(self.cf, mini_epoch)")
