import pytest
import torch
from omegaconf import OmegaConf

from weathergen.train.lr_scheduler import LearningRateScheduler


def make(lr_max, lr_steps, istep=0):  # world size 8, batch 1, the recipe's schedule
    cfg = OmegaConf.create(
        dict(
            lr_start=1e-6,
            lr_max=lr_max,
            lr_final_decay=2e-6,
            lr_final=0.0,
            num_steps_warmup=256,
            num_steps_cooldown=512,
            policy_warmup="cosine",
            policy_decay="cosine",
            policy_cooldown="linear",
            parallel_scaling_policy="sqrt",
        )
    )
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(3))], lr=1e-6)
    return opt, LearningRateScheduler(opt, 1, 8, istep, lr_steps, cfg)


def lrs(lr_max, lr_steps):
    opt, sch = make(lr_max, lr_steps)
    out = []
    for _ in range(lr_steps):
        out.append(opt.param_groups[0]["lr"])
        sch.step()
    return out


@pytest.mark.parametrize("lr_max,lr_steps", [(5e-5, 32768), (1e-5, 3072), (1e-5, 2560)])
def test_decay_ends_at_lr_final_decay(lr_max, lr_steps):
    v = lrs(lr_max, lr_steps)
    end_decay = lr_steps - 512
    assert max(v) == pytest.approx(lr_max * 8**0.5, rel=1e-3)
    assert v[end_decay - 1] == pytest.approx(2e-6 * 8**0.5, rel=2e-2)  # 5.66e-6, was 1.77e-7
    assert all(a >= b - 1e-15 for a, b in zip(v[256:], v[257:], strict=False))  # no rise after peak
    assert v[-1] < 2e-6 * 8**0.5 / 100  # cool-down reaches ~0


def test_resume_matches_continuous_schedule():
    v = lrs(1e-5, 2560)
    for k in (65, 256, 1281, 2048, 2300):
        opt, _ = make(1e-5, 2560, istep=k)
        assert opt.param_groups[0]["lr"] == pytest.approx(v[k], rel=1e-6)
