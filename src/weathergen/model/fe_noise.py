# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Opt-in latent noise / dropout at the forecast-engine input in inference (``forecast.fe_noise``).

In training, `ForecastingEngine.forward` perturbs its input latent with
``randn_like(tokens) * torch.norm(tokens) * fe_impute_latent_noise_std`` (``torch.norm`` is
the Frobenius norm of the whole latent) and runs its blocks with dropout ``fe_dropout_rate``;
inference is deterministic. This option reproduces (or varies) that regime in a validation /
inference rollout. The perturbation is applied to the tokens that enter the forecast engine,
i.e. after the forcing injection and its land-cell treatment, before the engine step::

    test_config:
      forecast:
        fe_noise:
          mode: train_like     # train_like | token_rms
          std: 1.0e-4          # train_like: x |z|_F; token_rms: x each token's own RMS
          dropout: true        # FE blocks in train mode (dropout active) for the step
          seed: 0
          steps: all           # all | first (only the rollout's first forecast step)

``train_like`` adds ``randn * torch.norm(tokens) * std`` (the training expression);
``token_rms`` adds ``randn * rms(token) * std`` per token, ``rms`` over the feature dim. All
tokens (auxiliary ones included, as in training) are perturbed. The noise of rollout step n
(n = 0 for the first forecast step) is drawn from a ``torch.Generator`` seeded from
``(seed, n)``, and the dropout masks of step n from the global RNG reseeded from
``(seed, n)`` inside ``torch.random.fork_rng`` (the caller's RNG state is restored), so a run
is reproducible and a chunked or resumed run equals a single chunk. ``dropout`` switches only
the engine's blocks to train mode; the engine's own training-mode noise stays off.

Unset or null disables it: the model call, the rollout and the saved rollout state are then
exactly the default ones. Read in validation and inference only.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import numbers

import numpy as np
import torch

FE_NOISE_MODES = ("train_like", "token_rms")
FE_NOISE_STEPS = ("all", "first")
_KEYS = ("mode", "std", "dropout", "seed", "steps")


@dataclasses.dataclass(frozen=True)
class FeNoise:
    """Latent noise / dropout at the forecast-engine input; see the module docstring."""

    mode: str
    std: float
    dropout: bool = False
    seed: int = 0
    steps: str = "all"

    def __post_init__(self):
        if self.mode not in FE_NOISE_MODES:
            raise ValueError(
                f"forecast.fe_noise.mode must be one of {FE_NOISE_MODES}, got {self.mode!r}"
            )
        s = self.std
        if isinstance(s, bool) or not isinstance(s, numbers.Real):
            raise ValueError(f"forecast.fe_noise.std must be a number, got {s!r}")
        s = float(s)
        if not math.isfinite(s) or s < 0.0:
            raise ValueError(f"forecast.fe_noise.std must be finite and >= 0, got {s!r}")
        object.__setattr__(self, "std", s)
        if not isinstance(self.dropout, bool):
            raise ValueError(f"forecast.fe_noise.dropout must be a bool, got {self.dropout!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral):
            raise ValueError(f"forecast.fe_noise.seed must be an integer, got {self.seed!r}")
        if self.seed < 0:
            raise ValueError(f"forecast.fe_noise.seed must be >= 0, got {self.seed!r}")
        object.__setattr__(self, "seed", int(self.seed))
        if self.steps not in FE_NOISE_STEPS:
            raise ValueError(
                f"forecast.fe_noise.steps must be one of {FE_NOISE_STEPS}, got {self.steps!r}"
            )
        if s == 0.0 and not self.dropout:
            raise ValueError("forecast.fe_noise with std 0 and no dropout does nothing; unset it")

    @classmethod
    def from_config(cls, cfg) -> FeNoise | None:
        """The setting of a `forecast.fe_noise` entry; None when unset or null."""
        if cfg is None:
            return None
        if not hasattr(cfg, "get") or not hasattr(cfg, "keys"):
            raise ValueError(f"forecast.fe_noise must be a mapping, got {cfg!r}")
        unknown = sorted(set(cfg.keys()) - set(_KEYS))
        if unknown:
            raise ValueError(f"forecast.fe_noise: unknown keys {unknown}, allowed {_KEYS}")
        if cfg.get("mode") is None or cfg.get("std") is None:
            raise ValueError("forecast.fe_noise needs mode and std")
        return cls(
            mode=cfg.get("mode"),
            std=cfg.get("std"),
            dropout=cfg.get("dropout", False),
            seed=cfg.get("seed", 0),
            steps=cfg.get("steps", "all"),
        )

    def as_dict(self) -> dict:
        """The setting saved with the rollout state."""
        return dataclasses.asdict(self)

    def active(self, n: int) -> bool:
        """Whether rollout step n (0 = the first forecast step) is perturbed."""
        return self.steps == "all" or n == 0

    def step_seed(self, n: int, stream: int) -> int:
        """Seed of rollout step n for stream 0 (noise) or 1 (dropout)."""
        return int(np.random.SeedSequence([self.seed, int(n), stream]).generate_state(1)[0])

    def perturb(self, tokens: torch.Tensor, n: int) -> torch.Tensor:
        """The engine input of rollout step n with the noise added (unchanged if std is 0)."""
        if self.std == 0.0 or not self.active(n):
            return tokens
        gen = torch.Generator(device=tokens.device)
        gen.manual_seed(self.step_seed(n, 0))
        noise = torch.randn(tokens.shape, generator=gen, dtype=tokens.dtype, device=tokens.device)
        if self.mode == "train_like":
            return tokens + noise * torch.norm(tokens) * self.std
        rms = tokens.float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(tokens.dtype)
        return tokens + noise * rms * self.std

    @contextlib.contextmanager
    def dropout_context(self, forecast_engine, n: int, device: torch.device):
        """Run the engine's blocks in train mode (dropout) with a step-seeded global RNG."""
        if not self.dropout or not self.active(n):
            yield
            return
        blocks = getattr(forecast_engine, "fe_blocks", None)
        if blocks is None:
            raise ValueError("forecast.fe_noise dropout needs a forecast engine with fe_blocks")
        devices = [device] if device.type == "cuda" else []
        prev = [m.training for m in blocks.modules()]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.step_seed(n, 1))
            blocks.train(True)
            try:
                yield
            finally:
                for m, t in zip(blocks.modules(), prev, strict=True):
                    m.training = t
