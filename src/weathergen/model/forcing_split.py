# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Opt-in inference split of the additive forcing injection (``forecast.forcing_split``).

The additive forcing injection of a forecast step is the per-cell field
``f_n = gate * emb(forcing_n)`` added to the cell tokens (see `ForcingInjection`). Given a
constant per-cell field ``C`` (typically the time mean of ``f_n`` over a year, computed
offline by scripts/instab/forcing_decomp.py), this option replaces the injected field by

    alpha_const * C + alpha_anom * (f_n - C)

so the constant and the anomaly part can be scaled separately. Configured under the mode
config's ``forecast`` block::

    test_config:
      forecast:
        forcing_split:
          const_path: /path/to/C.pt      # tensor (num_cells, dim) or {"const": tensor, ...}
          alpha_const: 1.0
          alpha_anom: 0.5

Unset or null disables it: the model call, the rollout and the saved rollout state are then
exactly the default ones. A set option is always active, also for ``alpha_const = alpha_anom
= 1`` (then the injection equals the default up to rounding, not bitwise). It acts only on the
``additive`` injection mode, is read in validation and inference only, and cannot be combined
with ``forecast.forcing_scale``. It composes with ``forecast.land_cells`` like the default
injection (``no_forcing`` restores the land tokens after it, ``freeze`` resets them after
every step). The saved setting includes a CRC32 of ``C``, so a resume with a different
constant file is refused.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import zlib
from pathlib import Path

import torch

_KEYS = ("const_path", "alpha_const", "alpha_anom")


def _alpha(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"forecast.forcing_split.{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"forecast.forcing_split.{name} must be finite and >= 0, got {value!r}")
    return value


def load_const(path) -> torch.Tensor:
    """The constant field (num_cells, dim), float32 on CPU, from a tensor or {"const": tensor}."""
    obj = torch.load(Path(path), map_location="cpu", weights_only=False)
    const = obj.get("const") if isinstance(obj, dict) else obj
    if not torch.is_tensor(const) or const.dim() != 2 or not const.is_floating_point():
        raise ValueError(
            f"forecast.forcing_split.const_path {path}: expected a 2-d float tensor "
            "(num_cells, dim) or a dict with one under 'const'"
        )
    const = const.detach().to(torch.float32).contiguous()
    if not torch.isfinite(const).all():
        raise ValueError(f"forecast.forcing_split.const_path {path}: non-finite values")
    return const


@dataclasses.dataclass(frozen=True, eq=False)
class ForcingSplit:
    """Constant field ``const`` (num_cells, dim) and the scales of its two parts."""

    const_path: str
    const: torch.Tensor
    alpha_const: float
    alpha_anom: float

    def __post_init__(self):
        object.__setattr__(self, "const_path", str(self.const_path))
        object.__setattr__(self, "alpha_const", _alpha("alpha_const", self.alpha_const))
        object.__setattr__(self, "alpha_anom", _alpha("alpha_anom", self.alpha_anom))
        c = self.const
        if not torch.is_tensor(c) or c.dim() != 2:
            raise ValueError("forecast.forcing_split: const must be a (num_cells, dim) tensor")
        crc = zlib.crc32(c.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
        object.__setattr__(self, "_crc32", crc)
        # per-device / dtype copies of the constant, built on first use
        object.__setattr__(self, "_cache", {})

    @classmethod
    def from_config(cls, cfg) -> ForcingSplit | None:
        """The setting of a `forecast.forcing_split` entry; None when unset or null."""
        if cfg is None:
            return None
        if not hasattr(cfg, "get") or not hasattr(cfg, "keys"):
            raise ValueError(
                "forecast.forcing_split must be a mapping with const_path, alpha_const, "
                f"alpha_anom, got {cfg!r}"
            )
        unknown = sorted(set(cfg.keys()) - set(_KEYS))
        if unknown:
            raise ValueError(f"forecast.forcing_split: unknown keys {unknown}, allowed {_KEYS}")
        path = cfg.get("const_path")
        if not path:
            raise ValueError("forecast.forcing_split needs const_path")
        return cls(
            const_path=str(path),
            const=load_const(path),
            alpha_const=cfg.get("alpha_const", 1.0),
            alpha_anom=cfg.get("alpha_anom", 1.0),
        )

    @property
    def const_crc32(self) -> int:
        return self._crc32

    def as_dict(self) -> dict:
        """The setting saved with the rollout state."""
        return {
            "const_path": self.const_path,
            "const_crc32": int(self._crc32),
            "const_shape": list(self.const.shape),
            "alpha_const": self.alpha_const,
            "alpha_anom": self.alpha_anom,
        }

    def _const_on(self, like: torch.Tensor) -> torch.Tensor:
        key = (like.device, like.dtype)
        c = self._cache.get(key)
        if c is None:
            c = self.const.to(device=like.device, dtype=like.dtype)
            self._cache[key] = c
        return c

    def apply(self, injection: torch.Tensor) -> torch.Tensor:
        """``alpha_const * C + alpha_anom * (f - C)`` for the injected field f (num_cells, dim)."""
        if tuple(injection.shape[-2:]) != tuple(self.const.shape):
            raise ValueError(
                f"forecast.forcing_split: constant of shape {tuple(self.const.shape)} does not "
                f"match the injection of shape {tuple(injection.shape)}"
            )
        c = self._const_on(injection)
        return self.alpha_const * c + self.alpha_anom * (injection - c)
