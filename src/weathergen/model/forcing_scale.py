# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Opt-in inference scaling of the per-step forcing injection (``forecast.forcing_scale``).

Every forecast step adds the forcing stream's contribution to the forecast-engine latent (or
its conditioning) through the gated residual of `ForcingInjection`: for the additive mode the
cell tokens become ``z + gate * emb(forcing)``. This option multiplies that residual by a
scalar ``alpha`` in a validation / inference rollout, i.e. the gate becomes ``alpha * gate``
(additive: ``z + alpha * gate * emb``; cross_attn: ``alpha`` times the attention delta;
global / adaln_local: ``alpha`` times the conditioning term). Configured under the mode
config's ``forecast`` block::

    test_config:
      forecast:
        forcing_scale: 0.5

Unset, null or ``1.0`` disables it: the model call, the rollout and the saved rollout state
are exactly the default ones. It is read in validation and inference only; training never
applies it. It composes with ``forecast.land_cells``: ``no_forcing`` still leaves the land
tokens unforced (the scaled injection is applied, then the land tokens are restored), and
``freeze`` still resets them after every step.
"""

from __future__ import annotations

import dataclasses
import math
import numbers


@dataclasses.dataclass(frozen=True)
class ForcingScale:
    """Scale ``alpha`` (finite, >= 0) of the forcing injection's gated residual."""

    alpha: float

    def __post_init__(self):
        a = self.alpha
        if isinstance(a, bool) or not isinstance(a, numbers.Real):
            raise ValueError(f"forecast.forcing_scale must be a number, got {a!r}")
        a = float(a)
        if not math.isfinite(a) or a < 0.0:
            raise ValueError(f"forecast.forcing_scale must be finite and >= 0, got {a!r}")
        object.__setattr__(self, "alpha", a)

    @classmethod
    def from_config(cls, value) -> ForcingScale | None:
        """The setting of a `forecast.forcing_scale` entry; None when unset or 1.0 (default)."""
        if value is None:
            return None
        fs = cls(value)
        return None if fs.alpha == 1.0 else fs

    def as_dict(self) -> dict:
        """The setting saved with the rollout state."""
        return {"alpha": self.alpha}
