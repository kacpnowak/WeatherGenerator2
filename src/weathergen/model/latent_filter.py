# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Opt-in latent filter for validation / inference rollouts (``forecast.latent_filter``).

The forecast engine advances the carried latent one step at a time,
``z*_{n+1} = F(z_n)``, where ``F`` is the forcing injection followed by the forecast engine
and ``z_n`` is the latent carried from the previous step (the one that was decoded as step
``n``). A filter replaces ``z*_{n+1}`` by a combination of it and the previous carried
latents before it is decoded and carried on. Configured under the mode config's
``forecast`` block, e.g.::

    test_config:
      forecast:
        latent_filter: {type: second_difference, nu: 0.1}

Unset, null or ``type: none`` disables it, and the rollout is exactly the unfiltered one.
It is read in validation and inference only; training never applies it.

Types (``nu`` in ``[0, 1)``):

``relax``
    ``z_{n+1} = (1 - nu) F(z_n) + nu z_n``: relaxation of the new latent toward the
    current one. For a mode of the step's Jacobian with eigenvalue ``lam`` the filtered
    step has eigenvalue ``(1 - nu) lam + nu``: ``lam = -1`` becomes ``-(1 - 2 nu)``,
    ``lam = 1`` is unchanged, and ``lam = 1 + d`` becomes ``1 + (1 - nu) d``. The filter
    therefore also slows every non-stationary component, like a time step of ``1 - nu``.

``second_difference``
    ``z_{n+1} = F(z_n) - nu (F(z_n) - 2 z_n + z_{n-1})
             = (1 - nu) F(z_n) + nu (2 z_n - z_{n-1})``:
    the new latent is relaxed toward the linear extrapolation of the two previous carried
    latents instead of toward ``z_n``; the correction is ``nu`` times the second
    difference of the three consecutive latents (a causal Robert-Asselin-type filter that
    acts on the newest time level). For ``z_n ~ mu^n`` along a Jacobian eigenvector with
    eigenvalue ``lam``,
    ``mu^2 - ((1 - nu) lam + 2 nu) mu + nu = 0``.
    At ``lam = 1`` the roots are ``mu = 1`` (unchanged) and ``mu = nu`` (a damped
    computational mode), and ``d mu / d lam = 1`` there, so slowly varying components
    keep their amplitude and growth rate to first order in ``lam - 1``. Alternating
    components are damped: ``lam = -1`` gives ``mu = -0.2, -0.5`` for ``nu = 0.1`` and
    ``|mu| = 0.5`` for ``nu = 0.25``; real ``lam`` is stable for
    ``-(1 + 3 nu) / (1 - nu) < lam < 1``. The filter needs two previous latents, so the
    first forecast step of a rollout (no ``z_{n-1}``) is not filtered. ``z_{n-1}`` is
    carried across rollout chunks (``ModelOutput.latent_filter_prev``) and saved with the
    resumable rollout state, so a chunked or resumed rollout equals the one-chunk rollout.

``z_n`` is the carried latent before the step's forcing is injected.
"""

from __future__ import annotations

import dataclasses

import torch

FILTER_TYPES = ("relax", "second_difference")


@dataclasses.dataclass(frozen=True)
class LatentFilter:
    """One of the `FILTER_TYPES` with its coefficient `nu` (see the module docstring)."""

    type: str
    nu: float

    def __post_init__(self):
        if self.type not in FILTER_TYPES:
            raise ValueError(
                f"forecast.latent_filter.type must be one of {FILTER_TYPES} or none, "
                f"got {self.type!r}"
            )
        if not 0.0 <= float(self.nu) < 1.0:
            raise ValueError(f"forecast.latent_filter.nu must be in [0, 1), got {self.nu!r}")
        object.__setattr__(self, "nu", float(self.nu))

    @classmethod
    def from_config(cls, cfg) -> LatentFilter | None:
        """The filter of a `forecast.latent_filter` config entry; None when it is disabled."""
        if cfg is None:
            return None
        filter_type = cfg.get("type")
        if filter_type is None or str(filter_type).lower() == "none":
            return None
        if cfg.get("nu") is None:
            raise ValueError("forecast.latent_filter needs a coefficient `nu`")
        unknown = set(cfg.keys()) - {"type", "nu"}
        if unknown:
            raise ValueError(f"unknown forecast.latent_filter keys: {sorted(unknown)}")
        return cls(type=str(filter_type), nu=cfg.get("nu"))

    @property
    def needs_prev(self) -> bool:
        """Whether the filter uses the latent one step before the current one."""
        return self.type == "second_difference"

    def as_dict(self) -> dict:
        return {"type": self.type, "nu": self.nu}

    def __call__(
        self,
        z_new: torch.Tensor,
        z_cur: torch.Tensor,
        z_prev: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Filtered latent ``z_{n+1}`` from ``z_new = F(z_n)``, ``z_cur = z_n`` and
        ``z_prev = z_{n-1}`` (None before the second step of a rollout).
        """
        nu = self.nu
        if self.type == "relax":
            return (1.0 - nu) * z_new + nu * z_cur
        # second_difference
        if z_prev is None:
            return z_new
        return z_new - nu * (z_new - 2.0 * z_cur + z_prev)
