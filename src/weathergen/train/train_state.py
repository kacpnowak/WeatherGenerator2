# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Training state saved next to each model checkpoint (a sidecar `.pt` file): the optimizer state,
the dynamic-loss channel-weight EMAs and the step counter it was saved at. Restored only when a
run is resumed (general.istep > 0), never when a new run starts from another run's weights.
"""

import logging
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from weathergen.utils.distributed import is_root

logger = logging.getLogger(__name__)


def train_state_path(model_path: Path, run_id: str, mini_epoch: int) -> Path:
    """Sidecar of the model checkpoint `<run_id>_latest` (mini_epoch -1) or `_chkptNNNNN`."""
    chkpt = "latest" if mini_epoch == -1 else f"chkpt{mini_epoch:05d}"
    return model_path / f"{run_id}_{chkpt}_train_state.pt"


def should_restore_train_state(istep: int) -> bool:
    """A resumed run continues its training state; a new run (istep 0) starts it afresh."""
    return istep > 0


def gather_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """
    Full (unsharded) optimizer state, keyed by parameter name, on rank 0's CPU; {} on the other
    ranks. Collective: every rank has to call it.
    """
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return get_optimizer_state_dict(model, optimizer, options=options)


def _with_all_parameters(model: torch.nn.Module, optimizer_state: dict) -> dict:
    """
    AdamW has state only for parameters that have had a gradient (with a spin-up the encoder gets
    one only in batches without spin-up), but set_optimizer_state_dict asks for every trainable
    parameter. The others get the state AdamW would start them with.
    """
    state = dict(optimizer_state["state"])
    for name, param in model.named_parameters():
        if param.requires_grad and name not in state:
            state[name] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros(param.shape, dtype=param.dtype),
                "exp_avg_sq": torch.zeros(param.shape, dtype=param.dtype),
            }
    return {**optimizer_state, "state": state}


def load_optimizer_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, optimizer_state: dict
) -> None:
    """
    Load the per-parameter optimizer state (AdamW's moments and step count) gathered by
    gather_optimizer_state. With a process group rank 0 passes the full state and the other ranks
    {}; rank 0 sends each rank its shards. Collective: every rank has to call it.

    The param group hyperparameters (lr, betas, eps, ...) stay as this run set them: the lr
    scheduler has already advanced them to the current step, and betas and eps scale with the
    world size.
    """
    hyperparameters = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    if optimizer_state:
        optimizer_state = _with_all_parameters(model, optimizer_state)
    options = StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=torch.distributed.is_initialized()
    )
    set_optimizer_state_dict(model, optimizer, optim_state_dict=optimizer_state, options=options)
    for group, kept in zip(optimizer.param_groups, hyperparameters, strict=True):
        group.update(kept)


def save_train_state(path: Path, optimizer_state: dict, dynamic_loss: dict, istep: int) -> None:
    """Write the training state on rank 0, via a temporary file so that `path` is never partial."""
    if not is_root():
        return
    state = {"optimizer": optimizer_state, "dynamic_loss": dynamic_loss, "istep": istep}
    path_tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    torch.save(state, path_tmp)
    path_tmp.replace(path)


def load_train_state(path: Path) -> dict | None:
    """The training state saved at `path` (on the CPU), or None if there is none."""
    if not path.exists():
        logger.warning(
            f"No training state at {path}: the optimizer and dynamic-loss state start afresh."
        )
        return None
    # not weights_only: the param groups hold the numpy scalars the lr scheduler left there
    return torch.load(path, map_location=torch.device("cpu"), mmap=True, weights_only=False)
