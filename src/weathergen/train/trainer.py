# ruff: noqa: T201

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import contextlib
import copy
import gc
import logging
import time
from math import sqrt

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

# FSDP2
from torch.distributed.tensor import DTensor

import weathergen.common.config as config
from weathergen.common.config import Config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.model.ema import EMAModel
from weathergen.model.engines import LatentState
from weathergen.model.fe_noise import FeNoise
from weathergen.model.forcing_scale import ForcingScale
from weathergen.model.forcing_split import ForcingSplit
from weathergen.model.land_cells import LandCells
from weathergen.model.latent_filter import LatentFilter
from weathergen.model.model import ModelOutput
from weathergen.model.model_interface import (
    init_model_and_shard,
)
from weathergen.model.utils import apply_fct_to_blocks, set_to_eval
from weathergen.train.collapse_monitor import CollapseMonitor
from weathergen.train.loss_calculator import LossCalculator
from weathergen.train.loss_modules.loss_module_physical import DynamicLossEMA
from weathergen.train.lr_scheduler import LearningRateScheduler
from weathergen.train.target_and_aux_ssl_teacher import EMATeacher
from weathergen.train.target_and_aux_utils import get_target_aux_calculator
from weathergen.train.train_state import (
    gather_optimizer_state,
    load_optimizer_state,
    load_train_state,
    save_train_state,
    should_restore_train_state,
    train_state_path,
)
from weathergen.train.trainer_base import TrainerBase
from weathergen.train.utils import (
    TRAIN,
    VAL,
    NoOpGradScaler,
    Stage,
    cfg_keys_to_filter,
    extract_batch_metadata,
    filter_config_by_enabled,
    get_active_stage_config,
    get_batch_size_from_config,
    get_target_idxs_from_cfg,
)
from weathergen.utils.distributed import is_root
from weathergen.utils.performance import NullThroughputTracker, ThroughputTracker, nvtx_range
from weathergen.utils.train_logger import TrainLogger, prepare_losses_for_logging
from weathergen.utils.utils import get_dtype
from weathergen.utils.validation_io import write_output

logger = logging.getLogger(__name__)

# cfg_keys_to_filter = ["losses", "model_input", "target_input"]


def prefix_length(batch) -> int:
    """Unsupervised spin-up steps at the start of a training batch's rollout (0 without)."""
    first = getattr(batch, "first_supervised_step", None)
    return 0 if first is None else first - batch.get_output_idxs()[0]


def resume_position(
    istep: int, steps_per_mini_epoch: int, num_mini_epochs: int, world_size_changed: bool = False
) -> tuple[int, int]:
    """
    Where a run continued after `istep` optimizer steps resumes: (mini_epoch_base, steps_done),
    with steps_done the optimizer steps already done in mini_epoch_base.

    A run stopped at the end of a mini-epoch resumes at the start of the next one, a complete run
    at (num_mini_epochs, 0). After a world-size change, the steps done no longer map to the same
    samples, so the interrupted mini-epoch restarts from its first step.
    """
    mini_epoch_base, steps_done = divmod(istep, steps_per_mini_epoch)
    if mini_epoch_base >= num_mini_epochs:
        return num_mini_epochs, 0
    return mini_epoch_base, 0 if world_size_changed else steps_done


def resume_istep(
    istep: int, steps_per_mini_epoch: int, num_mini_epochs: int, world_size_changed: bool = False
) -> int:
    """
    The optimizer step counter a continued run carries on from: `istep`, unless the world size
    changed. Then the interrupted mini-epoch restarts from its first step (resume_position) and
    the counter restarts with it, so the run still ends at num_mini_epochs * steps_per_mini_epoch.
    """
    if not world_size_changed:
        return istep
    mini_epoch_base, _ = resume_position(istep, steps_per_mini_epoch, num_mini_epochs, True)
    return mini_epoch_base * steps_per_mini_epoch


class Trainer(TrainerBase):
    def __init__(self, train_logging: Config):
        TrainerBase.__init__(self)

        self.train_logging = train_logging

        self.data_loader: torch.utils.data.DataLoader | None = None
        self.data_loader_validation: torch.utils.data.DataLoader | None = None
        self.dataset: MultiStreamDataSampler | None = None
        self.dataset_val: MultiStreamDataSampler | None = None
        self.device: torch.device = None
        self.ema_model = None
        self.grad_scaler: torch.amp.GradScaler | NoOpGradScaler = NoOpGradScaler()
        self.last_grad_norm = None
        self.loss_calculator: LossCalculator | None = None
        self.loss_calculator_val: LossCalculator | None = None
        self.lr_scheduler: LearningRateScheduler | None = None
        self.model = None
        self.model_params = None
        self.optimizer: torch.optim.Optimizer | None = None
        # spin-up length of each training step since the last metrics log
        self._prefix_hist: list[int] = []
        self.t_start: float = 0
        self.target_and_aux_calculators = None
        self.target_and_aux_calculators_val = None
        self.validate_with_ema_cfg = None
        self.validate_with_ema: bool = False
        self.batch_size_per_gpu = -1
        self.batch_size_validation_per_gpu = -1
        self.batch_size_test_per_gpu = -1
        self.collapse_monitor: CollapseMonitor | None = None
        self.perf_tracker: ThroughputTracker | NullThroughputTracker = NullThroughputTracker()
        self.t_training_start: float = 0
        self.training_loop_annotation_context = contextlib.nullcontext

    def get_batch_size_total(self, batch_size_per_gpu) -> int:
        """
        Get total, effective batch size across all DDP ranks
        """
        return self.world_size_original * batch_size_per_gpu

    def init(self, cf: Config, devices):
        # pylint: disable=attribute-defined-outside-init
        self.cf = OmegaConf.merge(
            OmegaConf.create(
                {
                    "latent_noise_kl_weight": 0.0,
                    "latent_noise_gamma": 2.0,
                    "latent_noise_use_additive_noise": False,
                    "latent_noise_deterministic_latents": True,
                    "latent_noise_saturate_encodings": 5,
                }
            ),
            cf,
        )
        cf = self.cf

        self.freeze_modules = cf.get("freeze_modules", "")

        # get training config and remove disabled options (e.g. because of overrides)
        self.training_cfg = cf.get("training_config")
        self.training_cfg = filter_config_by_enabled(self.training_cfg, cfg_keys_to_filter)
        assert len(self.training_cfg.model_input.keys()) != 0, (
            "You probably have no loss term enabled"
        )

        # validation and test configs are training configs, updated by specified keys
        self.validation_cfg = get_active_stage_config(
            self.training_cfg, cf.get("validation_config", {}), cfg_keys_to_filter
        )
        # test cfg is derived from validation cfg with specified keys overwritten
        self.test_cfg = get_active_stage_config(
            self.validation_cfg, cf.get("test_config", {}), cfg_keys_to_filter
        )

        # batch sizes
        self.batch_size_per_gpu = get_batch_size_from_config(self.training_cfg)
        self.batch_size_validation_per_gpu = get_batch_size_from_config(self.validation_cfg)
        self.batch_size_test_per_gpu = get_batch_size_from_config(self.test_cfg)

        for mode, mode_cfg in zip(
            ["training_config", "validation_config", "test_config"],
            [self.training_cfg, self.validation_cfg, self.test_cfg],
            strict=True,
        ):
            config.validate_forecast_policy_and_steps(mode_cfg.get("forecast", {}), mode)

        self.mixed_precision_dtype = get_dtype(cf.mixed_precision_dtype)

        self.devices = devices

        # Get world_size of previous, to be continued run before
        # world_size gets overwritten by current setting during init_ddp()
        self.world_size_original = cf.get("world_size_original", cf.get("world_size", None))
        cf.world_size_original = self.world_size_original

        self.log_grad_norms = cf.train_logging.get("log_grad_norms", False)

        # create output directory
        if is_root():
            config.get_path_results(cf).mkdir(exist_ok=True, parents=True)
            config.get_path_model(cf).mkdir(exist_ok=True, parents=True)

        self.train_logger = TrainLogger(cf, config.get_path_results(self.cf))

        # Initialize collapse monitor for SSL training
        collapse_config = cf.train_logging.get("collapse_monitoring", {})
        self.collapse_monitor = CollapseMonitor(collapse_config, None)  # device set later in run()
        # self.fcst_chunks_by_mode = {
        #    "training": self._init_fcst_chunks(self.training_cfg),
        #    "validation": self._init_fcst_chunks(self.validation_cfg),
        #    "test": self._init_fcst_chunks(self.test_cfg),
        # }

        if cf.train_logging.get("track_performance_metrics"):
            self.perf_tracker = ThroughputTracker(
                device=torch.device(self.devices[0]),
                warmup_steps=cf.train_logging.get("performance_tracking_warmup_steps", 2),
                batch_size_per_gpu=self.batch_size_per_gpu,
            )
        if cf.get("profiling", {}).get("nvtx_annotate", False):
            self.training_loop_annotation_context = nvtx_range

    def get_target_aux_calculators(self, mode_cfg):
        """
        Get target_aux_calculators for given mode_cfg
        """

        batch_size = get_batch_size_from_config(mode_cfg)

        # get target_aux calculators for different loss terms
        target_and_aux_calculators = {}
        for loss_name, loss_cfg in mode_cfg.losses.items():
            target_and_aux_calculators[loss_name] = get_target_aux_calculator(
                self.cf, loss_cfg, self.dataset, self.model, self.device, batch_size
            ).to_device(self.device)

        return target_and_aux_calculators

    def _get_output_target_and_auxs(self, mode_cfg, batch):
        output_target_and_auxs = {}
        for loss_name, loss_cfg in mode_cfg.losses.items():
            if loss_cfg.type != "LossPhysical":
                continue

            target_aux = self.target_and_aux_calculators_val[loss_name]
            target_idxs = get_target_idxs_from_cfg(mode_cfg, loss_name)
            output_target_and_auxs[loss_name] = target_aux.compute(
                self.cf.general.istep,
                batch.get_target_samples_view(target_idxs),
                self.model_params,
                self.model,
            )

        return output_target_and_auxs

    def _get_forecast_chunks(self, forecast_cfg, batch):
        """
        Split the batch's output steps into chunks of `forecast.chunk_size` model steps; an unset
        chunk_size runs them in one chunk. The steps are counted on the batch, as the unchunked
        model did: one step for masking-only (offset 0, num_steps 0), and the per-batch draw
        when num_steps is a list.
        """
        num_steps = len(batch.get_output_idxs())
        chunk_size = forecast_cfg.get("chunk_size") or num_steps
        n_full_chunks = num_steps // chunk_size
        remainder_fsteps = num_steps % chunk_size
        return [chunk_size] * n_full_chunks + ([remainder_fsteps] if remainder_fsteps else [])

    def _rollout_state_path(self):
        """File where the resumable rollout state (carried latent + progress) is stored."""
        return config.get_path_results(self.cf) / "rollout_state.pt"

    def _save_rollout_state(
        self,
        latent_state: LatentState,
        done_steps: int,
        idx: int,
        latent_filter: LatentFilter | None = None,
        latent_filter_prev: torch.Tensor | None = None,
        land_cells: LandCells | None = None,
        land_cells_init: torch.Tensor | None = None,
        forcing_scale: ForcingScale | None = None,
        forcing_split: ForcingSplit | None = None,
        fe_noise: FeNoise | None = None,
    ):
        """
        Persist the state needed to resume a long rollout after this chunk.

        The rollout is Markovian in the latent tokens, so the carried LatentState plus the
        number of completed forecast steps (and the initial-condition index) is sufficient to
        continue exactly. Written atomically (tmp + replace) so a crash mid-save cannot corrupt
        an existing checkpoint. Completed forecast steps are already persisted in the output
        store, so only the continuation state lives here. With a latent filter
        (forecast.latent_filter) its configuration and, for a two-level filter, the previous
        carried latent are saved too, so the resumed rollout continues the filtered one.
        Likewise with forecast.land_cells: its setting and, for freeze, the land tokens of the
        initial latent; and with forecast.forcing_scale, forecast.forcing_split and
        forecast.fe_noise: their settings.
        """

        def to_cpu(t):
            return t.detach().to("cpu") if isinstance(t, torch.Tensor) else t

        cpu_state = LatentState(
            class_token=to_cpu(latent_state.class_token),
            register_tokens=to_cpu(latent_state.register_tokens),
            patch_tokens=to_cpu(latent_state.patch_tokens),
            z_pre_norm=to_cpu(latent_state.z_pre_norm),
        )
        path = self._rollout_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        state = {
            "latent_state": cpu_state,
            "done_steps": int(done_steps),
            "idx": int(idx),
            "run_id": self.cf.general.run_id,
        }
        if latent_filter is not None:
            state["latent_filter"] = latent_filter.as_dict()
            state["latent_filter_prev"] = to_cpu(latent_filter_prev)
        if land_cells is not None:
            state["land_cells"] = land_cells.as_dict()
            state["land_cells_init"] = to_cpu(land_cells_init)
        if forcing_scale is not None:
            state["forcing_scale"] = forcing_scale.as_dict()
        if forcing_split is not None:
            state["forcing_split"] = forcing_split.as_dict()
        if fe_noise is not None:
            state["fe_noise"] = fe_noise.as_dict()
        torch.save(state, tmp)
        tmp.replace(path)

    @staticmethod
    def _check_resume_setting(resumed: dict, key: str, current) -> None:
        """Refuse to resume a rollout whose saved `key` setting (as_dict) differs from `current`."""
        saved = resumed.get(key)
        current = current.as_dict() if current is not None else None
        if saved != current:
            raise ValueError(
                f"resume {key} mismatch: saved {saved} vs current {current}; refusing to resume."
            )

    def _load_rollout_state(self):
        """Load a previously saved rollout state, or None if there is nothing to resume."""
        path = self._rollout_state_path()
        if not path.exists():
            return None
        state = torch.load(path, map_location="cpu", weights_only=False)
        ls = state["latent_state"]
        state["latent_state"] = LatentState(
            class_token=ls.class_token.to(self.device) if ls.class_token is not None else None,
            register_tokens=(
                ls.register_tokens.to(self.device) if ls.register_tokens is not None else None
            ),
            patch_tokens=ls.patch_tokens.to(self.device) if ls.patch_tokens is not None else None,
            z_pre_norm=ls.z_pre_norm.to(self.device),
        )
        if state.get("latent_filter_prev") is not None:
            state["latent_filter_prev"] = state["latent_filter_prev"].to(self.device)
        if state.get("land_cells_init") is not None:
            state["land_cells_init"] = state["land_cells_init"].to(self.device)
        return state

    def _process_validation_chunks(
        self,
        batch,
        mode_cfg,
        batch_size,
        mini_epoch,
        bidx,
        targets_and_auxs,
        preds_full,
    ):
        # chunks = self._get_fcst_chunks(mode_cfg)
        chunks = self._get_forecast_chunks(mode_cfg.get("forecast", {}), batch)
        source_samples = batch.get_source_samples()
        num_samples_write = mode_cfg.get("output", {}).get("num_samples", 0) * batch_size
        should_write_output = bidx < num_samples_write
        # lazy targets: materialize one chunk of targets at a time (long single-sample rollout)
        lazy = getattr(self.dataset_val, "_lazy_targets", False)
        if should_write_output:
            denormalize_data_fct = (
                (lambda x0, x1: x1)
                if mode_cfg.get("output", {}).get("normalized_samples", False)
                else self.dataset_val.denormalize_target_channels
            )
            # in lazy mode targets do not exist yet; they are built (and checked) per chunk below
            if not lazy and not targets_and_auxs:
                raise ValueError(
                    "Writing validation output requires targets. "
                    "Configure validation losses or set output.num_samples=0."
                )

        output_idxs = batch.get_output_idxs()
        forecast_step_offset = output_idxs[0] if len(output_idxs) > 0 else 0

        # resumable long rollout (lazy path only): the rollout is Markovian in the latent, so
        # we can checkpoint the carried latent per chunk and continue from it after a crash.
        forecast_cfg = mode_cfg.get("forecast", {})
        save_state = lazy and forecast_cfg.get("checkpoint_rollout", True)
        rollout_idx = self.dataset_val._materialize_ctx["idx"] if lazy else None
        # optional inference-time latent filter; unset leaves the model call unchanged
        latent_filter = LatentFilter.from_config(forecast_cfg.get("latent_filter"))
        rollout_kwargs = {} if latent_filter is None else {"latent_filter": latent_filter}
        if latent_filter is not None:
            logger.info(f"Latent filter for this rollout: {latent_filter.as_dict()}")
        # optional inference-time treatment of the latent land cells; unset leaves it unchanged
        land_cells = LandCells.from_config(
            forecast_cfg.get("land_cells"), self.cf.streams, self.cf.healpix_level
        )
        if land_cells is not None:
            rollout_kwargs["land_cells"] = land_cells
            logger.info(f"Land cells for this rollout: {land_cells.as_dict()}")
        # optional inference-time scale of the forcing injection; unset / 1.0 leaves it unchanged
        forcing_scale = ForcingScale.from_config(forecast_cfg.get("forcing_scale"))
        if forcing_scale is not None:
            rollout_kwargs["forcing_scale"] = forcing_scale
            logger.info(f"Forcing scale for this rollout: {forcing_scale.as_dict()}")
        # optional inference-time split of the forcing injection; unset leaves it unchanged
        forcing_split = ForcingSplit.from_config(forecast_cfg.get("forcing_split"))
        if forcing_split is not None:
            if forcing_scale is not None:
                raise ValueError(
                    "forecast.forcing_split and forecast.forcing_scale cannot be combined"
                )
            rollout_kwargs["forcing_split"] = forcing_split
            logger.info(f"Forcing split for this rollout: {forcing_split.as_dict()}")
        # optional inference-time noise / dropout at the FE input; unset leaves it unchanged
        fe_noise = FeNoise.from_config(forecast_cfg.get("fe_noise"))
        if fe_noise is not None:
            rollout_kwargs["fe_noise"] = fe_noise
            logger.info(f"FE noise for this rollout: {fe_noise.as_dict()}")

        cum_steps = 0
        chunk_start = 0
        if lazy and forecast_cfg.get("resume", False):
            resumed = self._load_rollout_state()
            if resumed is not None:
                done = resumed["done_steps"]
                assert resumed["idx"] == int(rollout_idx), (
                    f"resume idx mismatch: saved {resumed['idx']} vs current {rollout_idx}; "
                    "the initial condition differs, refusing to resume."
                )
                # state is saved on chunk boundaries, so skip whole completed chunks
                acc = 0
                while chunk_start < len(chunks) and acc + chunks[chunk_start] <= done:
                    acc += chunks[chunk_start]
                    chunk_start += 1
                assert acc == done, f"saved done_steps={done} not on a chunk boundary"
                self._check_resume_setting(resumed, "latent_filter", latent_filter)
                self._check_resume_setting(resumed, "land_cells", land_cells)
                self._check_resume_setting(resumed, "forcing_scale", forcing_scale)
                self._check_resume_setting(resumed, "forcing_split", forcing_split)
                self._check_resume_setting(resumed, "fe_noise", fe_noise)
                cum_steps = done
                # seed the rollout from the saved latent at the completed step offset; an empty
                # physical list makes the model resume at exactly `done` steps
                seed = ModelOutput(0, batch=batch.get_source_samples(), step_offset=done)
                seed.latent = [{"latent_state": resumed["latent_state"]}]
                seed.latent_filter_prev = resumed.get("latent_filter_prev")
                seed.land_cells_init = resumed.get("land_cells_init")
                source_samples = seed
                logger.info(
                    f"Resuming rollout from step {done} (skipping {chunk_start}/{len(chunks)} "
                    "completed chunks)."
                )

        with tqdm.tqdm(
            total=len(chunks),
            initial=chunk_start,
            disable=self.cf.with_ddp or len(chunks) <= 1,
            leave=False,
            desc=f"batch {bidx + 1} chunks",
        ) as chunk_pbar:
            for chunk_idx in range(chunk_start, len(chunks)):
                chunk_size = chunks[chunk_idx]
                chunk_pbar.set_postfix_str(
                    f"running {chunk_idx + 1}/{len(chunks)} ({chunk_size} steps)"
                )
                chunk_pbar.refresh()

                # absolute forecast-step range this chunk covers; matches the post-call
                # chunk_step_offset but is known before the model reads target_coords/conditions
                chunk_abs_range = (
                    forecast_step_offset + cum_steps,
                    forecast_step_offset + cum_steps + chunk_size,
                )
                if lazy:
                    # build this chunk's targets, move them to device (the coord-conditioned
                    # decoder and conditions read them during the forward), then compute the
                    # per-chunk target/aux used for output writing.
                    self.dataset_val._materialize_target_chunk(batch, chunk_abs_range)
                    batch.to_device(self.device)
                    if should_write_output:
                        targets_and_auxs = self._get_output_target_and_auxs(mode_cfg, batch)

                if self.ema_model is None:
                    source_samples = self.model(
                        self.model_params,
                        source_samples,
                        chunk_size,
                        **rollout_kwargs,
                    )
                else:
                    source_samples = self.ema_model.forward_eval(
                        self.model_params,
                        source_samples,
                        chunk_size,
                        **rollout_kwargs,
                    )

                chunk_step_offset = forecast_step_offset + source_samples.step_offset

                if should_write_output:
                    chunk_pbar.set_postfix_str(
                        f"writing {chunk_idx + 1}/{len(chunks)} "
                        f"(steps {chunk_step_offset}-{chunk_step_offset + chunk_size - 1})"
                    )
                    chunk_pbar.refresh()
                    timestep_idxs = list(range(chunk_step_offset, chunk_step_offset + chunk_size))
                    write_output(
                        self.cf,
                        mode_cfg,
                        batch_size,
                        mini_epoch,
                        bidx,
                        denormalize_data_fct,
                        batch,
                        source_samples,
                        targets_and_auxs,
                        timestep_idxs=timestep_idxs,
                        fstep_offset=chunk_step_offset,
                    )

                if preds_full is not None:
                    for step_idx in range(chunk_size):
                        preds_full.physical[chunk_step_offset + step_idx].update(
                            source_samples.physical[step_idx]
                        )
                        preds_full.latent[chunk_step_offset + step_idx].update(
                            source_samples.latent[step_idx]
                        )
                        # deep SSL predictions, or the SSL loss falls back to the unfused head
                        if source_samples.latent_deep is not None:
                            if preds_full.latent_deep is None:
                                preds_full.latent_deep = [{} for _ in preds_full.physical]
                            preds_full.latent_deep[chunk_step_offset + step_idx].update(
                                source_samples.latent_deep[step_idx]
                            )

                if chunk_size and chunk_idx < len(chunks) - 1:
                    if (
                        not source_samples.latent
                        or "latent_state" not in source_samples.latent[chunk_size - 1]
                    ):
                        raise ValueError("Missing latent_state for chunked forecast continuation.")

                if lazy:
                    # chunk consumed (forecast + written): drop its targets to bound memory
                    self.dataset_val._release_target_chunk(batch, chunk_abs_range)

                cum_steps += chunk_size

                # checkpoint the carried latent after every chunk (including the last) so the
                # rollout can be resumed after a crash or extended later to more forecast steps.
                # Kept on completion so a finished rollout stays continuable.
                if save_state:
                    self._save_rollout_state(
                        source_samples.get_last_latent_state(),
                        cum_steps,
                        rollout_idx,
                        latent_filter=latent_filter,
                        latent_filter_prev=source_samples.latent_filter_prev,
                        land_cells=land_cells,
                        land_cells_init=source_samples.land_cells_init,
                        forcing_scale=forcing_scale,
                        forcing_split=forcing_split,
                        fe_noise=fe_noise,
                    )

                chunk_pbar.update(1)

    def inference(self, cf, devices, run_id_contd, mini_epoch_contd):
        # general initalization
        self.init(cf, devices)

        cf = self.cf
        device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{device_type}:{cf.local_rank}")
        self.ema_model = None

        # create data loader
        # only one needed since we only run the validation code path
        self.dataset = MultiStreamDataSampler(
            cf,
            self.test_cfg,
            stage=VAL,
        )
        self.dataset_val = self.dataset

        # make sure number of loaders does not exceed requested samples
        loader_num_workers = min(self.test_cfg.samples_per_mini_epoch, cf.data_loading.num_workers)
        # lazy target materialization needs the data readers in the main process (the trainer
        # calls back into the dataset per chunk), so it runs without dataloader workers. This is
        # the single-sample long-rollout regime, where worker parallelism gains nothing anyway.
        if getattr(self.dataset, "_lazy_targets", False):
            loader_num_workers = 0
        loader_params = {
            "batch_size": None,
            "batch_sampler": None,
            "shuffle": False,
            "num_workers": loader_num_workers,
            "pin_memory": cf.data_loading.get("memory_pinning", False),
            # persistent_workers is only valid with at least one worker
            "persistent_workers": loader_num_workers > 0
            and cf.data_loading.get("persistent_workers", False),
        }
        self.data_loader_validation = torch.utils.data.DataLoader(
            self.dataset, **loader_params, sampler=None
        )

        self.model, self.model_params = init_model_and_shard(
            cf,
            self.dataset,
            run_id_contd,
            mini_epoch_contd,
            self.test_cfg.training_mode,
            devices[0],
            cf.with_ddp,
            cf.with_fsdp,
        )

        # get target_aux calculators for different loss terms
        self.target_and_aux_calculators_val = self.get_target_aux_calculators(self.test_cfg)

        self.loss_calculator_val = LossCalculator(cf, self.test_cfg, VAL, device=self.devices[0])

        if is_root():
            config.save(self.cf, mini_epoch=0)

        logger.info(f"Starting inference with id={self.cf.general.run_id}.")

        # inference validation set
        self.validate(0, self.test_cfg, self.batch_size_test_per_gpu)
        logger.info(f"Finished inference run with id: {cf.general.run_id}")

        # Without this, NCCL's heartbeat monitor keeps polling a TCPStore whose server has
        # already gone away, and the ranks never exit.
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def run(self, cf, devices, run_id_contd=None, mini_epoch_contd=None):
        # general initalization
        self.init(cf, devices)
        cf = self.cf

        device_type = torch.accelerator.current_accelerator()
        self.device = torch.device(f"{device_type}:{cf.local_rank}")

        # Update collapse monitor device
        self.collapse_monitor.device = self.device

        # create data loaders
        self.dataset = MultiStreamDataSampler(cf, self.training_cfg, stage=TRAIN)
        self.dataset_val = MultiStreamDataSampler(cf, self.validation_cfg, stage=VAL)

        loader_params = {
            "batch_size": None,
            "batch_sampler": None,
            "shuffle": False,
            "num_workers": cf.data_loading.num_workers,
        }
        self.data_loader = torch.utils.data.DataLoader(self.dataset, **loader_params, sampler=None)
        self.data_loader_validation = torch.utils.data.DataLoader(
            self.dataset_val, **loader_params, sampler=None
        )

        self.model, self.model_params = init_model_and_shard(
            cf,
            self.dataset,
            run_id_contd,
            mini_epoch_contd,
            self.training_cfg.training_mode,
            devices[0],
            cf.with_ddp,
            cf.with_fsdp,
        )

        validate_with_ema_cfg = self.validation_cfg.get("validate_with_ema")
        if validate_with_ema_cfg is not None:
            # if the config is specified and enabled not specified, then assume it is to be used
            self.validate_with_ema = validate_with_ema_cfg.get("enabled", True)
        else:
            self.validate_with_ema = False
        self.ema_model = None
        if self.validate_with_ema:
            meta_ema_model, _ = init_model_and_shard(
                cf,
                self.dataset,
                run_id_contd,
                mini_epoch_contd,
                cf.training_config.training_mode,
                devices[0],
                cf.with_ddp,
                cf.with_fsdp,
            )
            self.ema_model = EMAModel(
                self.model,
                meta_ema_model,
                halflife_steps=validate_with_ema_cfg.get("ema_halflife_in_thousands", 1e-3),
                rampup_ratio=validate_with_ema_cfg.get("ema_ramp_up_ratio", 0.09),
                is_model_sharded=(cf.with_ddp and cf.with_fsdp),
            )

        # get target_aux calculators for different loss terms
        self.target_and_aux_calculators = self.get_target_aux_calculators(self.training_cfg)
        self.target_and_aux_calculators_val = self.get_target_aux_calculators(self.validation_cfg)

        # if with_fsdp then parameter count is unreliable
        if is_root():
            # ddp-wrapped model does not expose this function
            if not cf.with_ddp:
                self.model.print_num_parameters()

        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        # aiming for beta1=0.9 and beta2=0.95 following the MAE paper
        # https://arxiv.org/pdf/2111.06377
        kappa = self.get_batch_size_total(self.batch_size_per_gpu)
        # aiming for beta1 = 0.9 at one node, ie kappa=B=4
        beta1 = max(0.5, 1.0 - kappa * (1.0 - self.training_cfg.optimizer.adamw.beta1))
        # aiming for beta2 = 0.95 at one node, ie B=4
        beta2 = max(0.9, 1.0 - kappa * (1.0 - self.training_cfg.optimizer.adamw.beta2))
        eps = self.training_cfg.optimizer.adamw.get("eps", 2e-08) / np.sqrt(kappa)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_cfg.learning_rate_scheduling.lr_start,
            weight_decay=self.training_cfg.optimizer.weight_decay,
            betas=(beta1, beta2),
            eps=eps,
            fused=True,
        )
        if cf.get("training_config").get("optimizer").get("grad_scaling", True):
            self.grad_scaler = torch.amp.GradScaler("cuda")
        assert len(self.dataset) > 0, f"No data found in {self.dataset}"

        # recover mini_epoch and the steps already done in it when continuing a run; the sampler
        # starts at mini_epoch 0, step 0 in a new process (ecmwf/WeatherGenerator#2953). Done
        # before the lr scheduler is built, which continues from the (realigned) step counter.
        steps_per_mini_epoch = len(self.dataset) // self.batch_size_per_gpu
        world_size_changed = cf.world_size != self.world_size_original
        istep_saved = self.cf.general.istep
        mini_epoch_base, steps_done = resume_position(
            istep_saved, steps_per_mini_epoch, self.training_cfg.num_mini_epochs, world_size_changed
        )
        self.cf.general.istep = resume_istep(
            istep_saved, steps_per_mini_epoch, self.training_cfg.num_mini_epochs, world_size_changed
        )
        self.dataset.set_resume_position(mini_epoch_base, steps_done)
        if is_root():
            if world_size_changed and istep_saved > 0:
                logger.warning(
                    f"World size changed from {self.world_size_original} to {cf.world_size}: "
                    f"the data order cannot be continued, mini_epoch {mini_epoch_base} restarts "
                    f"from its first step and istep from {istep_saved} to "
                    f"{self.cf.general.istep}."
                )
            logger.info(
                f"Training starts at mini_epoch {mini_epoch_base}, step {steps_done} of "
                f"{steps_per_mini_epoch} (istep={self.cf.general.istep})."
            )

        # lr is updated after each batch so account for this
        # TODO: conf should be read-only, do not modify the conf in flight
        len_ds = len(self.dataset)
        lr_steps = int((len_ds * self.training_cfg.num_mini_epochs) / self.batch_size_per_gpu)
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            self.batch_size_per_gpu,
            cf.world_size,
            cf.general.istep,
            lr_steps,
            self.training_cfg.learning_rate_scheduling,
        )

        if self.cf.general.istep > 0 and is_root():
            logger.info(f"Continuing run with learning rate: {self.lr_scheduler.get_lr()}")

        # Instantiate loss calculator modules to compute losses
        self.loss_calculator = LossCalculator(cf, self.training_cfg, TRAIN, device=self.device)
        val_cfg = self.validation_cfg
        self.loss_calculator_val = LossCalculator(cf, val_cfg, VAL, device=self.device)

        # continue the optimizer and dynamic-loss state when this run is resumed, but not when a
        # new run starts from another run's weights (general.istep=0); decided on the saved
        # istep, since a world-size change can realign the counter of a resumed run to 0
        if run_id_contd is not None:
            if should_restore_train_state(istep_saved):
                self._restore_train_state(run_id_contd, mini_epoch_contd, istep_saved)
            elif is_root():
                logger.info(
                    f"Starting from the weights of {run_id_contd} with a new optimizer and "
                    "dynamic-loss state (istep=0)."
                )

        if is_root():
            config.save(self.cf, None)
            logger.info(config.format_cf(self.cf))

        # run validation before training if requested
        self.validate_before_training()

        # training loop
        self.t_training_start = time.time()

        for mini_epoch in range(mini_epoch_base, self.training_cfg.num_mini_epochs):
            if is_root():
                logger.info(
                    f"Mini_epoch {mini_epoch} of {self.training_cfg.num_mini_epochs}: train."
                )
            self.train(mini_epoch)

            if is_root():
                logger.info(
                    f"Mini_epoch {mini_epoch} of {self.training_cfg.num_mini_epochs}: validate."
                )
            self.validate(mini_epoch, self.validation_cfg, self.batch_size_validation_per_gpu)

            if is_root():
                logger.info(
                    f"Mini_epoch {mini_epoch} of {self.training_cfg.num_mini_epochs}: save_model."
                )
            self.save_model(mini_epoch)
            # keep _latest no older than the last completed mini_epoch
            self.save_model(-1)

        # log final model
        self.save_model(self.training_cfg.num_mini_epochs)

        # Without this, NCCL's heartbeat monitor keeps polling a TCPStore whose server has
        # already gone away, and the ranks never exit.
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def validate_before_training(self):
        """
        Perform validation before training (eg. to check validation pipeline or data normalization)
        if config parameters are set accordingly
        """

        # validate once at the beginning as reference
        if self.validation_cfg.get("validate_before_training", None) is not None:
            validate_before_training = self.validation_cfg.get("validate_before_training")
            batch_size = self.batch_size_validation_per_gpu
            if type(validate_before_training) is bool:
                if validate_before_training:
                    self.validate(-1, self.validation_cfg, batch_size)
            elif type(validate_before_training) is int:
                if validate_before_training > 0:
                    cfg = copy.deepcopy(self.validation_cfg)
                    cfg.samples_per_mini_epoch = validate_before_training
                    self.validate(-1, cfg, batch_size)
            else:
                assert False, "validate_before_training must be integer or boolean."

    def train(self, mini_epoch):
        """
        Perform training for one epoch
        """

        cf = self.cf
        self.model.train()

        apply_fct_to_blocks(self.model, cf.freeze_modules, set_to_eval)

        dataset_iter = iter(self.data_loader)

        self.optimizer.zero_grad()

        # training loop
        self.t_start = time.time()
        for bidx, batch in enumerate(dataset_iter):
            with self.training_loop_annotation_context(f"batch_{bidx}"):
                if cf.data_loading.get("memory_pinning", False):
                    # pin memory for faster CPU-GPU transfer
                    batch = batch.pin_memory()

                batch.to_device(self.device)

            with torch.autocast(
                device_type=f"cuda:{cf.local_rank}",
                dtype=self.mixed_precision_dtype,
                enabled=cf.with_mixed_precision,
            ):
                x = batch.get_source_samples()
                preds = self.model(
                    self.model_params,
                    x,
                    x.get_output_len(),
                )

                self._prefix_hist.append(prefix_length(x))

                targets_and_auxs = {}
                for loss_name, target_aux in self.target_and_aux_calculators.items():
                    # find targets for this target-aux calculator
                    target_idxs = get_target_idxs_from_cfg(self.training_cfg, loss_name)
                    # apply target-aux calculator
                    targets_and_auxs[loss_name] = target_aux.compute(
                        self.cf.general.istep,
                        batch.get_target_samples(target_idxs),
                        self.model_params,
                        self.model,
                    )

                loss = self.loss_calculator.compute_loss(
                    preds=preds,
                    targets_and_aux=targets_and_auxs,
                    metadata=extract_batch_metadata(batch),
                    istep=self.cf.general.istep,
                )

                # TODO re-enable this, need to think on how to make it compatible with
                # student-teacher training
                # if cf.latent_noise_kl_weight > 0.0:
                #     kl = torch.cat([posterior.kl() for posterior in output.latent["posteriors"]])
                #     loss_values.loss += cf.latent_noise_kl_weight * kl.mean()

                [
                    target_aux.update_state_pre_backward(self.cf.general.istep, batch, self.model)
                    for _, target_aux in self.target_and_aux_calculators.items()
                ]
                [
                    target_aux.update_state_pre_backward(self.cf.general.istep, batch, self.model)
                    for _, target_aux in self.target_and_aux_calculators_val.items()
                ]

                # backward pass
                self.optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()

                # gradient clipping
                self.grad_scaler.unscale_(self.optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.training_cfg.optimizer.grad_clip
                )

                # log gradient norms
                if self.log_grad_norms:
                    if bidx % self.train_logging.terminal == 0:
                        self.last_grad_norm = self._get_tensor_item(total_norm)
                    if bidx % self.train_logging.metrics == 0:
                        self._log_instant_grad_norms(TRAIN)

                # optimizer step
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                # update learning rate
                self.lr_scheduler.step()

                batch_size_total = self.get_batch_size_total(self.batch_size_per_gpu)
                step = batch_size_total * self.cf.general.istep

                [
                    target_aux.update_state_post_opt_step(step, batch, self.model)
                    for _, target_aux in self.target_and_aux_calculators.items()
                ]
                [
                    target_aux.update_state_post_opt_step(step, batch, self.model)
                    for _, target_aux in self.target_and_aux_calculators_val.items()
                ]

            # EMA update
            if self.validate_with_ema:
                self.ema_model.update(self.cf.general.istep * batch_size_total, batch_size_total)

            self.perf_tracker.step(
                batch,
                self.cf.general.istep,
                log_fn=lambda m: self.train_logger.log_metrics(
                    TRAIN, m, step=self.cf.general.istep
                ),
            )
            # Compute collapse monitoring metrics
            if self.collapse_monitor.should_compute(self.cf.general.istep):
                self.collapse_monitor._compute_collapse_metrics(
                    self.cf,
                    batch_size_total,
                    self.target_and_aux_calculators,
                    preds,
                    targets_and_auxs,
                )

            self._log_terminal(bidx, mini_epoch, TRAIN)
            if bidx % self.train_logging.metrics == 0:
                self._log(TRAIN)
                # Log collapse metrics
                if self.collapse_monitor.should_log(self.cf.general.istep):
                    self._log_collapse_metrics(TRAIN)

            self.cf.general.istep += 1

            # save model checkpoint (with designation _latest), after the increment so that its
            # istep is the number of completed optimizer steps
            if bidx % self.train_logging.checkpoint == 0 and bidx > 0:
                self.save_model(-1)

            # release this step's batch and outputs before the next forward pass: the model output
            # holds the batch, including every forecast step's target coordinates (~2.3 GB for the
            # FESOM streams), which otherwise stays on the GPU through the next step
            del batch, x, preds, targets_and_auxs, loss

            # stopgap: something still refers to this step's batch through a reference cycle, so the
            # del above does not free it and its ~3 GB of GPU tensors stay allocated until Python's
            # cyclic garbage collector happens to run, sometimes during the next step's
            # backward (OOM). Collecting here frees it before the next forward; remove once
            # the cycle is broken.
            gc.collect()

        self.dataset.advance()

    def validate(self, mini_epoch, mode_cfg, batch_size):
        """
        Perform validation / test computation as specified by mode_cfg
        """

        cf = self.cf
        self.model.eval()

        dataset_val_iter = iter(self.data_loader_validation)

        num_samples_write = mode_cfg.get("output", {}).get("num_samples", 0) * batch_size
        compute_loss = mode_cfg.get("compute_loss", True)
        needs_targets_and_aux = compute_loss or num_samples_write > 0

        with torch.no_grad():
            # print progress bar but only in interactive mode, i.e. when without ddp
            num_val_batches = len(self.data_loader_validation)
            with tqdm.tqdm(total=num_val_batches, disable=self.cf.with_ddp) as pbar:
                for bidx in range(num_val_batches):
                    pbar.set_description_str(f"loading batch {bidx + 1}/{num_val_batches}")
                    pbar.refresh()
                    try:
                        batch = next(dataset_val_iter)
                    except StopIteration:
                        break

                    pbar.set_description_str(f"validating batch {bidx + 1}/{num_val_batches}")
                    pbar.set_postfix_str("moving batch to device")
                    pbar.refresh()

                    if cf.data_loading.get("memory_pinning", False):
                        # pin memory for faster CPU-GPU transfer
                        batch = batch.pin_memory()

                    should_write_output = bidx < num_samples_write
                    lazy = getattr(self.dataset_val, "_lazy_targets", False)
                    if compute_loss:
                        # lazy targets currently only supports the pure-rollout write path
                        assert not lazy, (
                            "Lazy target materialization does not support compute_loss=true; "
                            "set compute_loss=false for chunked long rollouts."
                        )
                        batch.to_device(self.device)
                    else:
                        if should_write_output and not lazy:
                            # eager: build all targets up front; lazy builds them per chunk
                            targets_and_auxs = self._get_output_target_and_auxs(mode_cfg, batch)
                        else:
                            targets_and_auxs = {}
                        batch.to_device(self.device, include_targets=False)
                    # evaluate model
                    with torch.autocast(
                        device_type=f"cuda:{cf.local_rank}",
                        dtype=self.mixed_precision_dtype,
                        enabled=cf.with_mixed_precision,
                    ):
                        pbar.set_postfix_str("running model")
                        pbar.refresh()
                        total_steps = batch.get_output_len()
                        preds_full = (
                            ModelOutput(total_steps, batch=batch.get_source_samples())
                            if compute_loss and total_steps > 0
                            else None
                        )
                        if compute_loss and needs_targets_and_aux:
                            targets_and_auxs = {}
                            for (
                                loss_name,
                                target_aux,
                            ) in self.target_and_aux_calculators_val.items():
                                target_idxs = get_target_idxs_from_cfg(mode_cfg, loss_name)
                                targets_and_auxs[loss_name] = target_aux.compute(
                                    self.cf.general.istep,
                                    batch.get_target_samples(target_idxs),
                                    self.model_params,
                                    self.model,
                                )

                        self._process_validation_chunks(
                            batch,
                            mode_cfg,
                            batch_size,
                            mini_epoch,
                            bidx,
                            targets_and_auxs,
                            preds_full,
                        )

                    if compute_loss and preds_full is not None:
                        _ = self.loss_calculator_val.compute_loss(
                            preds=preds_full,
                            targets_and_aux=targets_and_auxs,
                            metadata=extract_batch_metadata(batch),
                            istep=self.cf.general.istep,
                        )

                    pbar.update(batch_size)

                    if (bidx * batch_size) > mode_cfg.samples_per_mini_epoch:
                        break

                self._log_terminal(0, mini_epoch, VAL)
                self._log(VAL)

        # avoid that there is a systematic bias in the validation subset
        self.dataset_val.advance()

    def _dynamic_loss_emas(self) -> dict[str, DynamicLossEMA]:
        """The dynamic-loss EMAs of the training loss terms (by loss term name)."""
        return {
            loss_term: calculator.dynamic_loss_ema
            for loss_term, calc_term in self.loss_calculator.loss_calculators.items()
            for _, calculator in calc_term
            if getattr(calculator, "dynamic_loss_ema", None) is not None
        }

    def _restore_train_state(self, run_id_contd: str, mini_epoch_contd: int | None, istep: int):
        """
        Restore the optimizer and dynamic-loss state saved with the checkpoint this run resumes
        from; `istep` is the step counter its config recorded. Collective: rank 0 reads the file
        and sends the other ranks their optimizer shards and its dynamic-loss state (all ranks
        continue with rank 0's EMAs). Without a saved state the run continues as before, with
        a new optimizer state and the EMAs at ones.
        """
        mini_epoch = -1 if mini_epoch_contd is None else mini_epoch_contd
        model_path = config.get_path_model(self.cf, run_id=run_id_contd)
        file_state = train_state_path(model_path, run_id_contd, mini_epoch)

        state = load_train_state(file_state) if is_root() else None
        # every rank has to know whether there is a state, restoring the optimizer is collective
        shared = [
            None
            if state is None
            else (state["dynamic_loss"], state["istep"], "ema_teacher" in state)
        ]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(shared, src=0)
        if shared[0] is None:
            return
        dynamic_loss, istep_state, has_ema_teacher = shared[0]
        # every rank holds a full EMA teacher: each reads it from the (memory-mapped) file
        # instead of receiving gigabytes of pickled weights from rank 0
        ema_teacher_state = None
        if has_ema_teacher:
            local_state = state if is_root() else load_train_state(file_state)
            ema_teacher_state = local_state["ema_teacher"]
        self._restore_ema_teachers(ema_teacher_state, file_state)

        load_optimizer_state(self.model, self.optimizer, state["optimizer"] if is_root() else {})
        if is_root():
            if istep_state != istep:
                logger.warning(
                    f"Training state {file_state} was saved at istep={istep_state} but the run "
                    f"config records istep={istep}: the run was probably stopped while saving, "
                    "the checkpoint's weights and optimizer state may not match the step counter."
                )
            logger.info(
                f"Resuming {run_id_contd}: restored optimizer state from {file_state} "
                f"(saved at istep={istep_state})."
            )

        for loss_term, ema in self._dynamic_loss_emas().items():
            if not ema.enabled:
                continue
            saved = dynamic_loss.get(loss_term)
            if not saved:
                if is_root():
                    logger.warning(
                        f"No dynamic-loss state for loss term {loss_term} in {file_state}: its "
                        "channel weights start from ones."
                    )
                continue
            ema.load_state_dict(saved)
            if is_root():
                logger.info(
                    f"Resuming {run_id_contd}: restored dynamic-loss state of loss term "
                    f"{loss_term} for streams {sorted(saved)}."
                )

    def _ema_teachers(self, with_val: bool = True) -> list[EMATeacher]:
        """The EMATeacher target calculators of training (and validation)."""
        calcs = [self.target_and_aux_calculators] + (
            [self.target_and_aux_calculators_val] if with_val else []
        )
        return [
            c for cs in calcs if cs is not None for c in cs.values() if isinstance(c, EMATeacher)
        ]

    def _ema_teacher_state(self) -> dict | None:
        """State of the training EMA teacher (weights + target centering buffers), or None."""
        teachers = self._ema_teachers(with_val=False)
        if not teachers:
            return None
        teacher = teachers[0]
        return {
            "ema_model": {
                k: v.detach().cpu() for k, v in teacher.ema_model.ema_model.state_dict().items()
            },
            "postprocess_targets": {
                name: module.state_dict() for name, module in teacher.postprocess_targets.items()
            },
        }

    def _restore_ema_teachers(self, ema_state: dict | None, file_state) -> None:
        """
        Load a saved EMA teacher into the training and validation teachers. Without one, a
        resumed run keeps the teacher the EMA calculator reset from the student, as before.
        """
        teachers = self._ema_teachers()
        if not teachers:
            return
        if ema_state is None:
            if is_root():
                logger.warning(
                    f"No EMA teacher state in {file_state}: the teacher restarts from the student."
                )
            return
        for teacher in teachers:
            mkeys, ukeys = teacher.ema_model.ema_model.load_state_dict(
                ema_state["ema_model"], strict=False
            )
            for name, module in teacher.postprocess_targets.items():
                if name in ema_state.get("postprocess_targets", {}):
                    module.load_state_dict(ema_state["postprocess_targets"][name], strict=False)
            if is_root() and (mkeys or ukeys):
                logger.warning(f"EMA teacher restore: missing keys {mkeys}, unused keys {ukeys}")
        if is_root():
            logger.info(f"Restored EMA teacher state from {file_state}.")

    def _get_full_model_state_dict(self):
        maybe_sharded_sd = (
            self.model.state_dict() if self.ema_model is None else self.ema_model.state_dict()
        )
        if self.cf.with_ddp and self.cf.with_fsdp:
            cpu_state_dict = {}
            for param_name, sharded_param in maybe_sharded_sd.items():
                full_param = sharded_param.full_tensor()
                if is_root():
                    cpu_state_dict[param_name] = full_param.cpu()
                else:
                    del full_param
            return cpu_state_dict
        else:
            return maybe_sharded_sd

    def save_model(self, mini_epoch: int, name=None):
        # Saving at mini_epoch == max_mini_epoch means that we are saving the latest checkpoint.
        max_mini_epoch = self.training_cfg.num_mini_epochs
        assert mini_epoch <= max_mini_epoch, (mini_epoch, max_mini_epoch)
        model_state_dict = self._get_full_model_state_dict()
        base_path = config.get_path_model(self.cf)

        if is_root():
            filename = "".join(
                [
                    self.cf.general.run_id,
                    "_",
                    "latest" if mini_epoch == -1 else f"chkpt{mini_epoch:05d}",
                    ("_" + name) if name is not None else "",
                ]
            )
            file_out = base_path / (filename + ".chkpt")
            file_tmp = base_path / (filename + "_tmp.chkpt")
            # save temp file (slow)
            torch.save(model_state_dict, file_tmp)
            # move file (which is changing the link in the file system and very fast)
            file_tmp.replace(file_out)
            if is_root():
                logger.info(f"Saved model to {file_out}")
        del model_state_dict

        # optimizer and dynamic-loss state next to the model, continued when the run is resumed
        # (only from unnamed checkpoints); gathering is collective and only rank 0 receives the
        # full state (on its CPU)
        optimizer_state = None
        if name is None:
            optimizer_state = gather_optimizer_state(self.model, self.optimizer)
        if is_root():
            if optimizer_state is not None:
                dynamic_loss = {
                    loss_term: ema.state_dict()
                    for loss_term, ema in self._dynamic_loss_emas().items()
                }
                file_state = train_state_path(base_path, self.cf.general.run_id, mini_epoch)
                save_train_state(
                    file_state,
                    optimizer_state,
                    dynamic_loss,
                    self.cf.general.istep,
                    ema_teacher=self._ema_teacher_state(),
                )
                logger.info(
                    f"Saved optimizer and dynamic-loss state (istep={self.cf.general.istep}) to "
                    f"{file_state}"
                )

            # save config
            config.save(self.cf, mini_epoch)
        del optimizer_state

    def _log(self, stage: Stage):
        """
        Logs training or validation metrics.

        Args:
            stage: Stage Is it's VAL, logs are treated as validation logs.
                        If TRAIN, logs are treated as training logs

        Notes:
            - This method only executes logging on the main process (rank 0).
            - After logging, historical loss and standard deviation records are cleared.
        """
        loss_calculator = self.loss_calculator_val if stage == VAL else self.loss_calculator
        avg_loss, losses_all, stddev_all = prepare_losses_for_logging(
            loss_calculator.loss_hist,
            loss_calculator.losses_unweighted_hist,
            loss_calculator.stddev_unweighted_hist,
        )

        samples = self.cf.general.istep * self.get_batch_size_total(self.batch_size_per_gpu)

        if is_root():
            # plain logger
            if stage == VAL:
                self.train_logger.add_logs(stage, samples, losses_all, stddev_all)

            elif self.cf.general.istep >= 0:
                elapsed_time = time.time() - self.t_training_start
                self.train_logger.add_logs(
                    stage,
                    samples,
                    losses_all,
                    stddev_all,
                    avg_loss=avg_loss,
                    lr=self.lr_scheduler.get_lr(),
                    elapsed_training_time_seconds=elapsed_time,
                    extra_metrics=(
                        {"forecast_prefix_steps": float(np.mean(self._prefix_hist))}
                        if self._prefix_hist
                        else None
                    ),
                )

        loss_calculator.loss_hist = []
        loss_calculator.losses_unweighted_hist = []
        loss_calculator.stddev_unweighted_hist = []

        if stage == TRAIN:
            self._prefix_hist = []

    def _get_tensor_item(self, tensor):
        """
        When using FSDP2, tensor is a DTensor and we need full_tensor().item() instead of .item(),
        see here: https://gist.github.com/Kai-46/a9835ef3f36e76d06afee6c11f388144
        """
        return tensor.full_tensor().item() if isinstance(tensor, DTensor) else tensor.item()

    def _log_instant_grad_norms(self, stage: Stage):
        """
        Log instantaneous grad norms, we do not average because of the cost and because we want to
        measure the actual values.
        """
        grad_norms = {"grad_norm.total": self.last_grad_norm}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norms["grad_norm." + name] = self._get_tensor_item(
                    param.grad.norm() / sqrt(param.numel())
                )

        if is_root():
            self.train_logger.log_metrics(stage, grad_norms)

    def _log_terminal(self, bidx: int, mini_epoch: int, stage: Stage):
        print_freq = self.train_logging.terminal
        if bidx % print_freq == 0 and bidx > 0 or stage == VAL:
            # compute from last iteration
            loss_calculator = self.loss_calculator_val if stage == VAL else self.loss_calculator
            avg_loss, losses_all, _ = prepare_losses_for_logging(
                loss_calculator.loss_hist,
                loss_calculator.losses_unweighted_hist,
                loss_calculator.stddev_unweighted_hist,
            )

            if is_root():
                if stage == VAL:
                    if len(avg_loss) == 0:
                        logger.info(
                            f"validation ({self.cf.general.run_id}) : {mini_epoch:03d} : "
                            "loss disabled"
                        )
                    else:
                        logger.info(
                            f"""validation ({self.cf.general.run_id}) : {mini_epoch:03d} : 
                            {np.nanmean(avg_loss)}"""
                        )

                elif stage == TRAIN:
                    # samples per sec
                    dt = time.time() - self.t_start
                    len_dataset = len(self.data_loader) // self.batch_size_per_gpu
                    pstr = (
                        f"{mini_epoch:03d} : {bidx:05d}/{len_dataset:05d} : "
                        + f"{self.cf.general.istep:06d} : loss = {np.nanmean(avg_loss):.4E} "
                        + f"(lr={self.lr_scheduler.get_lr():.2E}, "
                    )
                    if self.log_grad_norms:
                        pstr += f"gradient norm={self.last_grad_norm:.3f}, "
                    if self._prefix_hist:
                        pstr += f"prefix={self._prefix_hist[-1]}, "
                    pstr += f"s/sec={(print_freq * self.batch_size_per_gpu) / dt:.3f})"
                    logger.info(pstr)
                    logger.info("\t")

                for key, value in losses_all.items():
                    if key.endswith("avg"):
                        val = np.nan if np.isnan(value).all() else f"{np.nanmean(value):0.4E}"
                        logger.info(
                            f"{key} : {val} \t",
                        )
                logger.info("\n")

            self.t_start = time.time()

    def _log_collapse_metrics(self, stage: Stage) -> None:
        """
        Log cached collapse monitoring metrics.
        """
        metrics = self.collapse_monitor.get_cached_metrics()
        if metrics and is_root():
            metrics["num_samples"] = self.cf.general.istep
            self.train_logger.log_metrics(stage, metrics)
