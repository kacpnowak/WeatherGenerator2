from pathlib import Path
from datetime import date
from enum import Enum
from typing import Any
from pydantic import BaseModel, PositiveInt

class EmbedOrientation(Enum):
    CHANNELS = "channels"
    COLUMNS = "columns"

class NormType(Enum):
    LAYERNORM = "LayerNorm"
    RMSNORM = "RMSNorm"

class Embed(BaseModel):
    net: str
    num_tokens: PositiveInt
    num_heads: PositiveInt
    dim_embed: PositiveInt
    num_blocks: PositiveInt

class EmbedTargetCoords(BaseModel):
    net: str
    dim_embed: PositiveInt

class TargetReadout(BaseModel):
    type: str
    num_layers: PositiveInt
    num_heads: PositiveInt

class PredHead(BaseModel):
    ens_size: PositiveInt
    num_layers: PositiveInt

class Stream(BaseModel):
    type: str
    filenames: list[Path]
    loss_weight: float
    source_variables: list[str|None]
    target_variables: list[str|None]
    diagnostic: bool
    masking_rate: float
    masking_rate_none: float
    token_size: PositiveInt
    embed: Embed
    embed_target_coords: EmbedTargetCoords
    target_readout: TargetReadout
    pred_head: PredHead
    name: str    


class Configuration(BaseModel):
    streams_directory: Path

    embed_orientation: str #EmbedOrientation
    embed_local_coords: bool
    embed_centroids_local_coords: bool
    embed_size_centroids: PositiveInt
    embed_unembed_mode: str

    target_cell_local_prediction: bool
    target_coords_local: bool

    ae_local_dim_embed: PositiveInt
    ae_local_num_blocks: PositiveInt
    ae_local_num_heads: PositiveInt
    ae_local_dropout_rate: float
    ae_local_with_qk_lnorm: bool

    ae_local_num_queries: PositiveInt
    ae_local_queries_per_cell: bool
    ae_adapter_num_heads: PositiveInt
    ae_adapter_embed: PositiveInt
    ae_adapter_with_qk_lnorm: bool
    ae_adapter_with_residual: bool
    ae_adapter_dropout_rate: float

    ae_global_dim_embed: PositiveInt
    ae_global_num_blocks: PositiveInt
    ae_global_num_heads: PositiveInt
    ae_global_dropout_rate: float
    ae_global_with_qk_lnorm: bool
    ae_global_att_dense_rate: float
    ae_global_block_factor: PositiveInt
    ae_global_mlp_hidden_factor: PositiveInt

    pred_adapter_kv: bool
    pred_self_attention: bool
    pred_dyadic_dims: bool
    pred_mlp_adaln: bool

    forecast_delta_hrs: int
    forecast_steps: int
    forecast_policy: str|None
    forecast_freeze_model: bool
    forecast_att_dense_rate: float

    fe_num_blocks: int
    fe_num_heads: PositiveInt
    fe_dropout_rate: float
    fe_with_qk_lnorm: bool

    healpix_level: PositiveInt

    with_mixed_precision: bool
    with_flash_attention: bool
    compile_model: bool

    with_fsdp: bool

    loss_fcts: list[list[str|float]]
    loss_fcts_val: list[list[str|float]]

    batch_size: PositiveInt
    batch_size_validation: PositiveInt

    masking_mode: str
    masking_rate: float
    masking_rate_sampling: bool
    sampling_rate_target: float

    num_epochs: PositiveInt
    samples_per_epoch: PositiveInt
    samples_per_validation: PositiveInt
    shuffle: bool

    lr_scaling_policy: str
    lr_start: float
    lr_max: float
    lr_final_decay: float
    lr_final: float
    lr_steps_warmup: PositiveInt
    lr_steps_cooldown: PositiveInt
    lr_policy_warmup: str
    lr_policy_decay: str
    lr_policy_cooldown: str

    grad_clip: float
    weight_decay: float
    norm_type: str #NormType
    nn_module: str

    data_path: Path
    start_date: PositiveInt #date
    end_date: PositiveInt #date
    start_date_val: PositiveInt #date
    end_date_val: PositiveInt #date
    len_hrs: PositiveInt
    step_hrs: PositiveInt
    input_window_steps: PositiveInt

    val_initial: bool

    loader_num_workers: PositiveInt
    data_loader_rng_seed: PositiveInt
    log_validation: int

    istep: PositiveInt
    run_history: list[Any]

    run_id: str
    desc: str

    rank: int
    num_ranks: int
    with_ddp: bool
    streams: list[Stream]
    lr_steps: PositiveInt

def read(fname: str):
    "Reads JSON file at path fname"
    json_string = Path(fname).read_text()
    config = Configuration.model_validate_json(json_string)
    return config.model_dump()

def write(cf: dict, fname: str):
    "Writes JSON file at path fname"
    config = Configuration(**cf)
    with open(Path(fname), "w") as f:
    f.write(config.model_dump_json(indent=2))