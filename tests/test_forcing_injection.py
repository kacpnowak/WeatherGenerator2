"""Tests for ForcingInjection and its sizing from the stream config."""

import copy

import numpy as np
import pytest
import torch
import torch.distributed as dist

from weathergen.model.forcing import (
    ForcingEmbed,
    ForcingInjection,
    check_one_latent_token_per_cell,
    forcing_num_vars_from_cfg,
)
from weathergen.model.norms import AdaLayerNorm

NUM_CELLS, NUM_VARS, DIM, DIM_AUX, NUM_AUX, BATCH = 12, 3, 16, 8, 2, 2
MODES = ["additive", "cross_attn", "global", "adaln_local"]


def _inputs(num_aux=NUM_AUX):
    torch.manual_seed(0)
    tokens = torch.randn(BATCH, num_aux + NUM_CELLS, DIM)
    condition = torch.randn(DIM_AUX)
    field = torch.cat([torch.randn(NUM_CELLS, NUM_VARS), torch.ones(NUM_CELLS, NUM_VARS)], -1)
    return tokens, condition, field


def _injection(mode):
    torch.manual_seed(1)
    return ForcingInjection(NUM_VARS, DIM, mode, dim_aux=DIM_AUX, num_heads=4)


def _assert_noop(injection, mode):
    tokens, condition, field = _inputs()
    out_tokens, out_condition = injection(tokens, condition, field, NUM_AUX)
    assert torch.allclose(out_tokens, tokens)
    if mode == "adaln_local":
        assert out_condition.shape == (NUM_AUX + NUM_CELLS, DIM_AUX)
        assert torch.allclose(out_condition, condition.expand_as(out_condition))
    else:
        assert torch.allclose(out_condition, condition)


@pytest.mark.parametrize("mode", MODES)
def test_injection_is_a_noop_at_initialisation(mode):
    _assert_noop(_injection(mode), mode)


POISON = 7.0


def _poison(module):
    # what FSDP's meta-device build leaves behind: arbitrary memory in every tensor
    with torch.no_grad():
        for tensor in [*module.parameters(), *module.buffers()]:
            tensor.fill_(POISON)


def _assert_reset(injection, mode):
    assert torch.equal(injection.gate, torch.zeros(1))
    assert torch.equal(injection.embed.mean, torch.zeros(NUM_VARS))
    assert torch.equal(injection.embed.stdev, torch.ones(NUM_VARS))
    for name, param in injection.named_parameters():
        assert not torch.all(param == POISON), f"{name} was not re-initialised"
    _assert_noop(injection, mode)


@pytest.mark.parametrize("mode", MODES)
def test_reset_parameters_restores_the_initial_state(mode):
    injection = _injection(mode)
    _poison(injection)
    injection.reset_parameters()
    _assert_reset(injection, mode)


@pytest.mark.parametrize("mode", MODES)
def test_model_reset_parameters_covers_the_forcing_module(mode):
    from weathergen.model.model import Model

    model = torch.nn.Module()  # stands in for the Model: reset_parameters only walks modules
    model.forcing_module = _injection(mode)
    _poison(model)
    Model.reset_parameters(model)
    _assert_reset(model.forcing_module, mode)


@pytest.mark.parametrize("mode", ["additive", "cross_attn"])
def test_token_modes_change_cell_tokens_only(mode):
    tokens, condition, field = _inputs()
    injection = _injection(mode)
    injection.gate.data.fill_(1.0)
    out_tokens, out_condition = injection(tokens, condition, field, NUM_AUX)
    assert out_tokens.shape == tokens.shape
    assert torch.equal(out_tokens[:, :NUM_AUX], tokens[:, :NUM_AUX])
    assert not torch.allclose(out_tokens[:, NUM_AUX:], tokens[:, NUM_AUX:])
    assert torch.equal(out_condition, condition)


def test_global_mode_changes_the_global_condition():
    tokens, condition, field = _inputs()
    injection = _injection("global")
    injection.gate.data.fill_(1.0)
    out_tokens, out_condition = injection(tokens, condition, field, NUM_AUX)
    assert torch.equal(out_tokens, tokens)
    assert out_condition.shape == (DIM_AUX,)
    assert not torch.allclose(out_condition, condition)


@pytest.mark.parametrize("num_aux", [0, NUM_AUX])
def test_adaln_local_condition_lines_up_with_tokens(num_aux):
    tokens, condition, field = _inputs(num_aux)
    injection = _injection("adaln_local")
    injection.gate.data.fill_(1.0)
    _, out_condition = injection(tokens, condition, field, num_aux)
    assert out_condition.shape == (num_aux + NUM_CELLS, DIM_AUX)
    # register/class tokens carry no forcing: they keep the global condition
    assert torch.allclose(out_condition[:num_aux], condition.expand(num_aux, -1))
    assert not torch.allclose(out_condition[num_aux:], condition.expand(NUM_CELLS, -1))
    # the per-token condition must broadcast inside the forecast engine's AdaLN
    assert AdaLayerNorm(DIM, DIM_AUX)(tokens, out_condition).shape == tokens.shape


@pytest.mark.parametrize("mode", MODES)
def test_forcing_parameters_receive_gradient(mode):
    tokens, condition, field = _inputs()
    injection = _injection(mode)
    optimiser = torch.optim.SGD(injection.parameters(), lr=0.1)

    def loss():
        out_tokens, out_condition = injection(tokens, condition, field, NUM_AUX)
        return out_tokens.pow(2).sum() + out_condition.pow(2).sum()

    loss().backward()
    assert injection.gate.grad.abs().sum() > 0  # the zero gate itself is trainable
    optimiser.step()
    optimiser.zero_grad()
    loss().backward()
    assert injection.embed.mlp[0].weight.grad.abs().sum() > 0  # then the embedding trains


@pytest.mark.parametrize("mode", MODES)
def test_empty_forcing_field_is_a_noop(mode):
    tokens, condition, _ = _inputs()
    out_tokens, out_condition = _injection(mode)(tokens, condition, [], NUM_AUX)
    assert out_tokens is tokens and out_condition is condition


@pytest.mark.parametrize("mode", ["global", "adaln_local"])
def test_adaln_modes_need_condition_stream(mode):
    with pytest.raises(ValueError, match="condition stream"):
        ForcingInjection(NUM_VARS, DIM, mode, dim_aux=0)


def test_forcing_num_vars_from_cfg():
    assert forcing_num_vars_from_cfg({"forcing_num_vars": 36, "train_source_channels": ["a"]}) == 36
    assert forcing_num_vars_from_cfg({"train_source_channels": ["a", "b"]}) == 2
    assert forcing_num_vars_from_cfg({"val_source_channels": ["a", "b", "c"]}) == 3
    assert forcing_num_vars_from_cfg({}) == 0


@pytest.mark.parametrize("mode", MODES)
def test_injection_needs_one_latent_token_per_cell(mode):
    check_one_latent_token_per_cell(mode, ae_local_num_queries=1)
    with pytest.raises(ValueError, match="ae_local_num_queries"):
        check_one_latent_token_per_cell(mode, ae_local_num_queries=2)


def test_disabled_injection_accepts_several_latent_tokens_per_cell():
    check_one_latent_token_per_cell("none", ae_local_num_queries=4)


def _trained(mode):
    """An injection whose output depends on the forcing (open gate)."""
    injection = _injection(mode)
    injection.gate.data.fill_(1.0)
    return injection


def _outputs(injection):
    tokens, condition, field = _inputs()
    return injection(tokens, condition, field, NUM_AUX)


def _assert_same_outputs(injection, reference):
    for out, ref in zip(_outputs(injection), _outputs(reference), strict=True):
        assert torch.equal(out, ref)


@pytest.mark.parametrize("mode", MODES)
def test_forcing_module_state_dict_holds_parameters_only(mode):
    # FSDP2 turns parameters into DTensors but leaves buffers plain tensors, and both the
    # checkpoint save (full_tensor()) and the sharded load (device_mesh) need DTensors
    state_dict = _injection(mode).state_dict()
    assert not [k for k in state_dict if k.endswith(("embed.mean", "embed.stdev"))]
    assert set(state_dict) == {name for name, _ in _injection(mode).named_parameters()}


@pytest.mark.parametrize("mode", MODES)
def test_state_dict_round_trip(mode):
    source, target = _trained(mode), ForcingInjection(NUM_VARS, DIM, mode, DIM_AUX, 4)
    target.load_state_dict(source.state_dict(), strict=True)
    assert torch.equal(target.embed.mean, torch.zeros(NUM_VARS))
    assert torch.equal(target.embed.stdev, torch.ones(NUM_VARS))
    _assert_same_outputs(target, source)


@pytest.mark.parametrize("mode", MODES)
def test_state_dict_copy_into_an_emptied_module(mode):
    # EMAModel.reset: to_empty (arbitrary memory, here NaN) then load the student's state dict
    source = _trained(mode)
    copied = copy.deepcopy(source).to_empty(device="cpu")
    with torch.no_grad():
        for tensor in [*copied.parameters(), *copied.buffers()]:
            tensor.fill_(float("nan"))
    copied.load_state_dict(source.state_dict())
    _assert_same_outputs(copied, source)


@pytest.mark.parametrize("mode", MODES)
def test_load_into_a_meta_built_module(mode):
    # continuing a forced run under FSDP: meta build, then load_state_dict(assign=True)
    source = _trained(mode)
    with torch.device("meta"):
        loaded = ForcingInjection(NUM_VARS, DIM, mode, DIM_AUX, 4)
    loaded.load_state_dict(source.state_dict(), assign=True)
    _assert_same_outputs(loaded, source)


def test_forcing_embed_with_explicit_statistics_normalises():
    torch.manual_seed(2)
    stats = ForcingEmbed(NUM_VARS, DIM, np.full(NUM_VARS, 2.0), np.full(NUM_VARS, 4.0))
    plain = ForcingEmbed(NUM_VARS, DIM)
    plain.load_state_dict(stats.state_dict())
    values, valid = torch.randn(NUM_CELLS, NUM_VARS), torch.ones(NUM_CELLS, NUM_VARS)
    assert torch.allclose(stats(values, valid), plain((values - 2.0) / 4.0, valid))


def test_fully_sharded_forcing_module_state_dict_gathers(tmp_path):
    # the checkpoint save loop under FSDP2, on a single CPU rank
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    if dist.is_initialized():
        pytest.skip("a process group is already initialised")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        model = torch.nn.Module()
        model.forcing_module = _injection("cross_attn")
        fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
        full = {name: t.full_tensor() for name, t in model.state_dict().items()}
        assert set(full) == {
            f"forcing_module.{n}" for n, _ in _injection("cross_attn").named_parameters()
        }
    finally:
        dist.destroy_process_group()
