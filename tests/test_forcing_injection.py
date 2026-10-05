"""Tests for ForcingInjection and its sizing from the stream config."""

import pytest
import torch

from weathergen.model.forcing import ForcingInjection, forcing_num_vars_from_cfg
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


@pytest.mark.parametrize("mode", MODES)
def test_injection_is_a_noop_at_initialisation(mode):
    tokens, condition, field = _inputs()
    out_tokens, out_condition = _injection(mode)(tokens, condition, field, NUM_AUX)
    assert torch.allclose(out_tokens, tokens)
    if mode == "adaln_local":
        assert out_condition.shape == (NUM_AUX + NUM_CELLS, DIM_AUX)
        assert torch.allclose(out_condition, condition.expand_as(out_condition))
    else:
        assert torch.allclose(out_condition, condition)


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
