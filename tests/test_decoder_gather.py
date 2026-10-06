"""The classic decoder may gather the latent neighbourhood inside its checkpoints."""

import gc
import types

import pytest
import torch

from weathergen.model.attention import MultiCrossAttentionHeadVarlen
from weathergen.model.engines import TargetPredictionEngineClassic
from weathergen.model.model import Model

CELLS, DIM, POINTS = 12, 8, 7


class _CrossAttention(MultiCrossAttentionHeadVarlen):
    """Stand-in for the flash-attention block: target tokens plus a projection of the latent."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.lin = torch.nn.Linear(DIM, DIM)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x_q, x_kv, x_q_lens=None, x_kv_lens=None, ada_ln_aux=None):
        assert x_kv.shape == (CELLS * 9, DIM)
        return x_q + self.dropout(self.lin(x_kv)).mean(0)


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(DIM, DIM)

    def forward(self, *args):
        return args[0] + self.lin(args[0])


def _engine():
    engine = TargetPredictionEngineClassic.__new__(TargetPredictionEngineClassic)
    torch.nn.Module.__init__(engine)
    engine.cf = types.SimpleNamespace(pred_self_attention=False)
    engine.tte = torch.nn.ModuleList([_CrossAttention(), _MLP(), _CrossAttention(), _MLP()])
    engine.train()
    return engine


def _inputs():
    torch.manual_seed(0)
    cells = torch.randn(CELLS, 1, DIM, requires_grad=True)  # (cells, queries per cell, dim)
    idxs = torch.randint(0, CELLS, (CELLS * 9,))
    targets = torch.randn(POINTS, DIM, requires_grad=True)
    lens = torch.full((CELLS + 1,), 9, dtype=torch.int32)
    lens[0] = 0
    return cells, idxs, targets, lens


def _run(engine, cells, idxs, targets, lens, gather_inside):
    latent = cells if gather_inside else cells[idxs].flatten(0, 1)
    kwargs = {"latent_idxs": idxs} if gather_inside else {}
    torch.manual_seed(123)  # same dropout masks in both runs
    out = engine(latent, targets, lens, torch.tensor([0, POINTS]), torch.zeros(POINTS, 3), **kwargs)
    grads = torch.autograd.grad((out**2).sum(), [cells, targets, *engine.parameters()])
    return out.detach(), grads


def test_gather_inside_matches_pre_gathered_latent():
    engine = _engine()
    cells, idxs, targets, lens = _inputs()
    out, grads = _run(engine, cells, idxs, targets, lens, gather_inside=True)
    ref_out, ref_grads = _run(engine, cells, idxs, targets, lens, gather_inside=False)
    torch.testing.assert_close(out, ref_out)
    for g, h in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(g, h)
    assert grads[0].abs().sum() > 0


def _gathered_tensors_alive():
    gc.collect()
    return sum(
        1
        for o in gc.get_objects()
        # type() rather than isinstance(): isinstance trips deprecated torch proxy objects
        if issubclass(type(o), torch.Tensor) and tuple(o.shape) == (CELLS * 9, DIM)
    )


@pytest.mark.parametrize("gather_inside", [True, False])
def test_gathered_latent_is_not_kept_for_backward(gather_inside):
    engine = _engine()
    cells, idxs, targets, lens = _inputs()
    latent = cells if gather_inside else cells[idxs].flatten(0, 1)
    kwargs = {"latent_idxs": idxs} if gather_inside else {}
    out = engine(latent, targets, lens, torch.tensor([0, POINTS]), torch.zeros(POINTS, 3), **kwargs)
    del latent
    # with the gather inside the checkpoints, nothing of the gathered size outlives the forward
    assert (_gathered_tensors_alive() == 0) is gather_inside
    (out**2).sum().backward()
    assert cells.grad is not None


class _RecordingEngine(torch.nn.Module):
    def __init__(self, accepts_idxs):
        super().__init__()
        self.accepts_idxs = accepts_idxs
        self.calls = []

    def forward(self, latent, output, latent_lens, output_lens, coordinates, **kwargs):
        assert self.accepts_idxs or not kwargs
        self.calls.append((latent, kwargs.get("latent_idxs")))
        return output


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(DIM, 3)

    def forward(self, x):
        return self.lin(x).unsqueeze(0)  # add ensemble dim


class _Output:
    def __init__(self):
        self.physical = {}

    def add_physical_prediction(self, step, name, pred):
        self.physical[name] = pred


@pytest.mark.parametrize(
    "decoder_type, gathers_inside",
    [("PerceiverIOCoordConditioning", True), ("PerceiverIO", False)],
)
def test_predict_decoders_hands_the_same_neighbourhood(decoder_type, gathers_inside):
    torch.manual_seed(0)
    engine = _RecordingEngine(accepts_idxs=gathers_inside)
    stub = types.SimpleNamespace(
        num_aux_tokens=0,
        num_healpix_cells=CELLS,
        cf=types.SimpleNamespace(ae_local_num_queries=1, decoder_type=decoder_type),
        data_stream_names=["S"],
        embed_target_coords=torch.nn.ModuleDict({"S": torch.nn.Linear(5, DIM)}),
        target_token_engines=torch.nn.ModuleDict({"S": engine}),
        pred_heads=torch.nn.ModuleDict({"S": _Head()}),
    )
    sample = types.SimpleNamespace(
        streams_data={
            "S": types.SimpleNamespace(
                target_coords=[None, torch.randn(POINTS, 5)],
                target_coords_lens=[None, torch.tensor([POINTS], dtype=torch.int32)],
            )
        }
    )
    batch = type("Batch", (), {"samples": [sample], "__len__": lambda self: 1})()
    params = types.SimpleNamespace(hp_nbours=torch.randint(0, CELLS, (CELLS, 9)))
    tokens = torch.randn(1, CELLS, DIM)

    Model.predict_decoders(stub, params, 1, 1, tokens, batch, _Output())

    expected = tokens.reshape(1, CELLS, 1, DIM).flatten(0, 1)[params.hp_nbours.flatten()]
    expected = expected.flatten(0, 1)
    (latent, latent_idxs), = engine.calls
    if gathers_inside:
        torch.testing.assert_close(latent[latent_idxs].flatten(0, 1), expected)
    else:
        assert latent_idxs is None
        torch.testing.assert_close(latent, expected)
