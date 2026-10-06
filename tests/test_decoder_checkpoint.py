"""Decoding under one checkpoint per stream gives the same predictions and gradients."""

import functools
import types

import torch
from torch.utils.checkpoint import checkpoint

from weathergen.model.model import Model

NUM_CELLS, DIM, COORD_DIM, CHANNELS = 12, 8, 5, 3
STREAMS = {"A": [4, 3], "B": [2, 5]}  # target points per sample, batch of 2


class _Engine(torch.nn.Module):
    """Stand-in decoder: target tokens plus a projection of their latent, nested checkpoint."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(DIM, DIM)

    def _block(self, latent, output):
        return output + self.lin(latent).mean(0)

    def forward(self, latent, output, latent_lens, output_lens, coordinates):
        assert latent.shape == (2 * NUM_CELLS * 9, DIM) and int(latent_lens[0]) == 0
        return checkpoint(self._block, latent, output, use_reentrant=False)


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(DIM, CHANNELS)

    def forward(self, x):
        return self.lin(x).unsqueeze(0)  # add ensemble dim


class _Output:
    def __init__(self):
        self.physical = {}

    def add_physical_prediction(self, step, name, pred):
        self.physical[name] = pred


def _setup():
    torch.manual_seed(0)
    stub = types.SimpleNamespace(
        num_aux_tokens=0,
        num_healpix_cells=NUM_CELLS,
        cf=types.SimpleNamespace(
            ae_local_num_queries=1, decoder_type="PerceiverIOCoordConditioning"
        ),
        data_stream_names=list(STREAMS),
        embed_target_coords=torch.nn.ModuleDict(
            {
                n: torch.nn.Sequential(torch.nn.Linear(COORD_DIM, DIM), torch.nn.Dropout(0.3))
                for n in STREAMS
            }
        ),
        target_token_engines=torch.nn.ModuleDict({n: _Engine() for n in STREAMS}),
        pred_heads=torch.nn.ModuleDict({n: _Head() for n in STREAMS}),
    )
    stub._embed_target_coords = functools.partial(Model._embed_target_coords, stub)
    stub._decode_stream = functools.partial(Model._decode_stream, stub)
    for module in (stub.embed_target_coords, stub.target_token_engines, stub.pred_heads):
        module.train()

    samples = [
        types.SimpleNamespace(
            streams_data={
                n: types.SimpleNamespace(
                    target_coords=[None, torch.randn(sizes[i], COORD_DIM)],
                    target_coords_lens=[None, torch.tensor([sizes[i]], dtype=torch.int32)],
                )
                for n, sizes in STREAMS.items()
            }
        )
        for i in range(2)
    ]
    batch = type("Batch", (), {"samples": samples, "__len__": lambda self: 2})()
    params = types.SimpleNamespace(hp_nbours=torch.randint(0, NUM_CELLS, (NUM_CELLS, 9)))
    tokens = torch.randn(2, NUM_CELLS, DIM, requires_grad=True)
    return stub, batch, params, tokens


def _reference(stub, batch, params, tokens):
    """The decode step written out without any outer checkpoint (the previous code path)."""
    idxs = params.hp_nbours.unsqueeze(0).repeat((2, 1, 1)).flatten(0, 1)
    nbors = tokens.reshape(2, NUM_CELLS, 1, DIM).flatten(0, 1)[idxs.flatten()].flatten(0, 1)
    nbors_lens = torch.full((2 * NUM_CELLS + 1,), 9, dtype=torch.int32)
    nbors_lens[0] = 0
    preds = {}
    for n in STREAMS:
        t_coords = torch.cat([s.streams_data[n].target_coords[1] for s in batch.samples])
        tcls = torch.cat([s.streams_data[n].target_coords_lens[1] for s in batch.samples])
        tc = stub.embed_target_coords[n](t_coords)
        tc = stub.target_token_engines[n](
            latent=nbors,
            output=tc,
            latent_lens=nbors_lens,
            output_lens=torch.cat([torch.zeros(1, dtype=torch.int32), tcls]),
            coordinates=t_coords,
        )
        preds[n] = torch.split(stub.pred_heads[n](tc), STREAMS[n], dim=1)
    return preds


def _loss_and_grads(stub, tokens, preds):
    loss = sum((p**2).sum() for pred in preds.values() for p in pred)
    modules = (stub.embed_target_coords, stub.target_token_engines, stub.pred_heads)
    params = [p for m in modules for p in m.parameters()]
    grads = torch.autograd.grad(loss, [tokens, *params])
    return loss.detach(), grads


def test_checkpointed_decode_matches_reference_values_and_gradients():
    stub, batch, params, tokens = _setup()

    torch.manual_seed(123)
    out = _Output()
    Model.predict_decoders(stub, params, 1, 1, tokens, batch, out)
    loss, grads = _loss_and_grads(stub, tokens, out.physical)

    torch.manual_seed(123)  # same dropout masks as the checkpointed forward
    ref = _reference(stub, batch, params, tokens)
    ref_loss, ref_grads = _loss_and_grads(stub, tokens, ref)

    for n in STREAMS:
        assert [p.shape for p in out.physical[n]] == [(1, k, CHANNELS) for k in STREAMS[n]]
        for p, q in zip(out.physical[n], ref[n], strict=True):
            torch.testing.assert_close(p, q)
    torch.testing.assert_close(loss, ref_loss)
    for g, h in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(g, h)
    assert grads[0].abs().sum() > 0  # gradient reaches the latent tokens


def test_decode_runs_without_grad():
    stub, batch, params, tokens = _setup()
    out = _Output()
    with torch.no_grad():
        Model.predict_decoders(stub, params, 1, 1, tokens.detach(), batch, out)
    assert set(out.physical) == set(STREAMS)


def test_rollout_decode_recompute_under_fsdp(tmp_path):
    """
    Under FSDP the backward pass recomputes an earlier rollout step's decode after the decoder's
    units have resharded (and skip the unshard in their pre-forward hook): the decode step has
    to gather their parameters itself.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        stub, batch, params, tokens = _setup()

        class Root(torch.nn.Module):
            """Stands in for the model: decodes two rollout steps with the same decoders."""

            def __init__(self):
                super().__init__()
                self.embed_target_coords = stub.embed_target_coords
                self.target_token_engines = stub.target_token_engines
                self.pred_heads = stub.pred_heads

            def forward(self, tokens):
                outputs = [_Output(), _Output()]
                Model.predict_decoders(stub, params, 1, 1, tokens, batch, outputs[0])
                Model.predict_decoders(stub, params, 1, 1, 2.0 * tokens, batch, outputs[1])
                return sum(
                    (p**2).sum() for o in outputs for pred in o.physical.values() for p in pred
                )

        mesh = init_device_mesh("cpu", (1,))
        root = Root()
        for engine in root.target_token_engines.values():
            fully_shard(engine.lin, mesh=mesh)  # like the decoder's attention and MLP units
        fully_shard(root, mesh=mesh)

        root(tokens).backward()

        assert tokens.grad is not None and tokens.grad.abs().sum() > 0
        for engine in root.target_token_engines.values():
            assert engine.lin.weight.grad is not None
    finally:
        dist.destroy_process_group()
