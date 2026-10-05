"""Tests for forcing-stream handling in MultiStreamDataSampler (no data access)."""

from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler


def test_fesom_streams_resolve_to_fesom_reader():
    # develop moved the fesom reader into readers_extra; the sampler must find it there
    cls = MultiStreamDataSampler._get_dataset_class(None, {"type": "fesom"}, "FESOM_NODES")
    assert cls.__name__ == "DataReaderFesom"
    assert cls.__module__ == "weathergen.readers_extra.data_reader_fesom"


def test_latent_output_uses_chunk_relative_step_index():
    import types

    import torch

    from weathergen.utils.validation_io import get_latent_output

    requested = []

    def get_latent_prediction(i):
        requested.append(i)
        return {"head": torch.zeros(1, 3)}

    sample = types.SimpleNamespace(streams_data={"s": types.SimpleNamespace(sample_idx=0)})
    batch = types.SimpleNamespace(
        get_output_idxs=lambda: [1, 2, 3, 4],
        get_source_samples=lambda: types.SimpleNamespace(get_samples=lambda: [sample]),
    )
    model_output = types.SimpleNamespace(get_latent_prediction=get_latent_prediction)

    latents = get_latent_output(batch, model_output, timestep_idxs=[3, 4], fstep_offset=3)

    assert requested == [0, 1]
    assert len(latents) == 2 and latents[0][0]["head"].shape == (3,)
