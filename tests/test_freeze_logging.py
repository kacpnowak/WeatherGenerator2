import logging

import torch.nn as nn

from weathergen.model.utils import apply_fct_to_blocks, freeze_weights

REGEX = ".*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*forecast_engine.*|latent_heads.*|latent_pre_norm|StreamEmbedder_.*"


class Named(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.lin = nn.Linear(2, 2)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.ae_local_engine = nn.Linear(2, 2)
        self.encoder.ae_aggregation_engine = nn.Linear(2, 2)
        self.encoder.embed_engine = nn.ModuleDict({"GLORYS3": Named("StreamEmbedder_GLORYS3")})
        self.target_token_engines = nn.ModuleDict({"GLORYS3": Named("TargetPredictionEngine_GLORYS3")})
        self.target_token_engines["GLORYS3"].latent_in_norm = nn.LayerNorm(2)
        self.latent_pre_norm = nn.LayerNorm(2)


def _trainable(m):
    return any(p.requires_grad for p in m.parameters())


def test_freeze_matches_named_and_dotted_modules_and_logs_each(caplog):
    model = Toy()
    with caplog.at_level(logging.INFO, logger="weathergen.model.utils"):
        apply_fct_to_blocks(model, REGEX, freeze_weights)
    logged = {r.getMessage() for r in caplog.records if r.getMessage().startswith("freeze_weights: ")}
    assert "freeze_weights: encoder.ae_local_engine" in logged
    assert "freeze_weights: StreamEmbedder_GLORYS3" in logged
    assert "freeze_weights: latent_pre_norm" in logged
    assert not _trainable(model.encoder.ae_local_engine)
    assert not _trainable(model.encoder.embed_engine["GLORYS3"])
    assert not _trainable(model.latent_pre_norm)
    assert _trainable(model.encoder.ae_aggregation_engine)
    assert _trainable(model.target_token_engines["GLORYS3"].lin)
    assert _trainable(model.target_token_engines["GLORYS3"].latent_in_norm)
