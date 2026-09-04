import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
STREAM = REPO / "config/streams/glorys_v3_seaft/glorys.yml"
BASE_STREAM = REPO / "config/streams/glorys_v3/glorys.yml"
RUN_JSON = Path(
    "/e/scratch/weatherai/shared_work/models/glorys_v3_20260819/model_glorys_v3_20260819_chkpt00045.json"
)
EXPECTED = {
    "zos": 6.91, "thetao_0.494025m": 2.44, "thetao_9.573m": 3.16, "thetao_15.8101m": 3.47,
    "thetao_25.2114m": 1.0, "thetao_34.4342m": 1.0, "thetao_47.3737m": 1.36,
    "thetao_65.8073m": 3.28, "thetao_92.3261m": 2.16, "thetao_130.666m": 1.0,
    "thetao_155.851m": 1.54, "thetao_222.475m": 1.81, "thetao_318.127m": 1.52,
    "thetao_453.938m": 1.67, "thetao_541.089m": 1.72, "so_0.494025m": 1.16,
    "so_9.573m": 1.7, "so_15.8101m": 1.82, "so_25.2114m": 1.0, "so_34.4342m": 1.0,
    "so_47.3737m": 1.26, "so_65.8073m": 2.25, "so_92.3261m": 1.0, "so_130.666m": 1.0,
    "so_155.851m": 1.11, "so_222.475m": 1.59, "so_318.127m": 1.74, "so_453.938m": 2.17,
    "so_541.089m": 2.39, "uo_0.494025m": 3.0, "uo_15.8101m": 1.11, "vo_0.494025m": 3.0,
    "vo_15.8101m": 1.02,
}


def _load(path):
    return yaml.safe_load(path.read_text())


def test_weights_mapping_matches_rule_with_family_floor():
    w = _load(STREAM)["GLORYS3"]["target_channel_weights"]
    assert w == EXPECTED


def test_weights_cover_exactly_the_target_channels():
    w = _load(STREAM)["GLORYS3"]["target_channel_weights"]
    targets = json.loads(RUN_JSON.read_text())["streams"]["GLORYS3"]["train_target_channels"]
    assert len(targets) == 33
    assert set(w) == set(targets)


def test_only_weights_differ_from_base_stream():
    seaft = _load(STREAM)["GLORYS3"]
    base = _load(BASE_STREAM)["GLORYS3"]
    seaft.pop("target_channel_weights")
    assert seaft == base


def test_atmo_stream_is_unchanged_era5():
    seaft = (REPO / "config/streams/glorys_v3_seaft/atmo.yml").read_text()
    base = (REPO / "config/streams/glorys_v3/atmo.yml").read_text()
    assert seaft == base
    assert "aifs-ea-an-oper-0001-mars-n320-1979-2024-6h-v1-for-single-v2.zarr" in seaft
