"""The run config is written atomically so a killed job never leaves a truncated JSON."""

import json

import pytest
from omegaconf import OmegaConf

from weathergen.common import config as wg_config


def _config(tmp_path, **extra):
    return OmegaConf.create(
        {"run_id": "atomic01", "full_model_path": str(tmp_path / "model"), **extra}
    )


def test_save_writes_readable_json(tmp_path):
    cf = _config(tmp_path, lr=0.5)
    wg_config.save(cf, 3)
    model_dir = tmp_path / "model"
    files = list(model_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"
    assert json.loads(files[0].read_text())["lr"] == 0.5
    assert not list(model_dir.glob("*.tmp"))


def test_save_leaves_no_partial_file(tmp_path, monkeypatch):
    wg_config.save(_config(tmp_path, lr=0.5), 3)
    model_dir = tmp_path / "model"
    (target,) = model_dir.iterdir()
    first = target.read_text()

    # a non-str payload makes the file write itself raise, after the file has been opened
    monkeypatch.setattr(wg_config.json, "dumps", lambda *args, **kwargs: 123)
    with pytest.raises(TypeError):
        wg_config.save(_config(tmp_path, lr=0.9), 3)
    monkeypatch.undo()

    assert target.read_text() == first
    assert json.loads(first)["lr"] == 0.5
    assert list(model_dir.iterdir()) == [target]
