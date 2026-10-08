"""The resumable rollout state lives in the run's results directory."""

import types

from weathergen.common import config
from weathergen.train.trainer import Trainer


def test_rollout_state_path_is_in_results_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_path_results", lambda cf: tmp_path / cf.general.run_id)
    trainer = Trainer.__new__(Trainer)
    trainer.cf = types.SimpleNamespace(general=types.SimpleNamespace(run_id="abcd1234"))
    assert trainer._rollout_state_path() == tmp_path / "abcd1234" / "rollout_state.pt"
