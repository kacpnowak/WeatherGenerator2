import importlib.util
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/seaft_score.py"
spec = importlib.util.spec_from_file_location("seaft_score", SCRIPT)
seaft_score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seaft_score)

HEADER = "channel/lead              candA_rmse  candA_bias  base_rmse  base_bias\n"


def _file(tmp_path, rows_by_week):
    out = ""
    for week, rows in rows_by_week.items():
        out += f"=== week {week} ===\n" + HEADER
        for ch, ld, cr, cb, br, bb in rows:
            out += f"{ch:24s}LD{ld:<3d}{cr:12.4f}{cb:12.4f}{br:12.4f}{bb:12.4f}\n"
    p = tmp_path / "cmp.txt"
    p.write_text(out)
    return p


def _rows(fam_ratio, rest_ratio, sst_bias_cand, sst_bias_base=-0.20):
    rows = []
    for ld in (1, 5, 10):
        rows.append(("zos", ld, 0.06 * fam_ratio, -0.03, 0.06, -0.03))
        rows.append(("uo_0.494025m", ld, 0.11 * fam_ratio, 0.0, 0.11, 0.0))
        rows.append(("vo_0.494025m", ld, 0.11 * fam_ratio, 0.0, 0.11, 0.0))
        rows.append(("thetao_0.494025m", ld, 0.60 * rest_ratio, sst_bias_cand, 0.60, sst_bias_base))
        rows.append(("so_541.089m", ld, 0.08 * rest_ratio, 0.0, 0.08, 0.0))
        rows.append(("thetao_9.573m", ld, 5.0, -1.0, 1.0, -1.0))  # artifact depth: must be ignored
    return rows


def _rows_with_exact_rest(fam_ratio, rest_ratio, sst_bias_cand, sst_bias_base=-0.20):
    """Like _rows but with base values that divide exactly for precise rest ratio testing."""
    rows = []
    for ld in (1, 5, 10):
        rows.append(("zos", ld, 0.06 * fam_ratio, -0.03, 0.06, -0.03))
        rows.append(("uo_0.494025m", ld, 0.11 * fam_ratio, 0.0, 0.11, 0.0))
        rows.append(("vo_0.494025m", ld, 0.11 * fam_ratio, 0.0, 0.11, 0.0))
        rows.append(("thetao_0.494025m", ld, 1.0 * rest_ratio, sst_bias_cand, 1.0, sst_bias_base))
        rows.append(("so_541.089m", ld, 0.5 * rest_ratio, 0.0, 0.5, 0.0))
        rows.append(("thetao_9.573m", ld, 5.0, -1.0, 1.0, -1.0))  # artifact depth: must be ignored
    return rows


def test_passing_epoch(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(0.95, 1.00, -0.21), "20240703": _rows(0.97, 1.005, -0.22)})
    res = seaft_score.score(p, "candA", "base")
    assert res["20240103"]["ok"] and res["20240703"]["ok"]
    assert abs(res["20240103"]["family"] - 0.95) < 1e-6
    assert abs(res["20240103"]["rest"] - 1.00) < 1e-6
    assert abs(res["20240103"]["sst_bias_delta"] - (-0.01)) < 1e-6
    assert seaft_score.overall(res) is True


def test_fails_on_family_not_improved(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(1.00, 1.00, -0.20), "20240703": _rows(0.9, 1.0, -0.2)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is False


def test_fails_on_rest_regression(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(0.9, 1.02, -0.20), "20240703": _rows(0.9, 1.0, -0.2)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is False


def test_fails_on_cold_bias(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(0.9, 1.0, -0.26), "20240703": _rows(0.9, 1.0, -0.2)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is False


def test_artifact_depths_are_ignored(tmp_path):
    # the thetao_9.573m row has ratio 5.0; if it were counted, rest would blow past 1.01
    p = _file(tmp_path, {"20240103": _rows(0.9, 1.0, -0.2)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is True


def test_rest_exactly_at_threshold_passes(tmp_path):
    # rest_ratio=1.01 exactly should pass (rule is <=1.01)
    p = _file(tmp_path, {"20240103": _rows_with_exact_rest(0.9, 1.01, -0.20), "20240703": _rows_with_exact_rest(0.9, 1.01, -0.20)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is True


def test_rest_just_past_threshold_fails(tmp_path):
    # rest_ratio=1.0101 should fail (exceeds <=1.01 threshold)
    p = _file(tmp_path, {"20240103": _rows_with_exact_rest(0.9, 1.0101, -0.20), "20240703": _rows_with_exact_rest(0.9, 1.0, -0.20)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is False


def test_bias_exactly_at_threshold_passes(tmp_path):
    # sst_bias_delta = -0.25 - (-0.20) = -0.05 exactly should pass (rule is >=-0.05)
    p = _file(tmp_path, {"20240103": _rows(0.9, 1.0, -0.25), "20240703": _rows(0.9, 1.0, -0.25)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is True


def test_bias_just_past_threshold_fails(tmp_path):
    # sst_bias_delta = -0.2501 - (-0.20) = -0.0501 should fail (exceeds >=-0.05 threshold)
    p = _file(tmp_path, {"20240103": _rows(0.9, 1.0, -0.2501), "20240703": _rows(0.9, 1.0, -0.20)})
    assert seaft_score.overall(seaft_score.score(p, "candA", "base")) is False
