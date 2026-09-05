import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ferec_score.py"
spec = importlib.util.spec_from_file_location("ferec_score", SCRIPT)
ferec_score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ferec_score)

HEADER = "channel/lead              cand_rmse  cand_bias  base_rmse  base_bias\n"


def _file(tmp_path, rows_by_week):
    out = ""
    for week, rows in rows_by_week.items():
        out += f"=== week {week} ===\n" + HEADER
        for ch, ld, cr, cb, br, bb in rows:
            out += f"{ch:24s}LD{ld:<3d}{cr:12.4f}{cb:12.4f}{br:12.4f}{bb:12.4f}\n"
    p = tmp_path / "cmp.txt"
    p.write_text(out)
    return p


def _rows(zos_ratio, other_ratio, sst_bias_cand, sst_bias_base=0.0):
    # base values 1.0 / 0.5 divide exactly so pooled ratios are bit-exact
    rows = []
    for ld in (1, 5, 10):
        rows.append(("zos", ld, 1.0 * zos_ratio, 0.0, 1.0, 0.0))
        rows.append(("thetao_0.494025m", ld, 1.0 * other_ratio, sst_bias_cand, 1.0, sst_bias_base))
        rows.append(("so_541.089m", ld, 0.5 * other_ratio, 0.0, 0.5, 0.0))
        rows.append(("thetao_9.573m", ld, 5.0, -1.0, 1.0, -1.0))  # artifact depth: ignored
    return rows


def test_pass_when_everything_slightly_better(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(0.99, 0.98, -0.01), "20240703": _rows(1.0, 0.97, 0.0)})
    res = ferec_score.score(p, "cand", "base")
    assert ferec_score.overall(res) is True
    assert abs(res["20240103"]["zos"] - 0.99) < 1e-12


def test_fail_when_overall_not_better(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(1.0, 1.0, 0.0)})
    assert ferec_score.overall(ferec_score.score(p, "cand", "base")) is False


def test_zos_boundary_inclusive_at_1_02(tmp_path):
    # zos exactly 1.02 passes if the overall mean is still < 1.00
    p = _file(tmp_path, {"20240103": _rows(1.02, 0.9, 0.0)})
    assert ferec_score.overall(ferec_score.score(p, "cand", "base")) is True
    p = _file(tmp_path, {"20240103": _rows(1.0201, 0.9, 0.0)})
    assert ferec_score.overall(ferec_score.score(p, "cand", "base")) is False


def test_bias_boundary_bit_exact(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(0.99, 0.98, -0.05)})
    assert ferec_score.overall(ferec_score.score(p, "cand", "base")) is True
    p = _file(tmp_path, {"20240103": _rows(0.99, 0.98, -0.0501)})
    assert ferec_score.overall(ferec_score.score(p, "cand", "base")) is False


def test_every_week_must_pass(tmp_path):
    p = _file(tmp_path, {"20240103": _rows(0.99, 0.98, 0.0), "20240703": _rows(0.99, 1.05, 0.0)})
    assert ferec_score.overall(ferec_score.score(p, "cand", "base")) is False
