#!/usr/bin/env python
"""ferec epoch selection rule over a full_compare.py output (spec 2026-09-05 §4).

Keep an epoch only if, on EVERY week in the file:
  overall : mean RMSE ratio (cand/base) over all exact-depth (channel, lead 1/5/10) rows  < 1.00
  zos     : mean RMSE ratio over zos x leads 1/5/10                                       <= 1.02
  bias    : lead-1 SST bias (cand - base)                                                 >= -0.05
usage: ferec_score.py <compare_file> <candidate_tag> <base_tag>   (exit 0 = PASS, 1 = FAIL)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seaft_score import ARTIFACT_DEPTHS, LEADS, parse  # noqa: E402

OVERALL_MAX = 1.00
ZOS_MAX = 1.02
SST_BIAS_MIN_DELTA = -0.05


def _is_artifact(ch):
    dep = ch.split("_")[-1].rstrip("m") if "_" in ch else ""
    return dep in ARTIFACT_DEPTHS


def score(path, cand, base):
    out = {}
    for week, rows in parse(path).items():
        allr, zos = [], []
        for (ch, ld), r in rows.items():
            if ld not in LEADS or _is_artifact(ch):
                continue
            a, b = r.get(f"{cand}_rmse"), r.get(f"{base}_rmse")
            if a is None or b is None:
                continue
            allr.append(a / b)
            if ch == "zos":
                zos.append(a / b)
        sst = rows[("thetao_0.494025m", 1)]
        delta = sst[f"{cand}_bias"] - sst[f"{base}_bias"]
        overall, zos_s = sum(allr) / len(allr), sum(zos) / len(zos)
        ok = overall < OVERALL_MAX and zos_s <= ZOS_MAX and delta >= SST_BIAS_MIN_DELTA
        out[week] = {"overall": overall, "zos": zos_s, "sst_bias_delta": delta, "ok": ok}
    return out


def overall(res):
    return bool(res) and all(v["ok"] for v in res.values())


if __name__ == "__main__":
    path, cand, base = sys.argv[1:4]
    res = score(path, cand, base)
    for week, v in res.items():
        print(f"{week}: overall {v['overall']:.4f} (<{OVERALL_MAX})  zos {v['zos']:.4f} (<={ZOS_MAX})  "
              f"SST-bias delta {v['sst_bias_delta']:+.3f} (>={SST_BIAS_MIN_DELTA})  -> {'ok' if v['ok'] else 'FAIL'}")
    print("PASS" if overall(res) else "FAIL")
    sys.exit(0 if overall(res) else 1)
