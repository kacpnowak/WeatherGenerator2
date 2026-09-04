#!/usr/bin/env python
"""seaft epoch selection rule (spec §7) over a full_compare.py output.

usage: seaft_score.py <compare_file> <candidate_tag> <base_tag>
exit 0 = candidate passes on every week in the file, 1 = fails.
"""
import re
import sys

FAMILY = {"zos", "uo_0.494025m", "vo_0.494025m"}
# depths absent from the staged reference (scored vs nearest scoreboard level) - never counted
ARTIFACT_DEPTHS = {"9.573", "15.8101", "25.2114", "34.4342", "65.8073", "130.666"}
LEADS = (1, 5, 10)
FAMILY_MAX = 1.00      # strictly better than base on the family
REST_MAX = 1.01        # everything else within 1%
SST_BIAS_MIN_DELTA = -0.05   # LD1 SST bias may not be colder than base by more than this


def parse(path):
    weeks, cur, cols = {}, None, None
    for line in open(path):
        if line.startswith("==="):
            cur = line.split()[2]
            weeks[cur] = {}
            continue
        if line.startswith("channel/lead"):
            cols = re.findall(r"(.+?_(?:rmse|bias))", line.split(None, 1)[1].replace(" ", ""))
            continue
        m = re.match(r"(\S+)\s+LD(\d+)\s+(.*)", line)
        if not m or cur is None:
            continue
        vals = [float(v) if v != "--" else None for v in m.group(3).split()]
        weeks[cur][(m.group(1), int(m.group(2)))] = dict(zip(cols, vals))
    return weeks


def _is_artifact(ch):
    dep = ch.split("_")[-1].rstrip("m") if "_" in ch else ""
    return dep in ARTIFACT_DEPTHS


def score(path, cand, base):
    out = {}
    for week, rows in parse(path).items():
        fam, rest = [], []
        for (ch, ld), r in rows.items():
            if ld not in LEADS or _is_artifact(ch):
                continue
            a, b = r.get(f"{cand}_rmse"), r.get(f"{base}_rmse")
            if a is None or b is None:
                continue
            (fam if ch in FAMILY else rest).append(a / b)
        sst = rows[("thetao_0.494025m", 1)]
        delta = sst[f"{cand}_bias"] - sst[f"{base}_bias"]
        fam_s, rest_s = sum(fam) / len(fam), sum(rest) / len(rest)
        ok = fam_s < FAMILY_MAX and rest_s <= REST_MAX and delta >= SST_BIAS_MIN_DELTA
        out[week] = {"family": fam_s, "rest": rest_s, "sst_bias_delta": delta, "ok": ok}
    return out


def overall(res):
    return bool(res) and all(v["ok"] for v in res.values())


if __name__ == "__main__":
    path, cand, base = sys.argv[1:4]
    res = score(path, cand, base)
    for week, v in res.items():
        print(f"{week}: family {v['family']:.4f} (<{FAMILY_MAX})  rest {v['rest']:.4f} (<={REST_MAX})  "
              f"SST-bias delta {v['sst_bias_delta']:+.3f} (>={SST_BIAS_MIN_DELTA})  -> {'ok' if v['ok'] else 'FAIL'}")
    print("PASS" if overall(res) else "FAIL")
    sys.exit(0 if overall(res) else 1)
