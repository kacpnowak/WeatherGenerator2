"""Training loss vs spin-up length from a run's train_metrics.json (CPU, read-only).

    python spinup_loss.py --runs fvars003 fvars004 --out DIR [--fig DIR]

The metrics file holds one line per `train_logging.metrics` training steps. Every per-channel
loss key ends with the absolute forecast step k of the rollout, counted from the encoded
initial state: a training step with spin-up length p (forecast.prefix_steps, no gradient)
supervises k = p+1 .. p+15. Each key is the mean over the line's training steps (DDP-averaged;
the spin-up length is identical on all ranks, see draw_prefix_steps) that supervised step k.
The loss of a step does not depend on whether it is the first supervised step or a later one
(the forward is the same computation), so the value at key k is, in expectation, the loss of
the first supervised step after a spin-up of s = k-1 days. Binning is on s.

Values are the unweighted per-channel MSE in normalised units (LossPhysical.<STREAM>.mse.<ch>),
before the dynamic (EMA) channel weights. Lines are grouped into mini-epochs by the validation
lines that end every mini-epoch; each block is labelled with the checkpoint written at its end.
Every (line, key) counts once (the number of training steps behind a key is not logged).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BINS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 181)]
FINE = [(a, a + 10) for a in range(0, 190, 10)]
STREAMS = ["SALT", "TEMP", "U", "V", "SURFACE"]
CHANNELS = ["salt_2", "salt_75", "salt_125", "salt_450", "salt_975", "salt_2035", "temp_2",
            "temp_75", "temp_450", "temp_975", "ssh", "a_ice", "u_2", "v_2"]
EMA_CH = ["salt_2", "salt_35", "salt_75", "salt_125", "salt_255", "salt_450", "salt_975",
          "salt_1600", "salt_2035"]
KEY = re.compile(r"^LossPhysical\.([A-Z]+)\.mse\.([a-z_0-9]+?)\.(\d+)$")
EMA = re.compile(r"^LossPhysical\.([A-Z]+)\.mse_ema_weight\.([a-z_0-9]+?)\.\d+$")


def load(run):
    lines = [json.loads(x) for x in open(ROOT / f"results/{run}/{run}_train_metrics.json")]
    ckpts = sorted((ROOT / f"models/{run}").glob(f"{run}_chkpt*.chkpt"))
    ck_times = [(int(p.stem[-5:]), p.stat().st_mtime) for p in ckpts]
    blocks, cur = [], []
    for d in lines:
        if d["stage"] == "val":
            t = d["weathergen.timestamp"] / 1000.0
            # checkpoint written closest to this validation (same end of mini-epoch)
            label = min(ck_times, key=lambda c: abs(c[1] - t))
            label = f"ep->chkpt{label[0]:02d}" if abs(label[1] - t) < 900 else "ep (no chkpt)"
            blocks.append((label, cur))
            cur = []
        elif d["stage"] == "train":
            cur.append(d)
    if cur:
        blocks.append(("current (in progress)", cur))
    return blocks


def per_key(lines):
    """{(stream, ch): {s: [values]}} with s = k - 1 (spin-up days before step k)."""
    acc = defaultdict(lambda: defaultdict(list))
    for d in lines:
        for k, v in d.items():
            m = KEY.match(k)
            if m and v is not None and np.isfinite(v):
                acc[(m.group(1), m.group(2))][int(m.group(3)) - 1].append(v)
    return acc


def binned(acc, stream=None, ch=None, bins=BINS):
    """Mean over (line, key) entries in each spin-up bin; stream mean = mean over channels."""
    keys = [(s, c) for (s, c) in acc if (stream is None or s == stream) and (ch is None or c == ch)]
    out = []
    for a, b in bins:
        vals = []
        if ch is not None:
            for key in keys:
                for s, v in acc[key].items():
                    if a <= s < b:
                        vals += v
            out.append((float(np.mean(vals)) if vals else np.nan, len(vals)))
        else:  # channel-averaged: mean over channels of the per-channel bin means
            cm = []
            n = 0
            for key in keys:
                v = [x for s, xs in acc[key].items() if a <= s < b for x in xs]
                if v:
                    cm.append(np.mean(v))
                    n = max(n, len(v))
            out.append((float(np.mean(cm)) if cm else np.nan, n))
    return out


def ema_weights(lines):
    acc = defaultdict(list)
    for d in lines:
        for k, v in d.items():
            m = EMA.match(k)
            if m and v is not None:
                acc[m.group(2)].append(v)
        if "LossPhysical.SALT.mse_ema_weight.avg" in d:
            acc["SALT.avg"].append(d["LossPhysical.SALT.mse_ema_weight.avg"])
    return {k: (float(np.mean(v)), float(np.min(v)), float(np.max(v))) for k, v in acc.items()}


def fmt(x):
    return "-" if not np.isfinite(x) else f"{x:.4g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["fvars003", "fvars004"])
    p.add_argument("--out", required=True)
    p.add_argument("--fig", default=None)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, data = [], {}
    bl = " | ".join(f"{x}-{min(y, 180)}" for x, y in BINS)
    for run in a.runs:
        blocks = load(run)
        all_lines = [d for _, ls in blocks for d in ls]
        prefixes = [d.get("forecast_prefix_steps") for d in all_lines
                    if d.get("forecast_prefix_steps") is not None]
        md.append(f"## {run}\n")
        md.append(f"{len(all_lines)} train metric lines (20 training steps each), "
                  f"{len(blocks)} blocks; mean logged spin-up per line "
                  f"{np.mean(prefixes):.1f} d (range of line means {np.min(prefixes):.1f}-"
                  f"{np.max(prefixes):.1f}).\n")
        rdata = {"blocks": {}}
        for label, ls in blocks + [("all", all_lines)]:
            acc = per_key(ls)
            r = {"n_lines": len(ls), "streams": {}, "channels": {}}
            for s in STREAMS:
                r["streams"][s] = binned(acc, stream=s)
            for c in CHANNELS:
                st = [s for (s, cc) in acc if cc == c]
                if st:
                    r["channels"][c] = binned(acc, stream=st[0], ch=c)
            r["fine_salt75"] = binned(acc, stream="SALT", ch="salt_75", bins=FINE)
            r["fine_SALT"] = binned(acc, stream="SALT", bins=FINE)
            r["fine_TEMP"] = binned(acc, stream="TEMP", bins=FINE)
            r["fine_temp2"] = binned(acc, stream="TEMP", ch="temp_2", bins=FINE)
            r["fine_ssh"] = binned(acc, stream="SURFACE", ch="ssh", bins=FINE)
            r["ema"] = ema_weights(ls)
            rdata["blocks"][label] = r
        data[run] = rdata

        md.append(f"### {run}: stream-mean MSE (normalised units) by spin-up bin (days)\n")
        md.append(f"| block | lines | stream | {bl} | ratio last/first populated bin |")
        md.append("|---|---|---|" + "---|" * len(BINS) + "---|")
        for label, r in rdata["blocks"].items():
            for s in STREAMS:
                v = [x for x, _ in r["streams"][s]]
                pop = [x for x in v if np.isfinite(x)]
                md.append(f"| {label} | {r['n_lines']} | {s} | " + " | ".join(fmt(x) for x in v)
                          + f" | {pop[-1] / pop[0]:.2f} |")
        md.append("")
        md.append(f"### {run}: per-channel MSE by spin-up bin, all blocks pooled "
                  "(n = (line, key) entries)\n")
        md.append(f"| channel | {bl} | n per bin |")
        md.append("|---|" + "---|" * len(BINS) + "---|")
        r = rdata["blocks"]["all"]
        for c, v in r["channels"].items():
            md.append(f"| {c} | " + " | ".join(fmt(x) for x, _ in v) + " | "
                      + "/".join(str(n) for _, n in v) + " |")
        md.append("")
        md.append(f"### {run}: salt_75 MSE by spin-up bin, per block\n")
        md.append(f"| block | {bl} |")
        md.append("|---|" + "---|" * len(BINS))
        for label, r in rdata["blocks"].items():
            md.append(f"| {label} | " + " | ".join(fmt(x) for x, _ in r["channels"]["salt_75"])
                      + " |")
        md.append("")
        md.append(f"### {run}: 10-day bins, all blocks pooled\n")
        md.append("| spin-up (d) | salt_75 | SALT mean | temp_2 | TEMP mean | ssh | n (salt_75) |")
        md.append("|---|---|---|---|---|---|---|")
        r = rdata["blocks"]["all"]
        for i, (lo, hi) in enumerate(FINE):
            row = [r["fine_salt75"][i], r["fine_SALT"][i], r["fine_temp2"][i], r["fine_TEMP"][i],
                   r["fine_ssh"][i]]
            if not np.isfinite(row[0][0]):
                continue
            md.append(f"| {lo}-{hi} | " + " | ".join(fmt(x) for x, _ in row)
                      + f" | {row[0][1]} |")
        md.append("")
        md.append(f"### {run}: dynamic-loss (EMA) weights of SALT channels, mean (min-max) "
                  "over the block's lines\n")
        md.append("| block | " + " | ".join(EMA_CH) + " | SALT avg |")
        md.append("|---|" + "---|" * (len(EMA_CH) + 1))
        for label, r in rdata["blocks"].items():
            e = r["ema"]
            cells = [f"{e[c][0]:.3f} ({e[c][1]:.2f}-{e[c][2]:.2f})" if c in e else "-"
                     for c in EMA_CH + ["SALT.avg"]]
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")
    (out / "tables.md").write_text("\n".join(md) + "\n")
    (out / "spinup_loss.json").write_text(json.dumps(data, indent=1, default=float))
    print("\n".join(md))

    if a.fig:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = Path(a.fig)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        for run, ls in zip(a.runs, ["-", "--"], strict=False):
            blocks = data[run]["blocks"]
            for j, (q, name) in enumerate([("fine_salt75", "salt_75"), ("fine_temp2", "temp_2"),
                                           ("fine_ssh", "ssh")]):
                ax = axs[j]
                labels = [lb for lb in blocks if lb != "all"]
                cmap = plt.get_cmap("viridis", max(len(labels), 2))
                for i, lb in enumerate(labels):
                    y = [x for x, _ in blocks[lb][q]]
                    x = [lo + 5 for lo, _ in FINE]
                    ax.plot(x, y, ls, color=cmap(i), lw=1,
                            label=f"{run} {lb}" if j == 0 else None)
                ax.set_title(f"{name}: training MSE vs spin-up (normalised)")
                ax.set_xlabel("spin-up before the supervised step (days)")
                ax.axvline(70, color="grey", lw=0.6, ls=":")
                ax.set_yscale("log")
        axs[0].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "d6_spinup_loss.png", dpi=130)


if __name__ == "__main__":
    main()
