"""Branch a new rollout from a saved latent state of a keep-states run.

sbatch_rollout_keep_states.sh copies every rollout_state.pt of a run to
results/RUN/states/state_<ns>.pt. This picks the copy with a given done_steps, rewrites the
inference settings stored with it (forecast.forcing_scale / forcing_split / fe_noise) to the
ones the branch will run with, so the trainer's resume check accepts it, and writes it as
results/NEW_RUN/rollout_state.pt. Then run sbatch_instab_rollout.sh with NEW_RUN, the same
start date and the matching --options (WG_EXTRA_OPTS); it continues from done_steps.

    python scripts/instab/branch_from_state.py fvnnre 3420 fvtr_fs08 --forcing-scale 0.8
    python scripts/instab/branch_from_state.py fvnnre 3420 fvtr_n0 \
        --fe-noise '{"mode": "train_like", "std": 1e-4, "dropout": false, "seed": 0}'
    python scripts/instab/branch_from_state.py --list fvnnre
"""

import argparse
import json
from pathlib import Path

import torch

from weathergen.model.fe_noise import FeNoise
from weathergen.model.forcing_scale import ForcingScale
from weathergen.model.forcing_split import ForcingSplit

RESULTS = Path(__file__).resolve().parents[2] / "results"


def states(run: str) -> dict[int, Path]:
    out = {}
    for p in sorted((RESULTS / run / "states").glob("state_*.pt")):
        s = torch.load(p, map_location="cpu", weights_only=False)
        out[int(s["done_steps"])] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("done_steps", nargs="?", type=int)
    ap.add_argument("new_run", nargs="?")
    ap.add_argument("--list", metavar="RUN")
    ap.add_argument("--forcing-scale", type=float, default=None)
    ap.add_argument("--fe-noise", type=json.loads, default=None)
    ap.add_argument("--forcing-split", type=json.loads, default=None,
                    help='e.g. {"const_path": "C.pt", "alpha_const": 1.0, "alpha_anom": 0.0}')
    a = ap.parse_args()
    if a.list:
        for k, p in states(a.list).items():
            print(k, p.name)
        return
    src = states(a.run).get(a.done_steps)
    if src is None:
        raise SystemExit(f"no saved state of {a.run} at done_steps={a.done_steps}")
    s = torch.load(src, map_location="cpu", weights_only=False)
    fs = ForcingScale.from_config(a.forcing_scale)
    fn = FeNoise.from_config(a.fe_noise)
    s["forcing_scale"] = fs.as_dict() if fs is not None else None
    s["fe_noise"] = fn.as_dict() if fn is not None else None
    fsp = ForcingSplit.from_config(a.forcing_split)
    s["forcing_split"] = fsp.as_dict() if fsp is not None else None
    dst = RESULTS / a.new_run / "rollout_state.pt"
    if dst.parent.exists() and any(dst.parent.iterdir()):
        raise SystemExit(f"{dst.parent} is not empty")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(s, dst)
    print(f"{src} (done_steps={s['done_steps']}) -> {dst}; forcing_scale={s['forcing_scale']}, "
          f"fe_noise={s['fe_noise']}, forcing_split={s['forcing_split']}")


if __name__ == "__main__":
    main()
