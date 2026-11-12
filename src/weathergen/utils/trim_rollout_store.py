"""
Trim a long rollout's zip store back to the steps its rollout_state.pt records.

A rollout writes each chunk to the store and then saves rollout_state.pt. A job that dies
between the two leaves the store one chunk ahead of the state, and the next job stops when it
tries to write that chunk again ("Group already exists"). A zip cannot drop entries, so the
store is copied without the leads after the saved step and then replaces the original.

    python -m weathergen.utils.trim_rollout_store <results/<run_id>>
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

import torch

#: <sample>/<stream>/<lead>/... entries of a rollout store
_LEAD_ENTRY = re.compile(r"^\d+/[^/]+/(\d+)/")


def _lead(name: str) -> int | None:
    match = _LEAD_ENTRY.match(name)
    return None if match is None else int(match.group(1))


def trim_rollout_store(path: Path, done_steps: int) -> int:
    """
    Remove the entries of leads after `done_steps` from the zip store at `path`; returns how many
    leads were removed. A store without such leads is left untouched.
    """
    with zipfile.ZipFile(path) as source:
        entries = source.infolist()
        stale = {
            lead for e in entries if (lead := _lead(e.filename)) is not None and lead > done_steps
        }
        if not stale:
            return 0
        path_tmp = path.with_name(path.name + ".tmp")
        with zipfile.ZipFile(path_tmp, "w", allowZip64=True) as target:
            for entry in entries:
                if _lead(entry.filename) in stale:
                    continue
                # streamed in blocks with the entry's own settings (rollout stores are uncompressed)
                with source.open(entry) as src, target.open(entry, "w", force_zip64=True) as dst:
                    while block := src.read(1 << 24):
                        dst.write(block)
    path_tmp.replace(path)
    return len(stale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("results_dir", type=Path, help="results/<run_id> of the rollout")
    parser.add_argument("--mini-epoch", type=int, default=0)
    args = parser.parse_args()

    state = torch.load(
        args.results_dir / "rollout_state.pt", map_location="cpu", weights_only=False
    )
    path = args.results_dir / f"validation_chkpt{args.mini_epoch:05d}_rank0000.zip"
    removed = trim_rollout_store(path, state["done_steps"])
    sys.stdout.write(f"{path}: removed {removed} leads after step {state['done_steps']}\n")


if __name__ == "__main__":
    main()
