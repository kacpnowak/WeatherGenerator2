#!/usr/bin/env python
"""
Decomposition of the additive forcing injection (fable report, section 2 D1, step 1).

    python forcing_decomp.py --out DIR --tag TAG [--days 363] [--roll-days 150]
        [--save-const FILE] -- <inference args>

<inference args> as for latent_error.py (one sample, lazy rollout from the start date,
forecast.chunk_size < forecast.num_steps, num_steps >= --days + 1).

The per-cell injection of rollout step k is f_k = gate * emb(forcing_k), read exactly as in a
rollout (forcing index k = 1..days, the field consumed before the k-th forecast-engine step)
and computed by the model's own forcing module under bf16 autocast, applied to a zero latent
(so the module returns f_k itself, in the dtype the rollout adds). Reported:

- C = time mean of f_k; |C| / RMS_k |f_k|; cos(C, f_k);
- a_k = f_k - C; |a_k| relative to |f_k| and to the latent norm |z| (the encoded initial
  latent and a free default rollout of --roll-days steps, z_n = FE(z_(n-1) + f_n));
- the embedding MLP's last-layer bias b2: |gate * b2| per cell, its broadcast norm relative
  to |C| and its projection share of |C|^2; the cell-uniform part of C (mean over cells);
- per-cell |C_i| and the per-cell RMS of |a_k,i| for land / ocean (land cells as
  land_cells.land_cell_mask);
- |sum_k f_k| and |sum_k a_k| growth (linear vs sqrt(n)).

-> OUT/forcing_decomp_TAG.json, OUT/forcing_decomp_TAG.npz (per-cell maps, per-day series);
   --save-const: C as a float32 tensor (num_cells, dim) with metadata, for
   forecast.forcing_split.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instab_probe as ip  # noqa: E402

import weathergen.common.config as config  # noqa: E402
import weathergen.utils.cli as cli  # noqa: E402
from weathergen.common.logger import init_loggers  # noqa: E402
from weathergen.model.land_cells import land_cell_mask  # noqa: E402
from weathergen.train.trainer import Trainer  # noqa: E402

logger = logging.getLogger("forcing_decomp")


def encode(model, mp, src):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tokens, _ = model.encoder(mp, src)
        tokens = tokens.reshape((len(src), src.get_num_source_steps(), *tokens.shape[1:])).sum(1)
    return tokens


def cls_norms(x, masks_t):
    """Frobenius norm over all / land / ocean cells of a (cells, dim) tensor."""
    sq = x.double().pow(2).sum(-1)
    return {c: float(sq[m].sum().sqrt()) for c, m in masks_t.items()}


class DecompTrainer(Trainer):
    args = None

    def validate(self, mini_epoch, mode_cfg, batch_size):
        self.model.eval()
        args = self.args
        ds, model, mp, dev = self.dataset_val, self.model, self.model_params, self.device
        fm = model.forcing_module
        if fm is None or fm.mode != "additive":
            mode = getattr(fm, "mode", None)
            raise ValueError(f"needs an additive forcing injection, got {mode}")
        batch = next(iter(self.data_loader_validation))
        batch.to_device(dev, include_targets=False)
        D, R = args.days, args.roll_days
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        level = self.cf.healpix_level
        land = land_cell_mask(self.cf.streams, level)
        n_cells = len(land)
        masks = {"all": np.ones(n_cells, bool), "land": land, "ocean": ~land}
        masks_t = {c: torch.from_numpy(m).to(dev) for c, m in masks.items()}
        na = model.num_aux_tokens
        res = {"tag": args.tag, "days": D, "roll_days": R, "num_cells": n_cells,
               "num_land": int(land.sum()), "num_aux": int(na),
               "gate": float(fm.gate.detach().float().cpu())}
        logger.info(f"{args.tag}: gate {res['gate']:+.5f}, land cells {res['num_land']}")
        t0 = time.time()
        with torch.no_grad():
            ip.materialize_forcing(ds, batch, (1, max(D, R) + 1), dev)
            res["forcing_seconds"] = time.time() - t0
            logger.info(f"forcing for {max(D, R)} steps read in {res['forcing_seconds']:.0f} s")
            src = batch.get_source_samples()
            z0 = encode(model, mp, src)
            res["z_dtype"] = str(z0.dtype)
            zeros = torch.zeros_like(z0)

            def f_of(k):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out, _ = fm(zeros, src.conditions[k], src.forcing[k], na)
                if out.dtype != torch.float32:
                    logger.warning(f"injection dtype {out.dtype}")
                return out[0, na:].float()  # (cells, dim): 0 + gate * emb exactly

            # pass 1: time mean C and per-day norms
            s = torch.zeros(n_cells, z0.shape[-1], dtype=torch.float64, device=dev)
            f_norm = {c: [] for c in masks}
            f_cellmean = []
            for k in range(1, D + 1):
                f = f_of(k)
                s += f.double()
                for c, v in cls_norms(f, masks_t).items():
                    f_norm[c].append(v)
                f_cellmean.append(f.double().mean(0).cpu())
            C = (s / D).float()
            del s
            # pass 2: anomalies
            a_norm = {c: [] for c in masks}
            cosCf, cum_f, cum_a = [], [], []
            a_cell_sq = torch.zeros(n_cells, dtype=torch.float64, device=dev)
            Sf = torch.zeros_like(C, dtype=torch.float64)
            Sa = torch.zeros_like(C, dtype=torch.float64)
            for k in range(1, D + 1):
                f = f_of(k)
                a = f - C
                for c, v in cls_norms(a, masks_t).items():
                    a_norm[c].append(v)
                a_cell_sq += a.double().pow(2).sum(-1)
                cosCf.append(float((f.double() * C.double()).sum()
                                   / (f.double().norm() * C.double().norm())))
                Sf += f.double()
                Sa += a.double()
                cum_f.append(float(Sf.norm()))
                cum_a.append(float(Sa.norm()))
            del Sf, Sa
            C_norm = cls_norms(C, masks_t)
            rms_f = {c: float(np.sqrt(np.mean(np.square(v)))) for c, v in f_norm.items()}
            rms_a = {c: float(np.sqrt(np.mean(np.square(v)))) for c, v in a_norm.items()}
            # bias of the embedding MLP's last layer
            last = fm.embed.mlp[-1]
            b2 = last.bias.detach().float()
            gb2 = fm.gate.detach().float() * b2  # (dim,)
            gb2_cells = gb2.unsqueeze(0).expand(n_cells, -1)
            Cbar = C.double().mean(0)  # cell-uniform part of C
            res.update({
                "C_norm": C_norm, "rms_f_norm": rms_f, "rms_a_norm": rms_a,
                "C_over_rms_f": {c: C_norm[c] / rms_f[c] for c in masks},
                "C2_share_of_mean_f2": {c: C_norm[c] ** 2 / rms_f[c] ** 2 for c in masks},
                "a_over_f": {c: rms_a[c] / rms_f[c] for c in masks},
                "cos_C_f": {"min": min(cosCf), "mean": float(np.mean(cosCf)), "max": max(cosCf)},
                "b2_norm": float(b2.norm()), "W2_fro": float(last.weight.detach().float().norm()),
                "gate_b2_norm_per_cell": float(gb2.norm()),
                "gate_b2_broadcast_norm": float(gb2_cells.double().norm()),
                "gate_b2_broadcast_over_C": float(gb2_cells.double().norm()) / C_norm["all"],
                "gate_b2_proj_share_of_C2": float((C.double() * gb2_cells.double()).sum()
                                                  / C.double().pow(2).sum()),
                "C_minus_gate_b2_over_C": float((C.double() - gb2_cells.double()).norm())
                / C_norm["all"],
                "C_cellmean_norm_per_cell": float(Cbar.norm()),
                "C_cellmean_share_of_C2": float(Cbar.pow(2).sum() * n_cells
                                                / C.double().pow(2).sum()),
                "cos_Ccellmean_gate_b2": float((Cbar * gb2.double()).sum()
                                               / (Cbar.norm() * gb2.double().norm())),
                "cum_f_over_n": [cum_f[i] / (i + 1) for i in range(D)][:: max(D // 73, 1)],
                "cum_f": cum_f, "cum_a": cum_a, "cos_C_f_series": cosCf,
                "f_norm": f_norm, "a_norm": a_norm,
            })
            # per-cell maps
            C_cell = C.double().norm(dim=-1)
            a_cell = (a_cell_sq / D).sqrt()
            for nm, v in (("C_cell", C_cell), ("a_cell_rms", a_cell)):
                res[nm] = {}
                for c in ("land", "ocean"):
                    x = v[masks_t[c]].cpu().numpy()
                    res[nm][c] = {"mean": float(x.mean()), "median": float(np.median(x)),
                                  "p10": float(np.percentile(x, 10)),
                                  "p90": float(np.percentile(x, 90))}
            ratio = (a_cell / C_cell.clamp_min(1e-12)).cpu().numpy()
            res["a_over_C_cell"] = {c: {"median": float(np.median(ratio[masks[c]])),
                                        "mean": float(ratio[masks[c]].mean())}
                                    for c in ("land", "ocean")}
            # latent norms: encoded IC and a free default rollout
            z = z0
            lat = {"n": [], "z_all": [], "z_land": [], "z_ocean": [], "token_rms_all": [],
                   "token_rms_ocean": [], "f_over_z": [], "a_over_z": [], "C_over_z": [],
                   "train_noise_per_entry_std": []}

            def rec(n, z, k):
                zc = z[0, na:].float()
                nz = cls_norms(zc, masks_t)
                lat["n"].append(n)
                lat["z_all"].append(nz["all"])
                lat["z_land"].append(nz["land"])
                lat["z_ocean"].append(nz["ocean"])
                tr = zc.double().pow(2).mean(-1).sqrt()
                lat["token_rms_all"].append(float(tr.mean()))
                lat["token_rms_ocean"].append(float(tr[masks_t["ocean"]].mean()))
                # training-mode impute noise: per-entry std = |z|_F (all tokens) * 1e-4
                lat["train_noise_per_entry_std"].append(float(z.float().norm()) * 1e-4)
                if k is not None:
                    lat["f_over_z"].append(f_norm["all"][k - 1] / nz["all"])
                    lat["a_over_z"].append(a_norm["all"][k - 1] / nz["all"])
                    lat["C_over_z"].append(C_norm["all"] / nz["all"])

            rec(0, z, None)
            cond_coords = mp.rope_coords
            for n in range(1, R + 1):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    zin, cond = fm(z, src.conditions[n], src.forcing[n], na)
                    z = model.forecast_engine(zin, cond, coords=cond_coords)
                rec(n, z, n)
            res["latent"] = lat
            res["seconds"] = time.time() - t0
        np.savez_compressed(out_dir / f"forcing_decomp_{args.tag}.npz",
                            C_cell=C_cell.cpu().numpy(), a_cell_rms=a_cell.cpu().numpy(),
                            land=land, cell_lat=ip.healpix_cell_latlon(level)[0],
                            cell_lon=ip.healpix_cell_latlon(level)[1],
                            f_cellmean=torch.stack(f_cellmean).numpy())
        (out_dir / f"forcing_decomp_{args.tag}.json").write_text(json.dumps(res, indent=1))
        if args.save_const:
            torch.save({"const": C.cpu(), "tag": args.tag, "days": D,
                        "first_forcing_index": 1, "gate": res["gate"],
                        "start_date": str(self.test_cfg.get("start_date"))}, args.save_const)
            logger.info(f"C saved to {args.save_const}")
        logger.info(f"forcing_decomp {args.tag}: written to {out_dir} ({res['seconds']:.0f} s)")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--days", type=int, default=363)
    p.add_argument("--roll-days", type=int, default=150)
    p.add_argument("--save-const", default=None)
    if "--" not in sys.argv:
        raise SystemExit("separate the inference arguments with --")
    split = sys.argv.index("--")
    args = p.parse_args(sys.argv[1:split])
    inf = cli.get_inference_parser().parse_args(sys.argv[split + 1:])
    cli_overwrite = config.from_cli_arglist(inf.options)
    cf = config.load_merge_configs(inf.private_config, inf.from_run_id, inf.mini_epoch,
                                   inf.base_config, *inf.config, {}, cli_overwrite)
    cf = config.set_run_id(cf, inf.run_id, inf.reuse_run_id)
    devices = Trainer.init_torch()
    cf = Trainer.init_ddp(cf)
    init_loggers(log_path=config.get_path_logs(cf))
    logging.getLogger("forcing_decomp").setLevel(logging.INFO)
    cf.general.run_history += [(inf.from_run_id, cf.general.istep)]
    tr = DecompTrainer(cf.train_logging)
    tr.args = args
    tr.inference(cf, devices, inf.from_run_id, inf.mini_epoch)


if __name__ == "__main__":
    main()
