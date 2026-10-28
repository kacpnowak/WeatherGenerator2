#!/usr/bin/env python
"""
Latent-error decomposition of free rollouts (landcells report, section 3 step 1).

    python latent_error.py --out DIR --tag TAG --days D [--modes none freeze]
        [--gain-days 40 80] -- <inference args>

<inference args> as for instab_probe.py (one sample, lazy rollout: forecast.chunk_size <
forecast.num_steps, num_steps >= D + 2). One process per model and start date:

1. Encoded truth. z_true_n = E(x_n) for n = 0..D: the source batch of the sample whose base
   index is idx0 + n (the window of target step n of the rollout's sample), encoded exactly
   as the rollout's initial latent (encoder under bf16 autocast, summed over source steps,
   no forcing: the forcing of day k is injected before the k-th forecast-engine step).
   Check: z_true_0 equals the rollout's z_0 (encoding of the loader's batch).
   Noise floor: s_n = S_(n+1)(z_true_n), the one-day step from the encoded truth, and
   |z_true_(n+1) - s_n| per cell class (default step, and the freeze step with the land
   tokens of z_true_n kept), and the lead-2 minus lead-1 latent S(S(z_true_(n-1))) - S(z_true_n).
   The encoded truth is off the forecast-latent manifold (|z_true - S(z_true)| ~ 55 % of |z|
   for fvars003), so a second reference is used: the lead-1 latent r_n = S_n(z_true_(n-1))
   (freeze: land tokens replaced by those of z_0 before the step), error g_n = z_n - r_n
   (g_1 = 0), decomposed the same way; gain probes also along g_N.
2. Free rollouts z_n (bf16 autocast, the model's step: z_n = FE(z_(n-1) + f_n), f_n = the
   additive forcing injection of step n, as applied; freeze: land tokens reset to z_0 after
   each step), for each land-cell mode. Checked against Model.forward for 5 steps.
   e_n = z_n - z_true_n, R_n = sum_k (dz_k - f_k) = (z_n - z_0) - F_n, F_n = sum_k f_k,
   dz_n = z_n - z_(n-1), D_n = z_n - z_0. Per cell class (land, land-adjacent ocean, other
   ocean, ocean, all): |e|^2 shares, cosines of e with R, F, dz, D (fraction of |e|^2 on the
   direction = cos^2) and the fraction of |e|^2 in span{R, F}.
3. Gain probes at --gain-days N on each trajectory: S(z) = step N -> N+1 (with the mode's
   land treatment), J_eps v = (S(z + e v) - S(z - e v)) / 2e, e = eps |z|, |v| = 1;
   q = <v, J v>, gain = |J v|, for v = next increment z_(N+1) - z_N, increment dz_N, e_N,
   e_N on ocean cells only, R_N, F_N, a random direction; fp32 (SDPA, TF32 off) and bf16.

-> OUT/latenterr_TAG.json, OUT/latenterr_TAG.npz (per-day per-cell maps).
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instab_probe as ip  # noqa: E402  (patches flash attention for fp32 inputs)

import weathergen.common.config as config  # noqa: E402
import weathergen.utils.cli as cli  # noqa: E402
from weathergen.common.logger import init_loggers  # noqa: E402
from weathergen.model.land_cells import LandCells, land_cell_mask  # noqa: E402
from weathergen.train.trainer import Trainer  # noqa: E402

logger = logging.getLogger("latent_error")

CLASSES = ("all", "land", "adj", "other", "ocean")
# per-cell sums kept for every day: squares and cross products of e, R, F, dz, D
PAIRS = ("ee", "RR", "FF", "II", "DD", "eR", "eF", "eI", "eD", "RF", "gg", "gR", "gF", "gI",
         "gD", "eg")
KEYS = ("ee_share", "cos_eR", "cos_eF", "cos_eI", "cos_eD", "frac_span_RF", "cos_RF", "cos_eg")


def class_masks(land_mask: np.ndarray, level: int) -> dict:
    """Cell classes: land, land-adjacent ocean (an ocean cell with a land cell among its 8
    nested-HEALPix neighbours), other ocean, ocean, all."""
    from astropy_healpix import HEALPix

    n = 12 * 4**level
    nb = HEALPix(nside=2**level, order="nested").neighbours(np.arange(n))  # (8, n), -1 none
    land = land_mask.astype(bool)
    has_land_nb = np.zeros(n, bool)
    for row in nb:
        ok = row >= 0
        has_land_nb[ok] |= land[row[ok]]
    adj = ~land & has_land_nb
    return {"all": np.ones(n, bool), "land": land, "adj": adj, "other": ~land & ~adj,
            "ocean": ~land}


class Stepper:
    """The rollout's one-day step z_(k-1) -> z_k, with the land-cell treatment."""

    def __init__(self, model, mp, src, land_cells=None, land_init=None):
        self.m, self.mp, self.src = model, mp, src
        self.lc, self.land_init = land_cells, land_init
        self.na = model.num_aux_tokens

    def forcing(self, z, k):
        """Forcing injection f_k (as applied) at z, and the forced tokens."""
        m = self.m
        zin, cond = m.forcing_module(z, self.src.conditions[k], self.src.forcing[k], self.na)
        if self.lc is not None:
            zin = self.lc.after_forcing(z, zin, self.na)
        return zin, cond

    def __call__(self, z, k, precision="bf16", return_forcing=False):
        m = self.m
        tf32 = torch.backends.cuda.matmul.allow_tf32
        if precision == "fp32":
            ip.set_fe_attention_dtype(m, torch.float32)
            torch.backends.cuda.matmul.allow_tf32 = False
            ctx = torch.autocast(device_type="cuda", enabled=False)
            z = z.float()
        else:
            ip.set_fe_attention_dtype(m, torch.bfloat16)
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        try:
            with ctx:
                zin, cond = self.forcing(z, k)
                out = m.forecast_engine(zin, cond, coords=self.mp.rope_coords)
                if self.lc is not None:
                    out = self.lc.after_step(out, self.land_init, self.na)
        finally:
            ip.set_fe_attention_dtype(m, torch.bfloat16)
            torch.backends.cuda.matmul.allow_tf32 = tf32
        if return_forcing:
            return out, zin.float() - z.float()
        return out


def encode(model, mp, src):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tokens, _ = model.encoder(mp, src)
        tokens = tokens.reshape((len(src), src.get_num_source_steps(), *tokens.shape[1:])).sum(1)
    return tokens


def encode_truth(tr, idx0, days):
    """z_true_n, n = 0..days (fp32, on the device), and the source window start times."""
    ds, model, mp, dev = tr.dataset_val, tr.model, tr.model_params, tr.device
    ctx0 = ds._materialize_ctx
    num_fc = ctx0["num_forecast_steps"]
    zt, times = [], []
    t0 = time.time()
    try:
        for n in range(days + 1):
            idx = idx0 + (ds.time_step * n) // ds.step_timedelta
            b = ds._build_batch_skeleton(idx, num_fc)
            if b.sources_empty() or b.sources_nan():
                raise ValueError(f"empty / NaN source batch at day {n} (idx {idx})")
            b.to_device(dev, include_targets=False)
            src = b.get_source_samples()
            zt.append(encode(model, mp, src).float())
            times.append(str(ds.time_window_handler.window(idx).start))
            del b, src
            if n % 10 == 0:
                logger.info(f"encoded truth day {n} ({times[-1]}), {time.time() - t0:.0f} s")
    finally:
        ds._materialize_ctx = ctx0
    return zt, times


def cell_view(x, na, n_cells):
    return x[:, na:].reshape(n_cells, -1) if x.dim() == 3 else x.reshape(n_cells, -1)


def aggregate(per_cell: dict, masks_t: dict, err: str) -> dict:
    """Class sums of the per-cell sums and the derived shares / cosines of the error `err`
    ("e": encoded-truth reference, "g": lead-1 reference)."""
    out = {}
    ee = err + err
    tot = float(per_cell[ee].sum())
    for c, m in masks_t.items():
        s = {k: float(v[m].sum()) for k, v in per_cell.items()}
        r = {"ee_share": s[ee] / tot if tot > 0 else float("nan")}
        for x, xx in (("R", "RR"), ("F", "FF"), ("I", "II"), ("D", "DD")):
            den = np.sqrt(s[ee] * s[xx])
            r[f"cos_e{x}"] = s[f"{err}{x}"] / den if den > 0 else float("nan")
        # fraction of |err|^2 in span{R, F}
        G = np.array([[s["RR"], s["RF"]], [s["RF"], s["FF"]]])
        b = np.array([s[f"{err}R"], s[f"{err}F"]])
        try:
            r["frac_span_RF"] = (float(b @ np.linalg.solve(G, b)) / s[ee] if s[ee] > 0
                                 else float("nan"))
        except np.linalg.LinAlgError:
            r["frac_span_RF"] = float("nan")
        den = np.sqrt(s["RR"] * s["FF"])
        r["cos_RF"] = s["RF"] / den if den > 0 else float("nan")
        den = np.sqrt(s["ee"] * s["gg"])
        r["cos_eg"] = s["eg"] / den if den > 0 else float("nan")
        r.update({f"sum_{k}": v for k, v in s.items() if k in (ee, "RR", "FF", "II", "DD")})
        out[c] = r
    return out


def gain_probe(stepper, z, k, dirs, ocean_t, na, n_cells):
    """q and |J v| (all cells and ocean cells) along the directions, fp32 and bf16."""
    znorm = float(z.norm())
    res = {}
    with torch.no_grad():
        for prec, epss in (("fp32", (1e-3, 1e-2, 3e-2, 1e-1)), ("bf16", (1e-2, 3e-2, 1e-1))):
            res[prec] = {}
            for name, v in dirs.items():
                rows = {}
                for eps in epss:
                    e = eps * znorm
                    j = (stepper(z + e * v, k, prec).float()
                         - stepper(z - e * v, k, prec).float()) / (2 * e)
                    jc, vc = cell_view(j, na, n_cells), cell_view(v, na, n_cells)
                    vo = vc[ocean_t]
                    vo_n = float(vo.norm())
                    rows[f"{eps:.0e}"] = {
                        "q": float((v * j).sum()), "gain": float(j.norm()),
                        "gain_ocean": float(jc[ocean_t].norm()),
                        # growth of the ocean part of v: <v_o, (Jv)_o> / |v_o|^2
                        "q_ocean": float((vo * jc[ocean_t]).sum()) / vo_n**2 if vo_n > 0
                        else float("nan"),
                    }
                res[prec][name] = rows
                logger.info(f"gain {prec} {name}: "
                            + ", ".join(f"{e}: q {r['q']:.3f} g {r['gain']:.3f}"
                                        for e, r in rows.items()))
    return res


def run_mode(tr, src, mode, zt, masks, gain_days, days, out_npz, tag):
    model, mp, dev = tr.model, tr.model_params, tr.device
    na = model.num_aux_tokens
    n_cells = len(masks["all"])
    masks_t = {c: torch.from_numpy(m).to(dev) for c, m in masks.items()}
    lc = None if mode == "none" else LandCells(mode=mode, land_mask=torch.from_numpy(
        masks["land"]))
    z0 = zt[0].clone()
    land_init = lc.initial_tokens(z0, na) if lc is not None else None
    st = Stepper(model, mp, src, lc, land_init)

    # check the step against Model.forward (5 steps, one chunk)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(mp, src, 5, **({} if lc is None else {"land_cells": lc}))
    ref = [out.latent[i]["latent_state"].z_pre_norm.float() for i in range(5)]
    del out
    z = z0
    check = []
    with torch.no_grad():
        for k in range(1, 6):
            z = st(z, k).float()
            check.append(float((z - ref[k - 1]).norm() / ref[k - 1].norm()))
    logger.info(f"{mode}: own step vs Model.forward, rel diff steps 1-5: {check}")
    del ref

    # two error references: e = z_n - z_true_n (encoded truth); g = z_n - r_n with the lead-1
    # reference r_n = S_n(z_true_(n-1)) (the mode's step; freeze: land tokens of z_true_(n-1)
    # replaced by those of z_0, as in the rollout), which lies on the forecast-latent manifold
    T = {"n": []}
    for c in CLASSES:
        for pre in ("", "g_"):
            for key in KEYS:
                T[f"{c}.{pre}{key}"] = []
        for key in ("e_rel", "g_rel", "R_rel", "F_rel", "D_rel", "I_rel"):
            T[f"{c}.{key}"] = []
    T["z_norm"], T["zt_norm"], T["e_norm"], T["g_norm"], T["f_rel"] = [], [], [], [], []
    maps = {k: np.zeros((days, n_cells), np.float32)
            for k in ("ee", "eR", "RR", "eF", "FF", "gg", "gR", "gF")}
    keep = {}
    z = z0.clone()
    Fc = torch.zeros_like(z0)
    t0 = time.time()
    with torch.no_grad():
        for n in range(1, days + 1):
            zn, f = st(z, n, return_forcing=True)
            zn = zn.float()
            dz = zn - z
            Fc += f
            D = zn - z0
            R = D - Fc
            e = zn - zt[n]
            zp = zt[n - 1]
            if lc is not None and lc.needs_init:
                zp = zp.index_copy(1, lc.token_index(zp, na), land_init.to(zp.dtype))
            g_err = zn - st(zp, n).float()
            if n in gain_days or n - 1 in gain_days:
                keep[n] = {"z": zn.clone(), "e": e.clone(), "g": g_err.clone(), "R": R.clone(),
                           "F": Fc.clone(), "dz": dz.clone()}
            V = {k: cell_view(x, na, n_cells) for k, x in
                 (("e", e), ("g", g_err), ("R", R), ("F", Fc), ("I", dz), ("D", D))}
            pc = {p: (V[p[0]] * V[p[1]]).sum(-1).double() for p in PAIRS}
            agg = {"": aggregate(pc, masks_t, "e"), "g_": aggregate(pc, masks_t, "g")}
            ztc = cell_view(zt[n], na, n_cells).pow(2).sum(-1).double()
            T["n"].append(n)
            for c in CLASSES:
                m = masks_t[c]
                ztn = float(ztc[m].sum().sqrt())
                for pre in ("", "g_"):
                    for key in KEYS:
                        T[f"{c}.{pre}{key}"].append(agg[pre][c][key])
                for x, xx, pre in (("e", "ee", ""), ("g", "gg", "g_"), ("R", "RR", ""),
                                   ("F", "FF", ""), ("D", "DD", ""), ("I", "II", "")):
                    T[f"{c}.{x}_rel"].append(float(np.sqrt(agg[pre][c][f"sum_{xx}"])) / ztn)
            T["z_norm"].append(float(zn.norm()))
            T["zt_norm"].append(float(zt[n].norm()))
            T["e_norm"].append(float(e.norm()))
            T["g_norm"].append(float(g_err.norm()))
            T["f_rel"].append(float(f.norm() / z.norm()))
            for k in maps:
                maps[k][n - 1] = pc[k].float().cpu().numpy()
            z = zn
            if n % 10 == 0:
                logger.info(f"{mode} day {n}: |e|/|zt| {T['all.e_rel'][-1]:.3f} "
                            f"|g|/|zt| {T['all.g_rel'][-1]:.3f} "
                            f"cos(e,R) {T['all.cos_eR'][-1]:.3f} cos(g,R) "
                            f"{T['all.g_cos_eR'][-1]:.3f} cos(g,F) {T['all.g_cos_eF'][-1]:.3f} "
                            f"adj share e {T['adj.ee_share'][-1]:.3f} "
                            f"g {T['adj.g_ee_share'][-1]:.3f}")
    res = {"mode": mode, "step_check_rel": check, "rollout_seconds": time.time() - t0,
           "trajectory": T, "gains": {}}
    for k, v in maps.items():
        out_npz[f"{mode}_{k}"] = v

    ocean_t = masks_t["ocean"]
    for N in gain_days:
        if N not in keep or N + 1 not in keep:
            continue
        zN = keep[N]["z"]
        oc_e = keep[N]["e"].clone()
        cell_view(oc_e, na, n_cells)[~ocean_t] = 0
        oc_g = keep[N]["g"].clone()
        cell_view(oc_g, na, n_cells)[~ocean_t] = 0
        g = torch.Generator(device=dev).manual_seed(0)
        dirs = {"next_increment": keep[N + 1]["dz"], "increment": keep[N]["dz"],
                "e": keep[N]["e"], "e_ocean": oc_e, "g": keep[N]["g"], "g_ocean": oc_g,
                "R": keep[N]["R"], "F": keep[N]["F"],
                "random0": torch.randn(zN.shape, generator=g, device=dev)}
        cos = {f"cos_{a}_{b}": float((dirs[a] * dirs[b]).sum()
                                     / (dirs[a].norm() * dirs[b].norm()))
               for a, b in (("e", "next_increment"), ("e", "increment"), ("e", "R"),
                            ("e", "F"), ("g", "R"), ("g", "F"), ("g", "next_increment"),
                            ("g", "e"), ("next_increment", "F"), ("increment", "R"))}
        dirs = {k: v / v.norm() for k, v in dirs.items()}
        t1 = time.time()
        gres = gain_probe(st, zN, N + 1, dirs, ocean_t, na, n_cells)
        # the bf16 step at z_N reproduces z_(N+1)
        with torch.no_grad():
            chk = float((st(zN, N + 1).float() - keep[N + 1]["z"]).norm()
                        / keep[N + 1]["z"].norm())
        res["gains"][N] = {"cosines": cos, "step_check_rel": chk, "z_norm": float(zN.norm()),
                           "seconds": time.time() - t1, **gres}
    del keep
    torch.cuda.empty_cache()
    return res


class LatErrTrainer(Trainer):
    args = None

    def validate(self, mini_epoch, mode_cfg, batch_size):
        self.model.eval()
        args = self.args
        ds, model, mp, dev = self.dataset_val, self.model, self.model_params, self.device
        batch = next(iter(self.data_loader_validation))
        batch.to_device(dev, include_targets=False)
        D = args.days
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        level = self.cf.healpix_level
        land = land_cell_mask(self.cf.streams, level)
        land_probe = ~ip.ocean_cells(level, ip.mesh_path(self.cf))
        if not np.array_equal(land, land_probe):
            raise ValueError("land mask differs from instab_probe's land cells")
        masks = class_masks(land, level)
        counts = {c: int(m.sum()) for c, m in masks.items()}
        logger.info(f"cell classes: {counts}")
        idx0 = int(ds._materialize_ctx["idx"])
        with torch.no_grad():
            ip.materialize_forcing(ds, batch, (1, D + 2), dev)
            src = batch.get_source_samples()
            z0 = encode(model, mp, src).float()
            t0 = time.time()
            zt, times = encode_truth(self, idx0, D)
            enc_s = time.time() - t0
            res = {"tag": args.tag, "days": D, "idx0": idx0, "truth_times": times,
                   "encode_seconds": enc_s, "class_counts": counts,
                   "z_true0_vs_z0_rel": float((zt[0] - z0).norm() / z0.norm()),
                   "z_shape": list(z0.shape), "num_aux": int(model.num_aux_tokens)}
            logger.info(f"z_true_0 vs z_0: {res['z_true0_vs_z0_rel']:.3e}, encode {enc_s:.0f} s")
            del z0
            na, n_cells = model.num_aux_tokens, len(masks["all"])
            masks_t = {c: torch.from_numpy(m).to(dev) for c, m in masks.items()}
            # noise floor: one-day step from the encoded truth
            st_def = Stepper(model, mp, src)
            lc_fz = LandCells(mode="freeze", land_mask=torch.from_numpy(masks["land"]))
            NF = {"n": []}
            for c in CLASSES:
                for key in ("nf_rel", "nf_fz_rel", "nf2_rel", "dtrue_rel", "Dtrue_rel",
                            "nf_share"):
                    NF[f"{c}.{key}"] = []
            nf_map = np.zeros((D, n_cells), np.float32)
            s_prev = None
            for n in range(D):
                s = st_def(zt[n], n + 1).float()
                # lead-2 minus lead-1 latent at day n + 1 (default step)
                r2 = (cell_view(st_def(s_prev, n + 1).float() - s, na, n_cells).pow(2).sum(-1)
                      .double() if s_prev is not None else None)
                s_prev = s
                st_fz = Stepper(model, mp, src, lc_fz, lc_fz.initial_tokens(zt[n], na))
                s_fz = st_fz(zt[n], n + 1).float()
                r = cell_view(zt[n + 1] - s, na, n_cells).pow(2).sum(-1).double()
                r_fz = cell_view(zt[n + 1] - s_fz, na, n_cells).pow(2).sum(-1).double()
                dt = cell_view(zt[n + 1] - zt[n], na, n_cells).pow(2).sum(-1).double()
                Dt = cell_view(zt[n + 1] - zt[0], na, n_cells).pow(2).sum(-1).double()
                ztc = cell_view(zt[n + 1], na, n_cells).pow(2).sum(-1).double()
                NF["n"].append(n + 1)
                for c in CLASSES:
                    m = masks_t[c]
                    ztn = float(ztc[m].sum().sqrt())
                    NF[f"{c}.nf_rel"].append(float(r[m].sum().sqrt()) / ztn)
                    NF[f"{c}.nf_fz_rel"].append(float(r_fz[m].sum().sqrt()) / ztn)
                    NF[f"{c}.nf2_rel"].append(float(r2[m].sum().sqrt()) / ztn if r2 is not None
                                              else float("nan"))
                    NF[f"{c}.dtrue_rel"].append(float(dt[m].sum().sqrt()) / ztn)
                    NF[f"{c}.Dtrue_rel"].append(float(Dt[m].sum().sqrt()) / ztn)
                    NF[f"{c}.nf_share"].append(float(r[m].sum() / r.sum()))
                nf_map[n] = r.float().cpu().numpy()
            res["noise_floor"] = NF
            npz = {"noise_floor_ee": nf_map, **{f"mask_{c}": m for c, m in masks.items()},
                   "cell_lat": ip.healpix_cell_latlon(level)[0],
                   "cell_lon": ip.healpix_cell_latlon(level)[1]}
            res["runs"] = {}
            for mode in args.modes:
                res["runs"][mode] = run_mode(self, src, mode, zt, masks, set(args.gain_days),
                                             D, npz, args.tag)
                (out_dir / f"latenterr_{args.tag}.json").write_text(
                    json.dumps(res, indent=1, default=float))
        np.savez_compressed(out_dir / f"latenterr_{args.tag}.npz", **npz)
        (out_dir / f"latenterr_{args.tag}.json").write_text(json.dumps(res, indent=1,
                                                                       default=float))
        logger.info(f"latent_error {args.tag}: written to {out_dir}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--days", type=int, default=150)
    p.add_argument("--modes", nargs="+", default=["none", "freeze"])
    p.add_argument("--gain-days", type=int, nargs="*", default=[40, 80])
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
    logging.getLogger("latent_error").setLevel(logging.INFO)
    logging.getLogger("instab_probe").setLevel(logging.INFO)
    cf.general.run_history += [(inf.from_run_id, cf.general.istep)]
    tr = LatErrTrainer(cf.train_logging)
    tr.args = args
    tr.inference(cf, devices, inf.from_run_id, inf.mini_epoch)


if __name__ == "__main__":
    main()
