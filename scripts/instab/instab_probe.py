#!/usr/bin/env python
"""
Instability probes of the fesom_vars models (TEST 1 and TEST 2 of the instab tests).

Runs the standard inference set-up (config merge, data sampler, model loading) through a
Trainer subclass whose `validate` runs a probe instead of the output-writing rollout:

    python instab_probe.py --mode encdec --out DIR --tag TAG [--steps 5] -- <inference args>
    python instab_probe.py --mode linear --out DIR --tag TAG --day N [...] -- <inference args>

<inference args> are those of the `inference` entry point (--from-run-id, -e, --run-id,
--options ...). The test config must give one sample (samples_per_mini_epoch=1) and a lazy
rollout (forecast.chunk_size < forecast.num_steps), so targets/forcing are built per step.

encdec  (TEST 1) encode the initial state x0 and decode the latent z0 = E(x0) without a
        forecast-engine step (step 0; out of distribution: the models were trained with
        forecast offset 1 only, the decoder has only seen forecast-engine outputs), then
        the normal rollout's steps 1..K. Per step and channel: wet-area-weighted mean error,
        RMSE, spatial std of the error (physical and normalised units), the truth's mean in
        normalised units, volume-mean T/S and heat-content errors, token statistics.
        -> OUT/encdec_TAG.json, OUT/encdec_TAG_maps.npz (error maps of a few channels).

linear  (TEST 2) free rollout (bf16 autocast, as the inference) to day N, then the
        Jacobian J of the one-day step F(z) = FE(z + forcing_{N+1}) at z_N, by central
        finite differences in fp32 (flash attention replaced by fp32 SDPA, TF32 off), and
        its leading eigenvalues with ARPACK (scipy.sparse.linalg.eigs, which='LM' and
        'SR'). Checks: F_bf16(z_N) == z_{N+1} of the rollout, decoded z_N vs the stored
        rollout output (by the caller), eigenpair residuals at other FD steps and in bf16.
        Leading eigenvectors are mapped to latent cells and decoded:
        (D(z + a v) - D(z - a v)) / 2a.
        -> OUT/linear_TAG_dN.json, OUT/linear_TAG_dN.npz

traj    free rollout (bf16 autocast) for --day D days, latent only; per day n: |z_n|, the
        relative first/second differences, the cosine of consecutive increments (all cells,
        ocean cells, land cells), the forcing injection |F_in(z_n) - z_n| / |z_n| of the
        step n -> n+1, and the per-cell token RMS on land vs ocean cells (HEALPix cells
        without vs with a mesh node). -> OUT/traj_TAG.json, OUT/traj_TAG.npz (cell maps)
        With test_config.forecast.land_cells set (freeze | no_forcing, see
        weathergen/model/land_cells.py) the rollout applies it, and the forcing statistics
        are those of the forcing as applied (no_forcing: zero on land cells).

Stream layouts: the fesom_vars layout (TEMP, SALT, U, V, SURFACE) and the older
FESOM_NODE / FESOM_ELEM layout (e.g. frcwin03, levels to 450 m). gain and traj (latent
only) work with both; encdec needs the fesom_vars layout (31 levels, mesh wet mask); linear
and baseline select the node stream(s) of the layout but were run only on fesom_vars models.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))  # paper_eval and diag_fvars_year are plain folders/files

import weathergen.common.config as config  # noqa: E402
import weathergen.model.attention as att_mod  # noqa: E402
import weathergen.utils.cli as cli  # noqa: E402
from weathergen.common.logger import init_loggers  # noqa: E402
from weathergen.model.land_cells import LandCells  # noqa: E402
from weathergen.model.model import ModelOutput  # noqa: E402
from weathergen.train.trainer import Trainer  # noqa: E402

logger = logging.getLogger("instab_probe")

STREAMS = ("TEMP", "SALT", "SURFACE")  # node streams of the fesom_vars layout
OLD_STREAMS = ("FESOM_NODE",)  # node stream of the FESOM_NODE / FESOM_ELEM layout
MESH_PATH = "/work/ab0995/a270225/atmorep_data/multiverse/core2/original/cycle_1/fesom.mesh.diag.nc"
MAP_CHANNELS = ("temp_2", "temp_75", "temp_125", "temp_450", "temp_975", "temp_2035",
                "salt_2", "salt_75", "salt_125", "salt_450", "ssh")
EIG_MAP_CHANNELS = ("salt_75", "salt_125", "temp_75", "temp_2", "temp_450", "ssh")
RHO0, CP = 1025.0, 3992.0
LAT_BANDS = tuple(range(-90, 91, 20))


# ── fp32 attention for the linearisation ─────────────────────────────────────────────
_orig_flash = att_mod.flash_attn_func


def _flash_or_sdpa(q, k, v, softcap=0.0, dropout_p=0.0, **kwargs):
    """flash_attn_func for bf16/fp16 inputs; exact fp32 SDPA for fp32 inputs."""
    if q.dtype == torch.float32:
        assert softcap == 0.0 and not kwargs.get("causal", False)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return out.transpose(1, 2)
    if q.dtype == torch.float64:  # explicit softmax attention, queries in chunks
        assert softcap == 0.0 and not kwargs.get("causal", False)
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = qt.shape[-1] ** -0.5
        outs = []
        for i in range(0, qt.shape[2], 1024):
            a = torch.softmax(qt[:, :, i:i + 1024] @ kt.transpose(-1, -2) * scale, dim=-1)
            outs.append(a @ vt)
        return torch.cat(outs, dim=2).transpose(1, 2)
    return _orig_flash(q, k, v, softcap=softcap, dropout_p=dropout_p, **kwargs)


att_mod.flash_attn_func = _flash_or_sdpa


def set_fe_attention_dtype(model, dtype):
    n = 0
    for m in model.forecast_engine.modules():
        if isinstance(m, att_mod.MultiSelfAttentionHead):
            m.dtype = dtype
            n += 1
    return n


# ── layout and geometry ──────────────────────────────────────────────────────────────
def node_streams(cf):
    """Decoded node streams of the model's stream layout."""
    if "TEMP" in cf.streams:
        return STREAMS
    if "FESOM_NODE" in cf.streams:
        return OLD_STREAMS
    raise ValueError(f"unknown stream layout: {list(cf.streams)}")


def mesh_path(cf):
    return cf.streams[node_streams(cf)[0]].get("mesh_path", MESH_PATH)


def ocean_cells(level, path=MESH_PATH):
    """True for the HEALPix cells (nested) holding at least one mesh node; the others are
    land cells without ocean data (same definition as report_test2.ocean_cells)."""
    from astropy_healpix import healpy as ahp
    from paper_eval.meshtools import load_mesh

    mesh = load_mesh(path)
    theta = np.radians(90.0 - mesh.lat)
    phi = np.radians(180.0 + mesh.lon)
    pix = ahp.ang2pix(2**level, theta, phi, nest=True)
    has = np.zeros(12 * 4**level, bool)
    has[pix] = True
    return has


def geometry(mesh_path):
    import diag_fvars_year as dfy

    return dfy.Geometry(mesh_path), dfy


def healpix_cell_latlon(level):
    from astropy_healpix import healpy as ahp

    theta, phi = ahp.pix2ang(2**level, np.arange(12 * 4**level), nest=True)
    return 90.0 - np.degrees(theta), np.degrees(phi) - 180.0


# ── batch helpers ────────────────────────────────────────────────────────────────────
def materialize_targets(ds, batch, step_range, device):
    """Targets (+ forcing) of an absolute step range; step 0 is the initial window."""
    if step_range[0] < ds.output_offset:
        off = ds.output_offset
        ds.output_offset = 0
        try:
            ds._materialize_target_chunk(batch, step_range)
        finally:
            ds.output_offset = off
    else:
        ds._materialize_target_chunk(batch, step_range)
    batch.to_device(device)


def materialize_forcing(ds, batch, step_range, device):
    """Forcing (and conditions) only: enough to advance the latent without decoding."""
    ctx = ds._materialize_ctx
    for _, cd in ds.condition_datasets.items():
        ds._build_condition_data(batch, cd.readers[0], ctx["idx"], ctx["num_output_steps"],
                                 step_range=step_range)
    for _, fd in ds.forcing_datasets.items():
        ds._build_forcing_data(batch, fd.readers[0], ctx["idx"], ctx["num_output_steps"],
                               step_range=step_range)
    batch.to_device(device, include_targets=False)


def stream_stats(ds, stream):
    r = ds.streams_datasets[stream].readers[0]
    r._lazy_init()
    mean = np.asarray(r.trg_mean_vec, np.float64).reshape(-1)
    std = np.asarray(r.trg_stdev_vec, np.float64).reshape(-1)
    chans = list(r.target_channels)
    assert len(chans) == len(mean) == len(std), (stream, len(chans), mean.shape, std.shape)
    return chans, mean, std


def to_node_order(batch, step, stream, pred):
    """pred (ens, n, C) in model order -> (n, C) float64 in the stored (node) order, with
    the target (normalised) and the target coordinates, as write_output does."""
    sd = batch.get_target_samples().samples[0].streams_data[stream]
    target, idxs_inv, coords = sd.target_tokens[step], sd.idxs_inv[step], sd.target_coords_raw[step]
    coords = coords.cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
    if idxs_inv is not None and len(idxs_inv) > 0:
        idx = torch.as_tensor(idxs_inv).long()
        pred = pred[:, idx.to(pred.device)]
        target = target[idx.to(target.device)]
        coords = coords[idx.cpu().numpy()]
    return (pred[0].float().cpu().numpy().astype(np.float64),
            target.float().cpu().numpy().astype(np.float64), coords)


def decode(model, model_params, z, src, step, streams, autocast=True):
    out = ModelOutput(1, batch=src)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
        model.predict_decoders(model_params, 0, step, z, src, out)
    return {s: out.physical[0][s][0] for s in streams}


def token_stats(z):
    z = z.float()
    norms = z.norm(dim=-1).flatten() / np.sqrt(z.shape[-1])
    return {"rms": float(z.pow(2).mean().sqrt()), "token_norm_p05": float(norms.quantile(0.05)),
            "token_norm_p50": float(norms.median()), "token_norm_p95": float(norms.quantile(0.95)),
            "token_mean_abs": float(z.mean(-1).abs().mean()), "token_std": float(z.std(-1).mean())}


# ── TEST 1 ───────────────────────────────────────────────────────────────────────────
def check_coords(coords, geo):
    """Max |dlat|, |dlon| between the target coordinates and the mesh nodes, for the
    column order (lat, lon) and (lon, lat); the smaller pair is the layout."""
    out = {}
    for name, (a, b) in (("lat_lon", (0, 1)), ("lon_lat", (1, 0))):
        lat, lon = coords[:, a], coords[:, b]
        dlon = np.abs(((lon - geo.lon) + 180.0) % 360.0 - 180.0)
        out[name] = (float(np.abs(lat - geo.lat).max()), float(dlon.max()))
    return out


def level_stats(geo, p, t, chans, mean, std):
    """Per channel: wet-area-weighted mean error, RMSE, spatial std (physical, normalised),
    truth mean (physical, normalised; area-weighted and plain node mean over wet nodes)."""
    out = {}
    for j, c in enumerate(chans):
        mask, w = geo.weight(c)
        e = p[:, j] - t[:, j]
        if not np.isfinite(t[mask, j]).all():
            raise ValueError(f"non-finite truth at wet points: {c}")
        w = np.where(mask, w, 0.0)
        e0 = np.where(mask, e, 0.0)
        ws = w.sum()
        me = (w * e0).sum() / ws
        rmse = np.sqrt((w * e0**2).sum() / ws)
        tm = (w * np.where(mask, t[:, j], 0.0)).sum() / ws
        out[c] = {"mean_err": me, "rmse": rmse, "std_err": np.sqrt(max(rmse**2 - me**2, 0.0)),
                  "mean_err_norm": me / std[j], "rmse_norm": rmse / std[j],
                  "truth_mean": tm, "truth_mean_norm": (tm - mean[j]) / std[j],
                  "truth_nodemean_norm": (t[mask, j].mean() - mean[j]) / std[j],
                  "stat_mean": mean[j], "stat_std": std[j]}
    return out


def column_stats(geo, f):
    """Volume-mean T, S (0-2150 m, 31 levels) and heat content error of the step."""
    temp = sorted((c for c in f if c.startswith("temp_")), key=geo.level)
    salt = sorted((c for c in f if c.startswith("salt_")), key=geo.level)
    nlev = len(temp)
    wet = np.stack([geo.wet(k) for k in range(nlev)])
    w = geo.area[:nlev] * wet
    dz = np.diff(geo.nz)[:nlev]
    vol = (w * dz[:, None]).sum()
    res = {}
    for name, chans in (("T", temp), ("S", salt)):
        ep = np.stack([np.where(wet[k], f[c][0] - f[c][1], 0.0) for k, c in enumerate(chans)])
        res[f"vm{name}_err"] = (w * dz[:, None] * ep).sum() / vol
    res["hc_err_ZJ"] = res["vmT_err"] * RHO0 * CP * vol / 1e21
    return res


def probe_encdec(tr, batch, args):
    ds, model, mp, dev = tr.dataset_val, tr.model, tr.model_params, tr.device
    K = args.steps
    if node_streams(tr.cf) != STREAMS:
        raise ValueError("encdec needs the fesom_vars stream layout")
    geo, _ = geometry(tr.cf.streams["TEMP"]["mesh_path"])
    materialize_targets(ds, batch, (0, 1), dev)
    materialize_targets(ds, batch, (1, K + 1), dev)
    src = batch.get_source_samples()
    stats = {s: stream_stats(ds, s) for s in STREAMS}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tokens, _ = model.encoder(mp, src)
        tokens = tokens.reshape((len(src), src.get_num_source_steps(), *tokens.shape[1:])).sum(1)
    preds = [decode(model, mp, tokens, src, 0, STREAMS)]
    latents = [tokens]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(mp, src, K)
    for i in range(K):
        preds.append({s: out.physical[i][s][0] for s in STREAMS})
        latents.append(out.latent[i]["latent_state"].z_pre_norm)

    times = {}
    res = {"tag": args.tag, "steps": {}, "tokens": {}, "init_time": None}
    maps = {}
    for s_idx in range(K + 1):
        f, lev = {}, {}
        for s in STREAMS:
            chans, mean, std = stats[s]
            pn, tn, coords = to_node_order(batch, s_idx, s, preds[s_idx][s])
            if pn.shape[0] != geo.n:
                raise ValueError(f"{s} step {s_idx}: {pn.shape[0]} points, mesh {geo.n}")
            if s_idx == 0 and s == "TEMP":
                res["coord_check_max_dlat_dlon"] = check_coords(coords, geo)
            p, t = pn * std + mean, tn * std + mean
            lev |= level_stats(geo, p, t, chans, mean, std)
            for j, c in enumerate(chans):
                f[c] = (p[:, j], t[:, j])
            sd = batch.get_target_samples().samples[0].streams_data[s]
            times[s_idx] = str(np.asarray(sd.target_times_raw[s_idx]).min())
        res["steps"][s_idx] = {"levels": lev, "column": column_stats(geo, f),
                               "valid_time_min": times[s_idx]}
        res["tokens"][s_idx] = token_stats(latents[s_idx])
        if s_idx in (0, 1, K):
            for c in MAP_CHANNELS:
                mask, _ = geo.weight(c)
                maps[f"err_s{s_idx}_{c}"] = np.where(mask, f[c][0] - f[c][1], np.nan).astype(
                    np.float32)
                if s_idx in (0, 1):
                    maps[f"true_s{s_idx}_{c}"] = np.where(mask, f[c][1], np.nan).astype(np.float32)
    # step-0 latent vs the forecast latents
    z0, z1 = latents[0].float(), latents[1].float()
    res["tokens"]["rel_dist_z1_z0"] = float((z1 - z0).norm() / z1.norm())
    res["tokens"]["cos_z1_z0"] = float((z1 * z0).sum() / (z1.norm() * z0.norm()))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"encdec_{args.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(out_dir / f"encdec_{args.tag}_maps.npz", **maps)
    logger.info(f"encdec {args.tag}: written to {out_dir}")


# ── TEST 2 ───────────────────────────────────────────────────────────────────────────
class StepMap:
    """F(z) = FE(z + forcing) for one day, in fp32 (SDPA attention) or bf16 autocast."""

    def __init__(self, model, model_params, src, fidx):
        self.model, self.mp, self.src, self.fidx = model, model_params, src, fidx
        self.forcing = src.forcing[fidx]
        self.condition = src.conditions[fidx]
        self.n_calls = 0

    def __call__(self, z, precision="fp32"):
        m = self.model
        self.n_calls += 1
        tf32 = torch.backends.cuda.matmul.allow_tf32
        if precision == "fp32":
            set_fe_attention_dtype(m, torch.float32)
            torch.backends.cuda.matmul.allow_tf32 = False
            ctx = torch.autocast(device_type="cuda", enabled=False)
            z = z.float()
        else:
            set_fe_attention_dtype(m, torch.bfloat16)
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        try:
            with ctx:
                t, cond = m.forcing_module(z, self.condition, self.forcing, m.num_aux_tokens)
                t = m.forecast_engine(t, cond, coords=self.mp.rope_coords)
        finally:
            set_fe_attention_dtype(m, torch.bfloat16)
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return t


def rollout_to_day(tr, batch, day, chunk=10):
    """Latents z_0..z_day of the free rollout (bf16 autocast, as the inference)."""
    ds, model, mp, dev = tr.dataset_val, tr.model, tr.model_params, tr.device
    materialize_forcing(ds, batch, (1, day + 2), dev)
    src = batch.get_source_samples()
    zs = {}
    source = src
    done = 0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        while done < day:
            n = min(chunk, day - done)
            source = model(mp, source, n)
            for i in range(n):
                zs[done + i + 1] = source.latent[i]["latent_state"].z_pre_norm
            done += n
    return zs


def latent_cell_energy(v, n_cells):
    e = v.reshape(n_cells, -1).pow(2).sum(-1)
    return (e / e.sum()).cpu().numpy()


def band_fractions(values, lat, weights=None):
    w = np.ones_like(values) if weights is None else weights
    tot = (w * values).sum()
    return {f"{a}..{b}": float((w * values)[(lat >= a) & (lat < b)].sum() / tot)
            for a, b in zip(LAT_BANDS[:-1], LAT_BANDS[1:])}


def probe_linear(tr, batch, args):
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    ds, model, mp, dev = tr.dataset_val, tr.model, tr.model_params, tr.device
    day = args.day
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"linear_{args.tag}_d{day:03d}"
    res = {"tag": args.tag, "day": day, "eps_rel": args.eps, "ncv": args.ncv, "k": args.k}

    t0 = time.time()
    zs = rollout_to_day(tr, batch, day + 1)
    res["rollout_seconds"] = time.time() - t0
    z = zs[day].detach().clone().float()
    z_next = zs[day + 1].detach().clone().float()
    # day-to-day change of the carried latent along the rollout: relative first and second
    # differences and the cosine between consecutive increments (-1: pure alternation)
    traj = {"n": [], "d1_rel": [], "d2_rel": [], "cos_consecutive_increments": []}
    for k in range(2, day + 2):
        inc1, inc0 = zs[k] - zs[k - 1], zs[k - 1] - zs[k - 2] if k >= 3 else None
        if inc0 is None:
            continue
        nk = zs[k].norm()
        traj["n"].append(k)
        traj["d1_rel"].append(float(inc1.norm() / nk))
        traj["d2_rel"].append(float((inc1 - inc0).norm() / nk))
        traj["cos_consecutive_increments"].append(
            float((inc1 * inc0).sum() / (inc1.norm() * inc0.norm())))
    res["trajectory"] = traj
    alt = (zs[day + 1] - 2 * zs[day] + zs[day - 1]).float() if day >= 2 else None
    del zs
    torch.cuda.empty_cache()
    src = batch.get_source_samples()
    F = StepMap(model, mp, src, day + 1)

    # the step map reproduces the rollout's step, and fp32 vs bf16
    with torch.no_grad():
        fb = F(z, "bf16").float()
        t1 = time.time()
        f32 = F(z, "fp32")
        torch.cuda.synchronize()
        res["fp32_step_seconds"] = time.time() - t1
    res["check_bf16_step_vs_rollout_rel"] = float((fb - z_next).norm() / z_next.norm())
    res["check_fp32_vs_bf16_step_rel"] = float((f32 - fb).norm() / fb.norm())
    res["z_norm"] = float(z.norm())
    res["step_rel_change"] = float((f32 - z).norm() / z.norm())
    res["tokens_z"] = token_stats(z)
    logger.info(f"checks: {json.dumps({k: v for k, v in res.items() if k != 'tokens_z'})}")

    shape = z.shape
    n = z.numel()
    znorm = z.norm()

    def jv(v_t, eps_rel=args.eps, precision="fp32"):
        vn = v_t.norm()
        if vn == 0:
            return torch.zeros_like(v_t)
        e = eps_rel * znorm / vn
        with torch.no_grad():
            fp = F(z + e * v_t, precision).float()
            fm = F(z - e * v_t, precision).float()
        return (fp - fm) / (2 * e)

    n_mv = [0]

    def matvec(x):
        n_mv[0] += 1
        x = np.asarray(x).reshape(-1)
        if np.iscomplexobj(x):
            re = jv(torch.from_numpy(x.real.astype(np.float32)).to(dev).reshape(shape))
            im = jv(torch.from_numpy(x.imag.astype(np.float32)).to(dev).reshape(shape))
            return (re.reshape(-1).double().cpu().numpy()
                    + 1j * im.reshape(-1).double().cpu().numpy())
        y = jv(torch.from_numpy(x.astype(np.float32)).to(dev).reshape(shape))
        return y.reshape(-1).double().cpu().numpy()

    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    rng = np.random.default_rng(1234)
    v0 = rng.standard_normal(n)
    eig_runs = {}
    for which, k in (("LM", args.k), ("SR", args.k_sr)):
        if k <= 0:
            continue
        n_mv[0] = 0
        t1 = time.time()
        try:
            vals, vecs = eigs(op, k=k, which=which, ncv=args.ncv, tol=args.tol,
                              maxiter=args.maxiter, v0=v0)
            converged = True
        except ArpackNoConvergence as err:
            vals, vecs, converged = err.eigenvalues, err.eigenvectors, False
        order = np.argsort(-np.abs(vals)) if which == "LM" else np.argsort(vals.real)
        vals, vecs = vals[order], vecs[:, order]
        eig_runs[which] = {"vals": vals, "vecs": vecs, "converged": converged,
                           "n_matvec": n_mv[0], "seconds": time.time() - t1}
        logger.info(f"ARPACK {which}: converged={converged} n_matvec={n_mv[0]} "
                    f"vals={np.round(vals, 4).tolist()}")
        res[f"eigs_{which}"] = {"real": vals.real.tolist(), "imag": vals.imag.tolist(),
                                "abs": np.abs(vals).tolist(), "converged": converged,
                                "n_matvec": n_mv[0], "seconds": time.time() - t1}
        (stem.with_suffix(".json")).write_text(json.dumps(res, indent=1, default=float))

    # residuals and sensitivity of the eigenpairs; latent and decoded structure
    streams = node_streams(tr.cf)
    geo, dfy = geometry(mesh_path(tr.cf))
    cell_lat, cell_lon = healpix_cell_latlon(tr.cf.healpix_level)
    n_cells = 12 * 4**tr.cf.healpix_level
    materialize_targets(ds, batch, (day, day + 1), dev)
    stats = {s: stream_stats(ds, s) for s in streams}
    base = decode(model, mp, z, src, day, streams)
    npz = {"cell_lat": cell_lat, "cell_lon": cell_lon}
    # decoded state at day N (node order), for the comparison with the stored rollout output
    for s in streams:
        chans, mean, std = stats[s]
        pn, tn, _ = to_node_order(batch, day, s, base[s])
        for c in ("salt_75", "temp_2", "ssh"):
            if c in chans:
                j = chans.index(c)
                npz[f"decoded_{c}"] = (pn[:, j] * std[j] + mean[j]).astype(np.float32)
                npz[f"truth_{c}"] = (tn[:, j] * std[j] + mean[j]).astype(np.float32)
    basin = dfy.basins(geo.lon, geo.lat)

    pairs = []
    for which, run in eig_runs.items():
        for i in range(min(len(run["vals"]), args.n_map if which == "LM" else args.n_map_sr)):
            pairs.append((which, i, run["vals"][i], run["vecs"][:, i]))
    res["eigvecs"] = []
    for which, i, lam, vec in pairs:
        info = {"which": which, "index": i, "lambda": [float(lam.real), float(lam.imag)]}
        vr = torch.from_numpy(np.real(vec).astype(np.float32)).to(dev).reshape(shape)
        vi = torch.from_numpy(np.imag(vec).astype(np.float32)).to(dev).reshape(shape)
        is_complex = abs(lam.imag) > 1e-6 * max(abs(lam), 1e-12)
        for eps_rel in sorted({args.eps, args.eps * 0.1, args.eps * 10}):
            jr = jv(vr, eps_rel)
            ji = jv(vi, eps_rel) if is_complex else torch.zeros_like(jr)
            # J (vr + i vi) - lam (vr + i vi)
            rr = jr - (lam.real * vr - lam.imag * vi)
            ri = ji - (lam.real * vi + lam.imag * vr)
            nv = torch.sqrt(vr.norm() ** 2 + vi.norm() ** 2)
            info[f"resid_eps{eps_rel:.0e}"] = float(torch.sqrt(rr.norm() ** 2 + ri.norm() ** 2)
                                                    / nv)
            # Rayleigh quotient v^H J v / v^H v
            num_r = (vr * jr).sum() + (vi * ji).sum()
            num_i = (vr * ji).sum() - (vi * jr).sum()
            info[f"rayleigh_eps{eps_rel:.0e}"] = [float(num_r / nv**2), float(num_i / nv**2)]
        for eps_rel in (1e-2, 3e-2):
            jr = jv(vr, eps_rel, "bf16")
            ji = jv(vi, eps_rel, "bf16") if is_complex else torch.zeros_like(jr)
            nv = torch.sqrt(vr.norm() ** 2 + vi.norm() ** 2)
            num_r = (vr * jr).sum() + (vi * ji).sum()
            num_i = (vr * ji).sum() - (vi * jr).sum()
            info[f"rayleigh_bf16_eps{eps_rel:.0e}"] = [float(num_r / nv**2), float(num_i / nv**2)]
        if alt is not None:
            # alignment of the rollout's own second difference at day N with the eigenvector
            nv = torch.sqrt(vr.norm() ** 2 + vi.norm() ** 2)
            proj = torch.sqrt(((alt * vr).sum()) ** 2 + ((alt * vi).sum()) ** 2)
            info["cos_with_rollout_second_difference"] = float(proj / (alt.norm() * nv))
        # latent structure (real part)
        ce = latent_cell_energy(vr, n_cells)
        info["latent_lat_bands"] = band_fractions(ce, cell_lat)
        top = np.argsort(-ce)
        info["latent_top1pct_share"] = float(ce[top[: n_cells // 100]].sum())
        npz[f"cellenergy_{which}{i}"] = ce.astype(np.float32)
        # decoded response of the real part
        vrn = vr / vr.norm()
        info["decoded"] = {}
        resp_first = None
        for alpha in args.alphas:
            a = alpha * float(znorm)
            dp = decode(model, mp, z + a * vrn, src, day, streams)
            dm = decode(model, mp, z - a * vrn, src, day, streams)
            resp = {}
            for s in streams:
                chans, mean, std = stats[s]
                rp, _, _ = to_node_order(batch, day, s, dp[s])
                rm, _, _ = to_node_order(batch, day, s, dm[s])
                r_norm = (rp - rm) / (2 * a)  # normalised units per unit latent amplitude
                for j, c in enumerate(chans):
                    mask, w = geo.weight(c)
                    w = np.where(mask, w, 0.0)
                    resp[c] = (r_norm[:, j], w, std[j])
            per_ch = {}
            for c, (r, w, sd) in resp.items():
                rms_n = float(np.sqrt((w * r**2).sum() / w.sum()))
                per_ch[c] = {"rms_norm": rms_n, "rms_phys": rms_n * sd}
            info["decoded"][f"alpha{alpha:g}"] = {"per_channel": per_ch}
            if alpha == args.alphas[-1]:
                det = {}
                for c in ("salt_75", "salt_125", "temp_75", "temp_2", "ssh"):
                    r, w, sd = resp[c]
                    e2 = w * r**2
                    det[c] = {"lat_bands": band_fractions(e2, geo.lat),
                              "basins": {b: float(e2[basin == bi].sum() / e2.sum())
                                         for bi, b in enumerate(dfy.BASINS)}}
                info["decoded"]["regions_alpha_last"] = det
                for c in EIG_MAP_CHANNELS:
                    r, w, sd = resp[c]
                    mask, _ = geo.weight(c)
                    npz[f"resp_{which}{i}_{c}"] = np.where(mask, r * sd, np.nan).astype(
                        np.float32)
            if resp_first is None:
                resp_first = resp
            else:  # linearity: response pattern and amplitude at the first vs this alpha
                lin = {}
                for c in ("salt_75", "temp_75", "temp_2", "ssh"):
                    (x, w, _), (y, _, _) = resp_first[c], resp[c]
                    m = w > 0
                    lin[c] = {"corr": float(np.corrcoef(x[m], y[m])[0, 1]),
                              "rms_ratio": float(np.sqrt((w * y**2).sum() / (w * x**2).sum()))}
                info["decoded"][f"linearity_alpha{args.alphas[0]:g}_vs_{alpha:g}"] = lin
        res["eigvecs"].append(info)
        (stem.with_suffix(".json")).write_text(json.dumps(res, indent=1, default=float))
        logger.info(f"eigvec {which}{i} lam={lam:.4f}: done")
    np.savez_compressed(stem.with_suffix(".npz"), **npz)
    (stem.with_suffix(".json")).write_text(json.dumps(res, indent=1, default=float))
    logger.info(f"linear {args.tag} day {day}: written {stem}")


def probe_baseline(tr, batch, args):
    """Decoded response of reference latent directions at day N of the free rollout, with
    the same definitions as the eigenvector responses of `probe_linear`: a random Gaussian
    direction (3 seeds), the rollout's own day-to-day increment z_(N+1) - z_N and its
    second difference z_(N+1) - 2 z_N + z_(N-1). -> OUT/baseline_TAG_dN.json"""
    ds, model, mp, dev = tr.dataset_val, tr.model, tr.model_params, tr.device
    day = args.day
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zs = rollout_to_day(tr, batch, day + 1)
    z = zs[day].float()
    dirs = {"increment": (zs[day + 1] - zs[day]).float(),
            "second_difference": (zs[day + 1] - 2 * zs[day] + zs[day - 1]).float()}
    for seed in range(3):
        g = torch.Generator(device=dev).manual_seed(seed)
        dirs[f"random{seed}"] = torch.randn(z.shape, generator=g, device=dev)
    del zs
    torch.cuda.empty_cache()
    src = batch.get_source_samples()
    streams = node_streams(tr.cf)
    geo, dfy = geometry(mesh_path(tr.cf))
    basin = dfy.basins(geo.lon, geo.lat)
    cell_lat, _ = healpix_cell_latlon(tr.cf.healpix_level)
    n_cells = 12 * 4**tr.cf.healpix_level
    materialize_targets(ds, batch, (day, day + 1), dev)
    stats = {s: stream_stats(ds, s) for s in streams}
    znorm = float(z.norm())
    res = {"tag": args.tag, "day": day, "z_norm": znorm, "directions": {}}
    for name, v in dirs.items():
        info = {"rel_norm": float(v.norm()) / znorm}
        ce = latent_cell_energy(v, n_cells)
        info["latent_lat_bands"] = band_fractions(ce, cell_lat)
        info["latent_top1pct_share"] = float(np.sort(ce)[::-1][: n_cells // 100].sum())
        vn = v / v.norm()
        info["decoded"] = {}
        for alpha in args.alphas:
            a = alpha * znorm
            dp = decode(model, mp, z + a * vn, src, day, streams)
            dm = decode(model, mp, z - a * vn, src, day, streams)
            per_ch, regions = {}, {}
            for s in streams:
                chans, mean, std = stats[s]
                rp, _, _ = to_node_order(batch, day, s, dp[s])
                rm, _, _ = to_node_order(batch, day, s, dm[s])
                r_norm = (rp - rm) / (2 * a)
                for j, c in enumerate(chans):
                    mask, w = geo.weight(c)
                    w = np.where(mask, w, 0.0)
                    rms_n = float(np.sqrt((w * r_norm[:, j] ** 2).sum() / w.sum()))
                    per_ch[c] = {"rms_norm": rms_n, "rms_phys": rms_n * std[j]}
                    if c in ("salt_75", "salt_125", "temp_75", "temp_2", "ssh"):
                        e2 = w * r_norm[:, j] ** 2
                        regions[c] = {"lat_bands": band_fractions(e2, geo.lat),
                                      "basins": {b: float(e2[basin == bi].sum() / e2.sum())
                                                 for bi, b in enumerate(dfy.BASINS)}}
            info["decoded"][f"alpha{alpha:g}"] = {"per_channel": per_ch}
            info["decoded"]["regions_alpha_last"] = regions
        res["directions"][name] = info
        logger.info(f"baseline {name}: done")
    (out_dir / f"baseline_{args.tag}_d{day:03d}.json").write_text(
        json.dumps(res, indent=1, default=float))


def probe_gain(tr, batch, args):
    """Finite-difference gain of the one-day step along reference directions at day N, as a
    function of the perturbation amplitude, in fp32 and fp64 (deep copy of the forcing module
    and forecast engine in double precision):
      q(eps) = <v, J_eps v>, |J_eps v|, with J_eps v = (F(z + e v) - F(z - e v)) / 2e,
      e = eps |z|, |v| = 1,
    for v = the rollout's second difference and increment at day N and random Gaussian
    directions; additivity |J(a+b) - J(a) - J(b)| / |J(a) + J(b)| for random a, b; and the
    size of the forcing injection |F_in(z) - z| / |z|. -> OUT/gain_TAG_dN.json"""
    import copy

    model, mp, dev = tr.model, tr.model_params, tr.device
    day = args.day
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zs = rollout_to_day(tr, batch, day + 1)
    z = zs[day].float()
    traj = {"n": [], "d1_rel": [], "cos_consecutive_increments": []}
    for k in range(max(3, day - 3), day + 2):
        inc1, inc0 = (zs[k] - zs[k - 1]).float(), (zs[k - 1] - zs[k - 2]).float()
        traj["n"].append(k)
        traj["d1_rel"].append(float(inc1.norm() / zs[k].float().norm()))
        traj["cos_consecutive_increments"].append(
            float((inc1 * inc0).sum() / (inc1.norm() * inc0.norm())))
    dirs = {"second_difference": (zs[day + 1] - 2 * zs[day] + zs[day - 1]).float(),
            "increment": (zs[day + 1] - zs[day]).float()}
    for seed in range(3):
        g = torch.Generator(device=dev).manual_seed(seed)
        dirs[f"random{seed}"] = torch.randn(z.shape, generator=g, device=dev)
    dirs = {k: v / v.norm() for k, v in dirs.items()}
    del zs
    torch.cuda.empty_cache()
    src = batch.get_source_samples()
    F = StepMap(model, mp, src, day + 1)
    znorm = float(z.norm())
    res = {"tag": args.tag, "day": day, "z_norm": znorm, "trajectory": traj}
    with torch.no_grad():
        zin, _ = model.forcing_module(z, F.condition, F.forcing, model.num_aux_tokens)
        res["forcing_injection_rel"] = float((zin - z).norm()) / znorm

        def jv(v, eps, prec):
            e = eps * znorm
            return (F(z + e * v, prec).float() - F(z - e * v, prec).float()) / (2 * e)

        res["fp32"] = {}
        for name, v in dirs.items():
            rows = {}
            for eps in (1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1):
                j = jv(v, eps, "fp32")
                rows[f"{eps:.0e}"] = {"q": float((v * j).sum()), "gain": float(j.norm())}
            res["fp32"][name] = rows
            logger.info(f"gain fp32 {name}: {rows}")
        res["bf16"] = {}
        for name in ("second_difference", "increment", "random0"):
            rows = {}
            for eps in (1e-2, 3e-2, 1e-1, 3e-1):
                j = jv(dirs[name], eps, "bf16")
                rows[f"{eps:.0e}"] = {"q": float((dirs[name] * j).sum()), "gain": float(j.norm())}
            res["bf16"][name] = rows
        res["additivity"] = {}
        a, b = dirs["random0"], dirs["random1"]
        for eps in (1e-4, 1e-3, 1e-2, 1e-1):
            ja, jb = jv(a, eps, "fp32"), jv(b, eps, "fp32")
            jab = jv((a + b) / np.sqrt(2), eps, "fp32") * np.sqrt(2)
            res["additivity"][f"{eps:.0e}"] = float((jab - ja - jb).norm() / (ja + jb).norm())
        (out_dir / f"gain_{args.tag}_d{day:03d}.json").write_text(json.dumps(res, indent=1))

        # fp64: copies of the forcing module and forecast engine in double precision
        fe64 = copy.deepcopy(model.forecast_engine).double()
        fm64 = copy.deepcopy(model.forcing_module).double()
        for m in fe64.modules():
            if isinstance(m, att_mod.MultiSelfAttentionHead):
                m.dtype = torch.float64
        forcing64 = F.forcing.double() if torch.is_tensor(F.forcing) else F.forcing
        rope = mp.rope_coords.double() if torch.is_tensor(mp.rope_coords) else mp.rope_coords
        z64 = z.double()

        def f64(x):
            with torch.autocast(device_type="cuda", enabled=False):
                t, cond = fm64(x, F.condition, forcing64, model.num_aux_tokens)
                return fe64(t, cond, coords=rope)

        res["fp64_vs_fp32_step_rel"] = float((f64(z64).float() - F(z, "fp32")).norm()
                                             / F(z, "fp32").norm())
        res["fp64"] = {}
        for name in ("second_difference", "increment", "random0"):
            v = dirs[name].double()
            rows = {}
            for eps in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
                e = eps * znorm
                j = (f64(z64 + e * v) - f64(z64 - e * v)) / (2 * e)
                rows[f"{eps:.0e}"] = {"q": float((v * j).sum()), "gain": float(j.norm())}
            res["fp64"][name] = rows
            logger.info(f"gain fp64 {name}: {rows}")
    (out_dir / f"gain_{args.tag}_d{day:03d}.json").write_text(json.dumps(res, indent=1))


def probe_traj(tr, batch, args, chunk=10):
    """Latent statistics along the free rollout, days 1..D (see the module docstring)."""
    ds, model, mp, dev = tr.dataset_val, tr.model, tr.model_params, tr.device
    D = args.day
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    level = tr.cf.healpix_level
    n_cells = 12 * 4**level
    ocean_np = ocean_cells(level, mesh_path(tr.cf))
    ocean = torch.from_numpy(ocean_np).to(dev)
    land = ~ocean
    materialize_forcing(ds, batch, (1, D + 2), dev)
    src = batch.get_source_samples()
    n_forcing = len(src.forcing) if getattr(src, "forcing", None) is not None else 0
    save_days = sorted({d for d in (1, 10, 40, 58, 70, 78, 100, 150, 200, 300, 360) if d <= D}
                       | {D})
    keys = ("n", "z_norm", "z_rms", "cell_rms_ocean", "cell_rms_land", "cell_meannorm_ocean",
            "cell_meannorm_land", "d1_rel", "d1_cell_rms_ocean", "d1_cell_rms_land",
            "d1_land_energy_share", "d2_rel", "d2_land_energy_share", "cos_inc", "cos_inc_ocean",
            "cos_inc_land", "forcing_rel", "forcing_cell_rms_ocean", "forcing_cell_rms_land",
            "forcing_land_energy_share")
    T = {k: [] for k in keys}
    land_cells = getattr(tr, "land_cells", None)
    lc_kwargs = {} if land_cells is None else {"land_cells": land_cells}
    if land_cells is not None:
        if not np.array_equal(land_cells.land_mask.numpy(), ~ocean_np):
            raise ValueError("forecast.land_cells mask differs from the probe's land cells")
        logger.info(f"traj {args.tag}: land cells {land_cells.as_dict()}")
    npz = {"ocean_cells": ocean_np, "cell_lat": healpix_cell_latlon(level)[0],
           "cell_lon": healpix_cell_latlon(level)[1]}
    sum_cell = torch.zeros(n_cells, dtype=torch.float64, device=dev)

    def cell_ms(x):  # per-cell mean square over the token dimension
        return x.reshape(n_cells, -1).pow(2).mean(-1)

    def cos(a, b):
        return float((a * b).sum() / (a.norm() * b.norm()))

    def stats(n, z, zm1, zm2):
        z = z.float()
        c = cell_ms(z)
        T["n"].append(n)
        T["z_norm"].append(float(z.norm()))
        T["z_rms"].append(float(z.pow(2).mean().sqrt()))
        T["cell_rms_ocean"].append(float(c[ocean].mean().sqrt()))
        T["cell_rms_land"].append(float(c[land].mean().sqrt()))
        T["cell_meannorm_ocean"].append(float(c[ocean].sqrt().mean()))
        T["cell_meannorm_land"].append(float(c[land].sqrt().mean()))
        sum_cell.add_(c.sqrt().double())
        if n in save_days:
            npz[f"cell_rms_d{n:03d}"] = c.sqrt().cpu().numpy().astype(np.float32)
        nan = float("nan")
        if zm1 is not None:
            i1 = (z - zm1.float())
            ci = cell_ms(i1)
            T["d1_rel"].append(float(i1.norm() / z.norm()))
            T["d1_cell_rms_ocean"].append(float(ci[ocean].mean().sqrt()))
            T["d1_cell_rms_land"].append(float(ci[land].mean().sqrt()))
            T["d1_land_energy_share"].append(float(ci[land].sum() / ci.sum()))
        else:
            for k in ("d1_rel", "d1_cell_rms_ocean", "d1_cell_rms_land", "d1_land_energy_share"):
                T[k].append(nan)
        if zm2 is not None:
            i0 = (zm1 - zm2).float()
            d2 = i1 - i0
            c2 = cell_ms(d2)
            T["d2_rel"].append(float(d2.norm() / z.norm()))
            T["d2_land_energy_share"].append(float(c2[land].sum() / c2.sum()))
            T["cos_inc"].append(cos(i1, i0))
            a1, a0 = i1.reshape(n_cells, -1), i0.reshape(n_cells, -1)
            T["cos_inc_ocean"].append(cos(a1[ocean], a0[ocean]))
            T["cos_inc_land"].append(cos(a1[land], a0[land]))
            if n in save_days:
                npz[f"d2_cell_rms_d{n:03d}"] = c2.sqrt().cpu().numpy().astype(np.float32)
        else:
            for k in ("d2_rel", "d2_land_energy_share", "cos_inc", "cos_inc_ocean",
                      "cos_inc_land"):
                T[k].append(nan)
        # forcing injection of the step n -> n+1 (forcing / condition index n + 1)
        fidx = n + 1
        if fidx < n_forcing and model.forcing_module is not None:
            # fp32, as forcing_injection_rel of the gain mode
            with torch.autocast(device_type="cuda", enabled=False):
                zin, _ = model.forcing_module(z, src.conditions[fidx], src.forcing[fidx],
                                              model.num_aux_tokens)
                if land_cells is not None:  # the forcing as applied in the rollout
                    zin = land_cells.after_forcing(z, zin, model.num_aux_tokens)
            fi = zin.float() - z
            cf_ = cell_ms(fi)
            T["forcing_rel"].append(float(fi.norm() / z.norm()))
            T["forcing_cell_rms_ocean"].append(float(cf_[ocean].mean().sqrt()))
            T["forcing_cell_rms_land"].append(float(cf_[land].mean().sqrt()))
            T["forcing_land_energy_share"].append(float(cf_[land].sum() / cf_.sum()))
        else:
            for k in ("forcing_rel", "forcing_cell_rms_ocean", "forcing_cell_rms_land",
                      "forcing_land_energy_share"):
                T[k].append(nan)

    t0 = time.time()
    hist = []  # z_(n-2), z_(n-1)
    source, done = src, 0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        while done < D:
            n = min(chunk, D - done)
            source = model(mp, source, n, **lc_kwargs)
            for i in range(n):
                z = source.latent[i]["latent_state"].z_pre_norm.detach()
                zm1 = hist[-1] if len(hist) >= 1 else None
                zm2 = hist[-2] if len(hist) >= 2 else None
                stats(done + i + 1, z, zm1, zm2)
                hist = (hist + [z])[-2:]
            done += n
            logger.info(f"traj {args.tag}: day {done}, |z| {T['z_norm'][-1]:.1f}, "
                        f"d1 {T['d1_rel'][-1]:.3f}, cos {T['cos_inc'][-1]:.3f}, "
                        f"forcing {T['forcing_rel'][-1]:.3f}")
    npz["cell_rms_mean"] = (sum_cell / D).cpu().numpy().astype(np.float32)
    res = {"tag": args.tag, "days": D, "seconds": time.time() - t0,
           "n_cells": n_cells, "n_ocean_cells": int(ocean_np.sum()),
           "n_land_cells": int((~ocean_np).sum()), "token_dim": int(z.shape[-1]),
           "z_shape": list(z.shape),
           "land_cells": None if land_cells is None else land_cells.as_dict(),
           "trajectory": T}
    (out_dir / f"traj_{args.tag}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out_dir / f"traj_{args.tag}.npz", **npz)
    logger.info(f"traj {args.tag}: written to {out_dir}")


class ProbeTrainer(Trainer):
    probe_args = None

    def validate(self, mini_epoch, mode_cfg, batch_size):
        self.model.eval()
        batch = next(iter(self.data_loader_validation))
        batch.to_device(self.device, include_targets=False)
        args = self.probe_args
        self.land_cells = LandCells.from_config(
            mode_cfg.get("forecast", {}).get("land_cells"), self.cf.streams, self.cf.healpix_level)
        if self.land_cells is not None and args.mode != "traj":
            raise ValueError("forecast.land_cells is only supported by the traj probe")
        with torch.no_grad():
            if args.mode == "encdec":
                probe_encdec(self, batch, args)
            elif args.mode == "baseline":
                probe_baseline(self, batch, args)
            elif args.mode == "gain":
                probe_gain(self, batch, args)
            elif args.mode == "traj":
                probe_traj(self, batch, args)
            else:
                logger.info(f"FE attention blocks: {set_fe_attention_dtype(self.model, torch.bfloat16)}")
                probe_linear(self, batch, args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["encdec", "linear", "baseline", "gain", "traj"])
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--day", type=int, default=10)
    p.add_argument("--eps", type=float, default=1e-3, help="FD step relative to |z|")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--k-sr", type=int, default=6)
    p.add_argument("--ncv", type=int, default=40)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--n-map", type=int, default=4)
    p.add_argument("--n-map-sr", type=int, default=2)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.03])
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
    logging.getLogger("instab_probe").setLevel(logging.INFO)
    cf.general.run_history += [(inf.from_run_id, cf.general.istep)]
    tr = ProbeTrainer(cf.train_logging)
    tr.probe_args = args
    tr.inference(cf, devices, inf.from_run_id, inf.mini_epoch)


if __name__ == "__main__":
    main()
