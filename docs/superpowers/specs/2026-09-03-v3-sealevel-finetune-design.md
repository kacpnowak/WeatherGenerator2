# v3 sea-level finetune (`seaft`) — design

Date: 2026-09-03. Status: approved in brainstorming (Kacper), pending spec review.
Supersedes Task 11 of `2026-08-19-v3-beat-glonet.md` for the v3 run `glorys_v3_20260819`.

## 1. Objective

A short, targeted finetune from v3 epoch 45 that closes v3's gap to the honestly evaluated
ft0808 (`glory_finetuning_20260808_191038`, ERA5 IC-window forcing) on the **sea-level
family** — `zos`, `uo_0.494025m`, `vo_0.494025m` at all leads — without regressing the rest
of the 33-channel column or deepening the temperature cold bias.

Why this family: zos is v3's weakest channel (+11–19% RMSE vs ft0808) and, through
`ssh`, `ssh_glo12`, `class4_sla`, `geo_u/geo_v` and the Lagrangian row, drives the
scoreboard rows we lose or only narrowly win.

## 2. Starting point (verified facts)

- Base checkpoint: `glorys_v3_20260819_chkpt00045.chkpt` (EMA weights). Checkpoint sweep
  e45/47/48/50/52/54 (`eval_output/compare_sweep45_54.txt`): e45 is best overall (RMSE
  ratio 1.023 vs ft0808); e48 within noise; e54+ worse with a growing cold bias.
- Training chain is stopped (~e56 of 96).
- All evaluation uses the training ERA5 zarr for the ATMO stream (commit dfce5f61); the
  OceanBench IFS parquets are retired.
- Levers already in the code: `target_channel_weights` mapping per channel
  (`data_reader_mesh.py`), `freeze_modules` regex (`re.fullmatch` over `named_modules()`
  names, `model/utils.py:apply_fct_to_blocks`), 10-step forecast training (phase C config).

## 3. Trainable scope

Trainable: the aggregation engine (`encoder.ae_aggregation_engine`) and the GLORYS3
decoder stack (`embed_target_coords.GLORYS3`, `target_token_engines.GLORYS3`,
`pred_heads.GLORYS3`). Everything else frozen, including the encoder-side GLORYS3 source
embedding, the ATMO embedding, local/global engines, the forecast engine and latent heads.

```yaml
freeze_modules: ".*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*forecast_engine.*|.*latent.*|encoder\\..*GLORYS.*|.*ATMO.*"
```

The only change from the Task 11 regex is `.*GLORYS.*` → `encoder\..*GLORYS.*`.

First-log acceptance: freeze lines present for local/global/adapter engines, forecast
engine, ATMO; **no** freeze line for `ae_aggregation_engine`, `target_token_engines.GLORYS3`
or `pred_heads.GLORYS3`.

## 4. Loss

Plain MSE × static channel weights; dynamic loss **off**; 10-step horizon, daily windows.

Weights are gap-derived from our own diagnostics:

```
ratio_c  = mean over {2024-01-03, 2024-07-03} × leads {1,5,10} of RMSE_e45,c / RMSE_ft0808,c
weight_c = clip(1 + 40 · (ratio_c − 1), 1, 8)
```

computed from `compare_sweep45_54.txt` and stored in `eval_output/seaft_channel_weights.json`.
Resulting values: `zos` 6.91; surface thetao 2.4–3.5; deep so 1.6–2.4; deep thetao
1.5–1.8; the rest ≈ 1. Depths without an exact reference level (9.6, 15.8, 25.2, 34.4,
65.8, 130.7 m) are scored against the nearest scoreboard level; their ratios are still
cross-model comparable and receive rule weights.

One explicit deviation from the rule: **`uo_0.494025m` and `vo_0.494025m` get a family
floor of 3.0** (rule gives 1.4–1.5 because their lead-1 gap is small; the lead-10 gap is
+5–6% and they feed the geostrophic and Lagrangian rows).

All 33 target channels stay in the loss so the unfrozen aggregation engine cannot let
un-targeted channels drift.

## 5. Data, schedule, validation

| item | value |
|---|---|
| training window | 2020-01-01 → 2023-12-31 (recency; includes the warm 2023-H2 regime) |
| windows | 24 h step / 24 h length, 10-step forecast, `pushforward: False` |
| samples per epoch | 4096 |
| epochs | 4 (short; the ft0818 lesson: long grinds reverse their gains) |
| LR | warmup 128 steps to `lr_max` 2e-5, cosine decay to 0, `parallel_scaling_policy: sqrt` |
| EMA | as phase C (`validate_with_ema`, halflife 4000 samples, ramp 0.09) |
| validation | 2019-01-01 → 2019-12-31, 64 samples (2022 is now inside the training window) |
| checkpoints | every 64 steps (`_latest`) and per epoch (`chkpt00000`–`chkpt00003`) |
| streams | `config/streams/glorys_v3_seaft/` = copy of `glorys_v3/` with `target_channel_weights` mapping on GLORYS3; ATMO = ERA5 zarr |
| account | `e-ext-2025e01-128` |

## 6. Run mechanics

- New run id (`glorys_v3_seaft_<date>`), forked from `glorys_v3_20260819` with
  `--mini-epoch 45` (`train_continue`).
- The finetune config **sets `general.istep: 0`**: the trainer derives the epoch counter
  from the run json's istep, and the inherited value (~23 000) would put the counter at 45
  > 4 so the training loop would not run.
- Because a config passed on a resume link is re-applied and would reset istep again, the
  **first 12 h link carries the config; any later link is a pure resume** (no
  `WEATHERGEN_CONFIG_EXTRA`). Expected ~2 epochs per link at the 10-step cost, so two links.
- Single test link first (Section 8); the second link is queued only after verification.

## 7. Selection and hand-off

After every finished epoch N (chkpt0000N):

1. Both 2024 smoke dates through the standard ERA5 pipeline
   (`v2_model_smoke.sh <run> seaftN <date> N config/config_forecasting_glorys_obench_v3.yml GLORYS3`).
2. `full_compare.py` against e45 (`obench_v3e45nowera5_*`) and ft0808
   (`obench_ft0808nowera5_*`); `smoke_table.py` for the readout; a small
   `seaft_score.py` implementing the rule below.
3. Keep rule (both dates must pass):
   - family score: mean RMSE ratio to e45 over zos/uo/vo × leads 1/5/10 **< 1.00**;
   - rest: mean RMSE ratio to e45 over all other exact-depth rows **≤ 1.01**;
   - bias: lead-1 SST bias not colder than e45's by more than **0.05 °C**.
4. Best passing epoch (lowest family score) → Task 12 pipeline: 52-date inference,
   14-level export, OceanBench 0.4.0 eval (`challenger_v3seaft.py`), scoreboard update.
   If no epoch passes, e45 goes to Task 12 unchanged.

Every decision is recorded in `.superpowers/sdd/2026-08-19-v3-beat-glonet/progress.md`.

## 8. Verification before committing compute

Test link log must show: the freeze set of Section 3; `'num_steps': 10`; epoch counter
starting at `000`; the reader echoing a 33-entry `target_channel_weights` vector with
`zos` = 6.91 and `uo/vo` = 3.0; `start_date 2020-01-01`, `end_date 2023-12-31`;
validation dates 2019; `lr_max` 2e-5; a first terminal loss line. Any mismatch → cancel,
fix config, resubmit (no chain until clean).

## 9. Out of scope

- Geostrophic-consistency auxiliary loss (∇zos ↔ surface currents): follow-up only if
  this finetune plateaus on the family.
- Mixed-layer depth and the 32-level export test: separate task.
- Any change to the frozen dynamics (encoders, forecast engine) or to the base run.
- Output-side corrections of any kind (standing rule).

## 10. Risks

- **Aggregation drift**: even with all channels in the loss, unfreezing ~270 M parameters
  for 4 epochs can move un-targeted channels; the ≤ 1.01 "rest" gate catches it.
- **Recency overfit**: 2020–2023 is ~1460 days for 16 k samples (~11 passes per day);
  the per-epoch smoke on 2024 dates is the guard, and 4 epochs is the cap.
- **Weight scale**: a 6.9× zos weight with MSE in normalized units changes the gradient
  balance substantially; if epoch 0 regresses the rest beyond the gate, halve the gain
  (40 → 20) and rerun — one retry, then stop.
