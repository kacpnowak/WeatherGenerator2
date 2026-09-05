# v2-lineage improvement — design

Date: 2026-09-05. Status: approved in brainstorming (Kacper), executed directly at his request.
Context: v3 is parked at e45 (15/20 vs GLONET); the seaft finetune was a negative result
(`2026-09-03-v3-sealevel-finetune-design.md`, Result section). ft0808 (v2 base + forecast-engine
finetune on the 10-step task) is the best WG entry at 17/20; it loses mld, class4_sla and ssh_glo12.

## 1. Objective and success
Produce the best WeatherGenerator scoreboard entry from the v2 lineage: **win mld** (the criterion's last
required miss) and add rows vs GLONET without regressing ft0808's 17/20. Success = a WG entry at ≥18/20 vs
GLONET including mld; secondary = every row at least as good as ft0808. The submission entry is the best
candidate on the full 52-date evaluation; the scoreboard keeps ft0808 alongside it. No output-side corrections.

## 2. Component B — export-axis policy (no training)
OceanBench's mixed-layer depth is the depth of the *first export level* at which the 0.03 kg/m³ density
threshold (relative to the first level) is crossed — no interpolation — and the reference is quantised to its
own axis (0.494, 47.37, 92.33, 155.85, 222.48, 318.13, 380.21, 453.94, 541.09 m below the 600 m cap). Our 14-level
export adds 9.6/15.8/25.2/34.4 m, so shallow mixed layers are reported ~20 m shallower than the reference can.
Re-export ft0808's existing 52 forecasts (`obench_ft0808era_*`) two ways and evaluate both:
- **B1** — the nine reference levels ≤ 541 m for all variables.
- **B2** — B1 for thetao/so, plus 9.573/15.8101/25.2114/34.4342 m kept only for uo/vo (T/S planes at those
  levels NaN-filled after export by `eval_output/nan_ts_levels.py`) to protect the Class-4 15 m currents.
Readout: mld, Class-4 T bins and currents, temperature rows, rows won vs GLONET. **Policy** = the variant with
more rows won vs GLONET; tie → B1. The policy applies to A and C.

## 3. Component A — ft0818 honest evaluation
`glory_finetuning_20260818_105654` (= ft0808 + 8-epoch aggregation-only finetune, 10-step, never evaluated
under ERA5 forcing): 52-date inference (two 12 h jobs, `obench_ft0818era_<date>`), export on the B1 axis
(re-export if B2 wins), OceanBench 0.4.0, scoreboard. Its per-date outputs are the smoke baselines for §4.

## 4. Component C — forecast-engine recency finetune (`ferec`)
Fork ft0818's latest checkpoint into `glorys_v2_ferec_<date>` with `config/config_glorys_v2_ferec.yml`:
- trainable = **forecast engine only**: `freeze_modules: ".*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*aggregation.*|.*latent.*|.*GLORYS.*|.*ATMO.*"`
  (ft0808's set plus ATMO); verified on the test link via `freeze_weights:` names and the trainable count
  (expected 30–60% of parameters);
- objective as ft0808/ft0818 (unweighted MSE; streams `config/streams/glorys_v2_ferec/` = `glorys_v2` without
  the leftover surface `target_channel_weights`) except `dynamic_loss L: 5.0` (was 20);
- window 2020-01-01 → 2023-12-31 (180-step epochs on 8 ranks), 10-step, `pushforward: False`, 8 epochs
  (~11.5k samples), `lr_max 2e-5` (×√8 at runtime), warmup 128, cooldown 256, cosine; EMA halflife 700 samples;
  validation 2019; `istep: 0` on the first link only; per-epoch checkpoints; two 12 h links.
- selection smokes at epochs 1, 3, 5, 7 on 2024-01-03 and 2024-07-03 vs the ft0818 baselines from §3
  (`scripts/ferec_score.py`): keep an epoch only if, on both dates, the mean RMSE ratio over all exact-depth
  rows is < 1.00, the zos ratio ≤ 1.02, and the lead-1 SST bias is not colder by > 0.05 °C.

## 5. Hand-off
Best candidate (ft0818, or the best passing ferec epoch) → full evaluation on the §2 axis → scoreboard, medal
table, verdict against the criterion. Ledger: `.superpowers/sdd/2026-08-19-v3-beat-glonet/progress.md`.

## 6. Budget and mechanics
B ≈ 3 h (CPU exports + 2 evals), A ≈ 8 h, C ≈ 15 h training + smokes. Everything is slurm-chained; the only
session-bound piece is the one-shot smoke sequencer (`eval_output/ferec_sequencer.sh`, re-armable).
Account `e-ext-2025e01-128`; ERA5 IC-window forcing everywhere; no heavy work on login nodes.

## 7. Out of scope
v3 (parked), output ensembling or any output correction, new datasets, decoder/aggregation finetunes.

## 8. Risks
Sparser T/S levels may cost the Class-4 SST bins (measured in B, decides the policy). A dynamics finetune on a
short window can reintroduce rollout instability or a mean-state shift (the 10-step objective and the per-epoch
gates guard). The short EMA makes checkpoints noisier (per-epoch smokes on two dates absorb that).
