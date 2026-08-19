# v3 GLORYS model: beat GLONET on OceanBench (design)

Date: 2026-08-19. Owner: Kacper Nowak. Status: draft for review.
Slurm account for ALL compute in this plan: `e-ext-2025e01-128` (not hclimrep).

## Design principle

Every component below is derived from this project's own measurements (bias
decompositions, ablation smokes, drift signatures) and plays to this model's
demonstrated strengths — deep/vertical structure, salinity, flat error growth.
Competitor scores define the scoreboard target only, never the mechanism; we do
not copy other models' recipes.

## Goal and success criteria

Outrank GLONET on a clear majority of the 20 OceanBench 0.4.0 scoreboard rows with
raw model outputs (no output-side corrections — no persistence blending, no bias
correction).

The v1/v2 lines already beat GLONET on ~13 rows (SSH, SSS, surface currents, deep
temps, geostrophic, Class-4 currents/SLA, Lagrangian) and v2 smokes flip
`temp_50m`/`temp_100m`. The contested rows this design targets:

- `sst_surface`, `class4_sst`, `sst_glo12` at short leads (GLONET LD1 ≈ 0.67 °C,
  our best ≈ 0.74)
- `mld` (GLONET ≈ 45-55 m, ours ≈ 62-67 m)

**Success**: ≥ 12/20 rows better than GLONET at the majority of leads, including
`temp_50m`, `temp_100m`, and `mld`. **Stretch**: `sst_surface` at any lead.

Feasibility arithmetic for LD1 SST: our 0.74² = 0.55 MSE decomposes into a
measured ≈ −0.25 °C mean bias (0.06 MSE) + variance. Removing the bias alone gives
√(0.55−0.06) ≈ 0.70; the remaining 0.70→0.67 must come from variance reduction
(capacity concentration + recency). Tight but reachable; the bias term is the
dependable part, which is why recency handling is a core component, not an option.

## Diagnosed failure modes this design must not repeat

(All measured in this project — see SDD ledger.)

1. **Climatology drift**: every long finetune at constant LR ≈ 1.4e-4 improved its
   training objective while deepening the 2024 cold/low bias (base run after
   epoch 17; uf0810; ft0818). Training loss is anti-correlated with the score once
   this mode starts. → LR decay is mandatory; checkpoint selection never uses
   training loss.
2. **Frozen-core finetunes cannot reshape the surface** (sfc0810 null result);
   **unfrozen surface up-weighting trades broad degradation for an eroding LD1
   gain** (uf0810). → No loss-weighting tricks in this design; capacity is
   allocated via the channel set instead.
3. **Wrong-regime validation** (2025-26 window without ATMO coverage) masked real
   degradation. → Validation and selection metrics defined below, all inside
   forcing coverage.

## Design

### Phase 0 — bank the current state (before any new training)

Run the full 52-date OceanBench eval of **ft0808**
(`glory_finetuning_20260808_191038`, best v2-line checkpoint by smoke):
parameterize `run_oceanbench_inference.sh` for the v2 stack (it hardcodes
`streams_directory=glorys_eval` and `output.streams=[GLORYS]`; needs
`--streams-dir` and `--output-stream` flags plus the `--eval-config`
`config_forecasting_glorys_obench_v2.yml` — and the config-merge trap documented
there means the driver must STOP passing streams_directory via `--options` when
the eval config already carries it), export to a challenger store, offline 0.4.0
eval, scoreboard update. Output: per-row baseline vs GLONET that phase 1 must
move.

### Phase 1 — fresh pretraining run ("v3")

**Channel design (asymmetric — the core idea).** Source stays wide; targets are
concentrated on scored quantities.

- ATMO source (6): `2t, 2d, 10u, 10v, ssrd, strd` (radiation from step 0).
- GLORYS2 source (133): unchanged full v2 set incl. ice (IC information is free —
  the stats-override null result showed the model leans on ocean inputs; deep
  velocities and ice inform the latent without costing target capacity).
- GLORYS2 targets (33 = 1 + 14×2 + 2×2):
  - `zos`
  - `thetao` + `so` at 14 levels:
    0.494025, 9.573, 15.8101, 25.2114, 34.4342, 47.3737, 65.8073, 92.3261,
    130.666, 155.851, 222.475, 318.127, 453.938, 541.089 m
    (all six scored depths exactly; a ~10 m level for the MLD density reference;
    contiguous coverage of the 10-300 m mixed-layer range for the MLD profile;
    salinity carried at the same levels because MLD is density-based)
  - `uo`, `vo` at 0.494025 and 15.8101 m only (the only scored velocities:
    surface rows, Class-4 15 m, drifters drogued at 15 m). v2 spent 60/129
    target channels on unscored velocities; that capacity moves to temperature.
- Ice stays source-only (as in v2).

**Training schedule.**

- Data: 1993-01-01 → **2023-06-30** (2023-H2 held out as the selection set — the
  most recent data the benchmark year can legitimately be proxied by).
- Validation config: 2022 (inside forcing coverage, disjoint from training).
- **Selection metric**: smoke-eval forecast RMSE (SST/SSH/47 m/92 m, leads 1/5/10)
  on 2023-H2 Wednesdays vs GLORYS — never training or validation loss.
- LR: warmup then **cosine decay to ~1e-6** over the full run (kills the drift
  mode). lr_max as in v2 (5e-5 nominal).
- Curriculum (chained configs, phases nested at the end of the run):
  epochs 0-48: full record, 4-step rollouts; epochs 48-64: sampling window
  narrowed to 2010→2023-06 (recency, addresses the warm-2024 bias at the
  training-distribution level); of those, epochs 56-64 additionally switch to
  10-step rollouts (ft0818 confirmed 10-step training helps lead-10
  consistency; keeping the phase short, recent, and LR-decayed avoids the
  drift cost it paid).
- Epochs/samples: 64 × 4096 like v2 (subject to queue reality); checkpoint every
  64 steps.
- **Drift watch**: smoke eval (2 dates) every ~4 epochs; abort/adjust criteria:
  SST bias deepening ≥ 0.1 °C over two consecutive smokes.
- Architecture: v2 sizing by default. OPEN QUESTION: scale up
  (`ae_local_dim_embed` 2048-class, per the eerie config) — larger model may lift
  all rows but changes cost and queue profile; decide at review.

**What is deliberately NOT in this design**: per-step atmospheric forcing during
rollout (architectural, out of budget); loss weighting (failed twice); output
corrections (ruled out); benchmark-year or 2-year-only training (leakage /
specialization).

### Phase 2 — evaluate and publish

Selection smokes on 2023-H2 → pick checkpoint → **aggregation-engine finetune**:
a short frozen-core finetune with

    freeze_modules: ".*global.*|.*local.*|.*adapter.*|.*q_cells.*|.*forecast_engine.*|.*latent.*|.*GLORYS.*"

i.e. the ATMO embedding and the aggregation engine stay trainable (Kacper:
aggregation-engine finetuning has proven to help substantially on other models
in this family). Same guardrails as everything else: LR-decayed, selected by
2023-H2 smoke skill with the bias-drift watch — the ft0818 lesson (long grinds
at flat LR reverse their gains) applies doubly here.
Then: full 52-date eval → export → offline 0.4.0 eval →
`build_scoreboard_data.py` → republish scoreboard artifact. Row-by-row
comparison vs GLONET closes the loop against the success criteria.

## Error handling / operational rules

- All sbatch jobs: `--account=e-ext-2025e01-128`.
- Compute nodes have no egress; heavy CPU work via CPU-node sbatch (standing
  rules).
- Same-run-id inference reruns require clearing `models/<run_id>` +
  `results/<run_id>` (stale-json trap).
- Fresh-run architecture = new channel counts → nothing loads from v2
  checkpoints; do not attempt warm starts.
- Eval side is ready as-is: v3 parquets carry all source channels; the eval
  stream configs will mirror the 45-target list (eval `target:` must match the
  trained pred head exactly — the 129-vs-133 lesson).

## Testing

- Config-level: reader instantiation asserts 45 targets / 133 sources and exact
  channel order vs the run json after launch.
- Epoch-cadence smokes as above (also catch any repeat of the five eval-stack
  traps burned down in the v2 smoke debugging).
- Phase-2 full eval is itself the acceptance test against the success criteria.

## Rough cost

Phase 0: ~5 h GPU inference + 1 h eval + export. Phase 1: v2-scale pretraining
(~10-14 chained 12 h GPU jobs). Phase 2: ~6 h. All on `e-ext-2025e01-128`.
