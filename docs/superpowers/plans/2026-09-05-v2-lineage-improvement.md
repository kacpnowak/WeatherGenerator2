# v2-Lineage Improvement Implementation Plan

> Executed directly by the controller at Kacper's request (2026-09-05); checkboxes track progress.
> Spec: `docs/superpowers/specs/2026-09-05-v2-lineage-improvement-design.md`.

**Goal:** a v2-lineage WG scoreboard entry that wins mld and ≥18/20 rows vs GLONET, without regressing ft0808.

## Global constraints
Account `e-ext-2025e01-128`; ≤ 12 h per job; ERA5 IC-window forcing (`glorys_v2_eval` streams); exports/compares on
CPU nodes; no output corrections; `rm -rf` of a run's model/results dirs before any same-run-id resubmission.

### Task B — export-axis experiment (ft0808 forecasts, no inference)
- [x] B1 export (9 reference levels) → eval: jobs 1672537 → 1672539; challenger `challenger_ft0808_B1.py`, store `wg_ft0808_B1.zarr`
- [x] B2 export (13 levels) → `nan_ts_levels.py` (T/S NaN at 9.573/15.8101/25.2114/34.4342) → eval: 1672540 → 1672541 → 1672542
- [x] Readout (2026-09-05): mld 70.2->48.5/48.0/56.6 m (B1=B2), class4 u/v better in B1, B2's class4_sst is a NaN-plane artefact -> **policy B1**. `build_scoreboard_data.py` on both reports (named `challenger_ft0808_B1_global_report.ipynb` / `_B2_`), rows vs GLONET, mld, class4 rows → **policy** (B1 on tie). Ledger.

### Task A — ft0818 honest evaluation
- [x] Inference 2×26 dates: jobs 1672543, 1672544 (`obench_ft0818era_`, obench_v2 config, GLORYS2, `--mini-epoch -1`)
- [x] Export on the B1 axis → eval: 1672545 → 1672546 (`challenger_ft0818.py`, `wg_ft0818_B1.zarr`)
- [x] Policy is B1: no re-export needed. If the policy is B2: re-export with the B2 levels + `nan_ts_levels.py`, re-eval. Readout + scoreboard row `ft0818`.

### Task C — ferec finetune
- [x] `config/config_glorys_v2_ferec.yml`, `config/streams/glorys_v2_ferec/` (weight-free), `scripts/ferec_score.py` + `tests/test_ferec_score.py`, `eval_output/ferec_epoch_smoke.sh`, `eval_output/ferec_sequencer.sh`
- [ ] Test link (first link carries the config): verify `freeze_weights:` set (forecast_engine NOT listed; aggregation, local/global, StreamEmbedder_*, decoder blocks listed), trainable 30–60%, `num_steps 10`, epochs 8, dates 2020→2023 / val 2019, `L: 5.0`, `lr_max 2e-05`, first line `0: 000 : 00010/00180`, no traceback. Then link 2 as pure resume.
- [ ] Smokes at epochs 1/3/5/7 (sequencer; compare jobs also depend on A's inference jobs so the ft0818 baselines exist). Keep rule per `ferec_score.py`. Ledger each.
- [ ] Winner → full 52-date eval on the policy axis → scoreboard (`WG ft0818` / `WG ferec` rows), medal table, verdict.
