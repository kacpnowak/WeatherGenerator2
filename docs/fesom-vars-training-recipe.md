# Training the fvars ocean emulator for long rollouts

How to train the per-variable FESOM emulator (streams TEMP, SALT, SURFACE, U, V, forced by
IFS_ATMO) so that it stays stable and accurate over multi-year free rollouts. Every setting
below is backed by a 9-year rollout comparison (long-rollout rounds 1 and 2, October-November
2026). Where a step has not been tested, it says so.

The best model so far is `fvr2mix`: stable for nine years, the best day-1 SST of any model
(0.385 °C RMSE), and the best deep-ocean salinity and temperature over years 2-9.

## The recipe in short

1. Start from the stage-2 model `fvars002` with the stage-3 config.
2. Turn off both train-only perturbations: latent noise and forecast-engine dropout.
3. Train cycle 1 with the usual 0-90 day spin-up.
4. Train further cycles, each forked from the last with a fresh learning-rate schedule.
5. In the last cycle, use a 0-365 day spin-up with a quarter of the steps having none.
6. Judge models only by multi-year rollouts, never by the 15-day validation loss.

## Settings and why

| Setting | Value | Evidence |
|---|---|---|
| `fe_impute_latent_noise_std` | `0.0` (`config/fesom_vars_r1_base.yml`) | With `1e-4` the noise is about 50 % of each token's RMS; models trained with it blow up at days 50-80 of a clean rollout (fvars003). Turning it off fixed this (fvnn003). |
| `fe_dropout_rate` | `0.0` (`config/fesom_vars_r1_drop.yml`) | Harmless, and the dropout-off lineage ended up marginally best. It removes the last train-only perturbation. |
| `training_config.forecast.prefix_steps` | `[0, 90]` in cycle 1, `[0, 365]` later | Spin-up steps run without gradients, so the supervised window starts from the model's own drifted state. With 0-90 days, deep salinity can jump near year 1 and stay 2-3x worse than climatology (runs A and R2 ctl). No run trained with 0-365 days showed this. |
| `training_config.forecast.prefix_zero_fraction` | `0.25` with the 0-365 range (`config/fesom_vars_r2_mix.yml`) | With 0-365 alone, only 1 step in 366 starts from the initial state: day-1 SST is the worst of all runs and the encoder hardly trains. A quarter of steps without spin-up gives the best day-1 SST and keeps the long-range skill. |
| Training length | several stage-3 cycles | The 15-day validation loss flattens because the learning rate decays to almost zero, not because the model has converged. A second cycle improved every 9-year score; a third with the long spin-up improved them again. |
| Loss | MSE with the dynamic channel weights only | A squared mean-error term (`mean_error_sq`, run D) made every score worse and was the only run whose error grew over the nine years. Do not add it. |

Stage-3 schedule per cycle (`config/fesom_vars_stage3.yml`): 10 mini-epochs x 2048 samples,
15-day supervised rollouts, cosine learning rate with peak `1e-5`, 256 warm-up and 512
cool-down steps. At world size 8 that is 2560 steps.

## Steps

The proven path is the lineage of `fvr2mix`: three cycles. The launcher is the private repo's
`sbatch_fvars_r1_train.sh`, run from the repo root. It reads `WG_RUN_ID`, an optional
`WG_OVERLAY`, `WG_EXTRA_OPTIONS` and `WG_FORK_FROM="<run> <mini_epoch>"`. Chain 3-4 links per
cycle with `--dependency=afterany`; a link that finds the run complete exits at once.

1. **Cycle 1**, from `fvars002` (the launcher's default start), with the stage-3 config, the
   base overlay (noise off) and the dropout overlay:
   `WG_RUN_ID=<id>1 WG_OVERLAY=config/fesom_vars_r1_drop.yml`.
   Keep the 0-90 day spin-up here: `fvars002` is unstable in clean rollouts, so a year-long
   spin-up from it would train on blown-up states.
2. **Cycle 2**, a fork with a fresh optimizer and schedule:
   `WG_RUN_ID=<id>2 WG_FORK_FROM="<id>1 10" WG_EXTRA_OPTIONS="general.istep=0"`.
3. **Cycle 3**, the mixed spin-up:
   `WG_RUN_ID=<id>3 WG_FORK_FROM="<id>2 10" WG_EXTRA_OPTIONS="general.istep=0" WG_OVERLAY=config/fesom_vars_r2_mix.yml`.

`general.istep=0` on a fork starts a new learning-rate cycle with a new optimizer state, as
a new run. Without it the fork continues the parent's step counter, schedule and optimizer.

**Untested shortcut:** use `config/fesom_vars_r2_mix.yml` already in cycle 2 and stop there.
It saves one cycle but has not been compared to the three-cycle path.

## Costs

On JUPITER (2 nodes x 4 GH200, world size 8), one cycle takes 26-32 h of training, so three
12-hour links:

| Spin-up | Time per step | One cycle |
|---|---|---|
| 0-90 days | ~40 s | ~26 h |
| 0-365 days, 25 % without | ~45 s | ~29 h |
| 0-365 days | ~55 s | ~32 h |

Each checkpoint is about 4.7 GB of weights plus a 9.3 GB training-state file (optimizer
moments and dynamic-loss weights), about 150 GB per cycle.

## Resuming

Resuming continues training exactly: the sampler replays the same data order, and the
optimizer and dynamic-loss state are restored from the `*_train_state.pt` file saved with
each checkpoint. A resumed link logs `restored optimizer state` and `restored dynamic-loss
state`; if they are missing, the run started with a fresh optimizer. Parameters that have
never had a gradient (with a spin-up the encoder updates only on steps without one, and
`latent_pre_norm` never does) get AdamW's initial state on restore.

## Evaluating

1. **Stability gate:** a 120-day rollout passes if every output is finite and SST RMSE stays
   below 3 °C (`python -m paper_eval.rollout_gate`).
2. **9-year rollouts** from 2201-01-01 with the private repo's `sbatch_fvars_rollout.sh`:
   3285 daily steps, about 21 s per step, so two 11-hour links plus a spare. A link stops
   itself after the next chunk once 11 h have passed, because a kill during a chunk write
   makes the whole zip store unreadable. If a link dies between writing a chunk and saving
   `rollout_state.pt`, the next link fails with "Group already exists"; repair the store
   with `python -m weathergen.utils.trim_rollout_store results/<run_id>` and resubmit.
3. **Scoring** with `paper_eval.compute` and `paper_eval.selfcheck` (about 8.5 h per 9-year
   rollout), then compare runs with `python -m paper_eval.r9_table` and the overlay figures
   of `python -m paper_eval.r9_curves`.
4. **What to compare:** SST RMSE over years 2-9, day-1 SST, the climatology ratios of each
   stream, and above all temperature and salinity at and below 450 m, where models fail
   first. A difference smaller than about 0.01 °C in SST cannot be resolved with one start
   date.
5. **A second start date** of 2001-01-01 (in the 2000-2009 held-out decade, with
   `WG_ROLLOUT_END_DATE=2010-01-02T00:00`) only checks the surface. Below 450 m every model
   is far worse than climatology in that decade, almost equally, which says more about the
   data than the models.

Training uses 2010-2199, validation 2200; 2000-2009 and 2201-2209 are held out.

## Things that look wrong but are not

- **Every job marked FAILED.** The venv's `_virtualenv.pth` imports a missing
  `_virtualenv.py` and prints a traceback at every Python start; the launchers treat any
  traceback as a failure. Check the log for real errors before acting.
- **The ACC-decay self-check fails** for long-spin-up models and for 2001 rollouts: it
  expects the anomaly correlation to fall with lead, and here it does not always. The
  scores are fine.
- **The 15-day validation loss stays flat** through the end of a cycle while the 9-year
  rollouts keep improving. It does not track long-range skill.
