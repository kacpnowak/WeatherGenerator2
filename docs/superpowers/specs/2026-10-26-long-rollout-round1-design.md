# Long-rollout round 1: four stage-3 variants of the fvars emulator

Date: 2026-10-26. Branch base: `kacpnowak/develop/forcings` (never rebase onto develop).

## Intent

fvars003 (per-variable FESOM streams, IFS-forced, three-stage from-scratch retrain) is not
broken. The aim is to make its free-running forced rollouts better over 1 to 10 years, with
the long-term ideal of a rollout that stays stable indefinitely. Round 1 tries four changes,
each as a rerun of stage 3 with exactly one change, so each can be read against fvars003
directly.

## What the evidence says (why these four)

One-year rollouts of the older prototypes (single init 2209-01-01, paper_eval, RMSE divided by
day-of-year-climatology RMSE, mean over channels, day 363):

| run | NODE (T/S/SSH) | ELEM (u/v) |
|---|---|---|
| yr1add04 (forced, 15-day fine-tune) | 3.97 | 1.30 |
| yr1win03 (spin-up window 0-90 d) | 1.28 | 0.87 |
| yr1sal01 (wet-point norm + spin-up) | 1.17 | 0.87 |

- The spin-up window (`forecast.prefix_steps`: advance 0-90 days without gradients, then
  supervise 15 days) was by far the largest gain.
- After a year the surface sits at climatology level (yr1sal01 SST RMSE 0.70 vs 0.70); NODE
  stays above 1 because of the deep ocean (450 m: T 0.35 vs 0.26, S 0.062 vs 0.036), and the
  older runs drift cold at depth (yr1add04 -0.63 degC at 450 m, yr1sal01 -0.15).
- In `src/weathergen/model/model.py` the spin-up steps run under `torch.no_grad()` but with
  the model in train mode, so forecasting-engine dropout (`fe_dropout_rate: 0.1`) is active
  during spin-up: the spun-up state is a dropout-perturbed trajectory, not the one inference
  produces. That works against the reason the spin-up window helps.
- The current loss (`mse` + `dynamic_loss`) is per point; a small offset that is consistent
  across the ocean is cheap within any 15-day window but accumulates over years.

These older runs are prototypes (trained through 2208, different streams); they are a
look-back reference only, never a pass/fail comparison.

## Resume bug (found while reviewing this design)

Every run that is continued from a checkpoint (`train_continue`, chained jobs) trains on
replayed data. This is project-wide (same code on develop and on upstream develop), reported
upstream as ecmwf/WeatherGenerator#2953 with an open one-line fix PR #2954. Three problems:

1. **Sample replay.** `Trainer.run()` computes `mini_epoch_base` from `istep` on resume, but
   `MultiStreamDataSampler.mini_epoch` restarts at 0 in the new process. The sampler's
   per-mini-epoch RNG (`reset()`, seeded with run seed x rank x worker x `mini_epoch + 13`),
   the spin-up draws (`draw_prefix_steps`, seeded with `[seed, mini_epoch, position]`) and the
   masking all depend on it, and `train_continue` keeps the run seed, so a resumed job
   replays mini-epoch 0, 1, ... while the trainer counts from `mini_epoch_base`.
2. **Partial-epoch redo.** Resuming from a mid-epoch `_latest` save restarts the interrupted
   mini-epoch from its first step, so its first part is trained twice and the run overshoots
   the learning-rate schedule (fvars003: 2,688 steps instead of 2,560, the last 128 at lr 0).
3. **Dynamic-loss state lost.** The `dynamic_loss` channel-weight EMA (`DynamicLossEMA`) is
   not saved in checkpoints, so it restarts from ones at every resume.

Found while planning, and fixed in the same change (the user asked for all resume problems
to be addressed):

4. **Optimizer state not saved.** Checkpoints hold only the model weights; AdamW's moments
   restart from zero at every resume (`_get_full_optimizer_state_dict` exists but is unused).
5. **`_latest` one step behind, and stale after an epoch.** The mid-epoch `_latest` save runs
   before `general.istep` is incremented, so it records one step fewer than completed; and
   the end-of-epoch save writes only `chkptNNNNN`, so `_latest` can be up to 63 steps older
   than the last completed mini-epoch.

Optimizer and dynamic-loss state go in a sidecar file next to each model checkpoint and are
restored only when resuming a run (`general.istep > 0`), never when a new run starts from
another run's weights (as fvars003 started from fvars002 with a fresh optimizer).

The whole fvars lineage was affected (fvars001 resumed 8 times, fvars002 twice, fvars003
twice: only about 960 steps' worth of distinct data in stage 3).

**Decided:** fix all three on our branch before the runs start, with #2954's change
(`self.dataset.mini_epoch = mini_epoch_base` right after it is computed) as the basis for
problem 1. Requirements for the fix as a whole:

- A run split into several jobs at arbitrary points (including mid-epoch) trains on the same
  sequence of samples, spin-up lengths and forecast-step draws per optimizer step as the
  same run done in one job, and ends at exactly `num_mini_epochs x steps per mini-epoch`
  optimizer steps.
- Skipping the already-trained steps of an interrupted mini-epoch must not load their data.
- The dynamic-loss EMA is saved with the checkpoint and restored on resume; a checkpoint
  without it (all existing ones) still loads, starting the EMA from ones as today.
- New runs (`mini_epoch_base == 0`, no resume) behave exactly as now.
- Bit-identical results across a resume are not required (dropout and GPU
  non-determinism), only identical data.
- Tested: a unit-level check in the style of #2953 (stubbed batches returning sample
  indices) comparing an uninterrupted run against one resumed mid-epoch, plus a test that
  the dynamic-loss state survives a save/load round trip.
- No upstream issue or PR is opened as part of this work.

Consequence: with the fix, the new runs see different (and about 2.7x more distinct) data than
fvars003 did, so run C, not fvars003, is the baseline for A, B and D. fvars003 still gets the
9-year rollout as a reference.

## Goals

- The resume fix above, landed and tested before any training run starts.
- Four runs, each = stage 3 of fvars rerun from the same starting weights with one change:

  | run id (my call) | change vs fvars003 |
  |---|---|
  | `fvdrop01` (A) | `fe_dropout_rate: 0.0`; all other dropout rates unchanged |
  | `fvspin01` (B) | `forecast.prefix_steps: [0, 365]` instead of `[0, 90]` |
  | `fvctrl01` (C) | none: fvars003's stage-3 settings rerun on JUPITER with the resume fix; the baseline for A, B, D |
  | `fvbias01` (D) | adds a mean-error loss term (below) |

- A 9-year free-running forced rollout for fvars003 and for each of A-D, scored with
  paper_eval, plus a comparison table across them.

## Non-goals

- No from-scratch retrain in this round (about 160 h on 8 GPUs). It becomes a question only if
  A wins.
- No EMA / weight averaging, no tuning sweeps, no combining changes. Winners get combined in
  round 2, chosen with the fvars003 9-year depth breakdown.
- No changes to the data split, normalisation statistics, or stream set.

## Anti-goals

- A comparison that is not like-for-like: different starting weights, schedule, sample count,
  world size, or rollout window between runs would make the round unreadable.
- Reading a difference that is not clearly larger than run-to-run noise as a win. The only
  noise estimate in this round is C against fvars003, and it is confounded (machine change,
  resume fix) and probably understates the true spread (same data seed, see below), so small
  differences are reported as inconclusive.

## Fixed setup (decided)

- Data split as fvars (`config/config_fesom_vars.yml`): train 2010-01-01..2199-12-31,
  validate 2200, never seen 2000-2009 and 2201-2209.
- Starting weights for all four: `fvars002_latest.chkpt`, the file fvars003 itself started
  from (`train_continue --from-run-id fvars002 -e -1`). Note: it is NOT identical to
  `fvars002_chkpt00012.chkpt` (saved ~1 h later, different size). Both are in
  `/e/scratch/weatherai/shared_work/models/fvars002/`.
- Stage-3 recipe = `config/fesom_vars_stage3.yml`: 10 mini-epochs x 2048 samples, lr
  1e-6 -> 1e-5 -> decay, warm-up 256, cool-down 512, 15 supervised days after a 0-90-day
  spin-up, 5x weight on temp_2 and salt_2, 256 validation samples, nothing frozen,
  `general.istep=0`. Each run is that overlay plus one small overlay of its own.
- Data seed: all four runs keep the inherited `data_loading.rng_seed` (1792056342), so they
  draw the same samples in the same order (B differs in spin-up lengths only) and A, B and D
  compare pairwise with C (my call). C therefore measures GPU non-determinism only, not
  data-order noise.
- Jobs: JUPITER caps jobs at 12 h, so each run is a chain of resumable jobs; with the resume
  fix the resume points no longer matter.
- Machine: JUPITER, 2 nodes x 4 GH200 = world size 8, matching Levante (the lr scales with
  sqrt(world size); keeping 8 keeps lr and steps per mini-epoch identical). Slurm account
  `e-ext-2025e01-128`, partition `booster`. Data at
  `/e/data1/climateai/hclimrep/data/ifs-fesom` (identical copy of the Levante archive,
  verified), mesh `/e/data1/climateai/hclimrep/data/ifs-fesom/fesom.mesh.diag.nc`.

## Evaluation (decided)

- Rollout: one free-running forced rollout from 2201-01-01 to 2209-12-30 (the last day with
  forcing; 2204 and 2208 are leap years, so about 3,286 daily steps; the builder computes the
  exact number), same mechanism as the Levante one-year script
  (`sbatch_fesom_forced_rollout.sh`: one sample, `forecast.chunk_size=30`,
  `forecast.resume=true`, `compute_loss=false`, all target points written). One year took
  ~1.5 h on one A100, so ~14 h per model: resumable chained jobs.
- Models rolled out: fvars003 (use `fvars003_chkpt00010.chkpt`, the end of stage 3; its
  `_latest` is an earlier mid-epoch save) as a reference, and the final end-of-training
  checkpoint of each of A-D.
- Scoring: paper_eval, per-variable stream layout (as `paper_eval/configs/yr1fv003.yaml`),
  climatology reference 2180-2209 as now (it includes the test years and slightly flatters
  the climatology baseline; kept for comparability, stated in the table). Report RMSE/clim
  ratios for NODE and ELEM, bias per depth (drift), and surface fields at day 365 and at
  every year end through year 9.
- Reference: the six older one-year runs, look-back only.

## Run D: mean-error loss term

For each stream, channel and supervised step, take the error (prediction - target) at the
sampled target points, average it over those points with the same location weights the MSE
uses (`cosine_latitude`), square it, and add it to the loss with weight lambda. A uniform
offset then costs extra beyond MSE; local errors are unchanged.

- Channel weighting (my call): the same per-channel weights the MSE gets, i.e. the
  dynamic-loss weights times the fixed 5x on temp_2/salt_2. The dynamic weights lean on the
  low-error deep channels, which is where the drift is.
- The new term does not feed back into the dynamic-loss weight updates; those keep tracking
  the MSE only.
- Lambda (my call): chosen so that the new term starts at roughly one tenth of the weighted
  training MSE in the first logged steps of a short probe, then fixed; no sweep.
- Off by default and enabled only through run D's overlay, so A, B and C are unaffected. New
  code in `src/weathergen/train/loss_modules/` with a unit test; the term is logged
  separately.

## Left to the builder

- How run D's term is wired into `LossPhysical` (new entry in `loss_fcts` or an option on
  `mse`), as long as the config default leaves every other run bit-identical.
- The JUPITER job scripts (training chain and rollout chain), modelled on Levante's
  `sbatch_fesom_vars_train.sh` and `sbatch_fesom_forced_rollout.sh`; whether `uv run` or the
  venv entry points are used.
- Where scoring runs: JUPITER is preferred (data, rollouts and checkpoints are here); that
  needs the newer paper_eval (Levante commit `1f8433ed`, unpushed) and the 2180-2209
  climatology zarrs (copy from Levante `paper_eval/cache_oe3i7whz/climatology/` or rebuild).
- Whether run B's longer spin-up needs a sample-count or time-limit adjustment to fit; the
  sample count and schedule must stay as stage 3.

## Prerequisites and state

- On this checkout (uncommitted): stream files, tests and `compute_wet_stats.py` repointed to
  the JUPITER mesh/data; `config/config_fesom_vars.yml`, `config/fesom_vars_stage{1,2,3}.yml`
  and `tests/test_fesom_vars_training_config.py` copied from Levante. Levante holds five
  unpushed commits (`fbf57a86`, `e9b4b90f`, `d762f3dd`, `1fa55d17`, `1f8433ed`); a
  `git diff --stat` on Levante shows they touch only configs, tests and `paper_eval/`, nothing
  under `src/`, and Levante's tracked tree is clean, so the model code here is what
  fvars001-003 trained with.
- Levante is reachable as `ssh levante` only while the user's ControlMaster connection is open
  on the JUPITER login node it was opened on.
- FESOM test suite passes on JUPITER against the JUPITER data (56 passed, 0 skipped).
- Checkpoints fvars001-003 are in `/e/scratch/weatherai/shared_work/models/`.
