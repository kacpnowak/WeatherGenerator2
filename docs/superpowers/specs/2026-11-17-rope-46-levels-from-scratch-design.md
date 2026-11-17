# From-scratch ocean emulator with S2RoPE on all 46 levels: design

Status: approved by the user on 2026-11-17. Nothing is launched until the smoke test in
"Verification before launch" has passed. ECMWF `upstream/develop` is not merged until this
training has finished (user decision).

## Intent

Train one new FESOM ocean emulator from scratch on Levante that

- has spherical RoPE (real Wigner-D rotations, `rope_mode: spherical_unitary`) in every attention
  module from the first step, and
- predicts temperature, salinity and currents on all 46 depth levels with wet points (2-5825 m)
  instead of 31 (2-2035 m),

following the long-rollout recipe of `docs/fesom-vars-training-recipe.md`. Why: the earlier
attempt grafted RoPE onto the finished model `fvnn003` and fine-tuned only the forecast engine;
it stayed worse than the baseline on one-year rollouts (memory note `s2rope-experiment`). A fair
test needs RoPE in the model from the start. The deeper levels are wanted for themselves.

## Goals, non-goals, anti-goals

Goals
- A model that is stable in multi-year free rollouts and can be compared with `fvnn003` and
  `fvnn10y` on the 31 shared levels.
- Every departure from the proven recipe is listed here and was decided on purpose.
- The run uses its GPU budget well: little time lost at mini-epoch boundaries and restarts.

Non-goals
- A no-RoPE control run (the user: two trainings in parallel are not affordable). Consequence:
  a better or worse result cannot be attributed to RoPE alone, because levels and size change
  too.
- JEPA pretraining, the spherical-harmonic RoPE variant, a wider trunk, a deeper forecast
  engine, a finer latent grid.
- Fixing that failed jobs exit with code 0, or stopping cleanly before the wall-time limit.
- Extending the `paper_eval` climatology below 2035 m. It is needed before deep levels can be
  scored, not before training.

Anti-goals
- Launching a stage on a code path that has not run on GPUs in the smoke test.
- A silent change of the training task (wrong streams, noise on, replayed data).

## Constraints

- Levante: 2 nodes x 4 A100 80 GB per job (world size 8, one sample per GPU), 12 h wall time,
  account `ab0995_gpu`. GPU priority is low at the moment, so jobs wait for a long time.
- The 15-day stage of the 31-level model peaks at 62.5 GB of 80 GB GPU memory.
- `/work` is 98 % full. One checkpoint is about 14 GB (4.7 GB weights, 9.3 GB training state).
- Old lineage cost on Levante: stage 1 216 node-hours, stage 2 and each stage-3 cycle about 65.
- Commit only when the user asks.

## Approaches considered

1. **Physical recipe from scratch (chosen by the user).** Stage 1 (4-day rollouts), stage 2
   (15-day), three stage-3 cycles with a spin-up, the last with the mixed 0-365 day spin-up. The
   only path shown to hold for nine to ten years.
2. **Temporal-JEPA pretraining, then forecast fine-tune** (`config/fesom_jepa_*.yml`). Built for
   46 levels with a RoPE overlay, but its fine-tune stops at 2-step forecasts; stages 2 and 3
   would still be needed and none of it is tested for long rollouts. Rejected.
3. **RoPE in the forecast engine only.** Keeps band 10 with the old head counts, but the
   encoder would be trained without relative positions. Rejected: the point of training from
   scratch is to have RoPE everywhere.

## The model

Base: `config/config_fesom_vars.yml` (the `fvnn003` family: HEALPix level 5, latent dim 2048,
16-block forecast engine). Changes:

| What | Old | New | Why |
|---|---|---|---|
| Depth levels of TEMP, SALT, U, V | 31 | 46 | the request; 127 -> 187 channels |
| RoPE | off | `spherical_unitary`, `rope_modules: all`, bands 1..10, `pe_global_enable: True` | the request; Wigner recovered better than SH in the graft |
| Heads of the aggregation and global engines | 32 | 16 | 128-dim heads are needed for band 10 (64-dim heads stop at 7). No parameter change |
| Stream embedding width (TEMP, SALT, U, V) | 256 | 512 | embeddings (8 M) and decoders (25 M) are 3 % of 1.17 B parameters and take the 47 % more channels |
| Decoder readout layers (TEMP, SALT, U, V) | 2 | 4 | same reason |
| `fe_impute_latent_noise_std` | 1e-4 | 0 from the start | the noise caused the rollout blow-ups |
| `fe_dropout_rate` | 0.1 | 0 from the start | recipe: no train-only perturbation on the rollout path |

SURFACE and the IFS forcing stream keep their settings. The trunk (latent width, engine depths)
is unchanged.

If the wider embeddings or the deeper readout do not fit in GPU memory in the 15-day stage
(smoke test, limit 75 GB), fall back first to 3 readout layers, then to 2. The embedding width
is kept.

### RoPE frame axis

**Changed after the first approval and confirmed by the user on 2026-11-17: the geographic
axis is kept and no tilt is built.**

Each cell's rotation is R_z(phi) R_y(theta), which is singular where theta = 0 or pi. Near the
axis the frames of neighbouring cells twist against each other by up to 1 / sin(angle from the
axis) times the normal amount. The approved draft proposed moving the axis to an antipodal
pair of land points. Measured on the mesh (`smoke46/pole/pole_metrics.py`; ocean cells are the
HEALPix cells that hold mesh nodes, twist ratio is the worst-direction value at the cell
centre):

| Axis | Nearest ocean (deg) | Ocean cells with ratio > 2 | > 5 | > 10 | Largest |
|---|---|---|---|---|---|
| Geographic poles | 0.0 / 11.5 | 12.0 % | 1.1 % | 0.25 % | 39 |
| Central China / Argentina (the best land pair) | 5.0 / 5.5 | 10.7 % | 1.1 % | 0.06 % | 12 |
| Central Greenland / East Antarctica | 3.6 / 4.3 | 13.5 % | 1.6 % | 0.18 % | 15 |

No antipodal land pair is more than 5.5 degrees from the ocean, so a tilt only removes the
extreme tail (about 24 cells around the North Pole). The share of ocean cells with a ratio
above 2 stays at 11-13 % and moves from the polar seas to the Kuroshio region and the
south-west Atlantic (China / Argentina) or to the subpolar North Atlantic (Greenland).
Keeping the geographic axis also keeps the upstream RoPE unchanged, comparable with the other
RoPE runs, and aligned with the east/north frame of u and v. `pe_global` stays on and gives
every cell its absolute identity.

## Data

- Stream directory for this machine: `config/streams/fesom_all_levels_forced_levante/`, a copy
  of `fesom_all_levels_forced/` with the Levante mesh path, plus `u.yml` and `v.yml` (46 levels,
  generated from `fesom_vars/u.yml` and `v.yml`; the set in `smoke46/streams/` was verified
  against the archive and the statistics files). The mesh path must be in the stream files: on
  a fresh `train` the stream directory is reloaded after the config merge, so overlays cannot
  change streams.
- Also add `u.yml` and `v.yml` (JUPITER mesh path) to `fesom_all_levels/` and
  `fesom_all_levels_forced/`, which lack them in git. One-letter file names need `git add -f`.
- Training 2010-2199, validation 2200, as before.

## Training plan

All stages on world size 8. Run ids `rw46s1`, `rw46s2`, `rw46c1`, `rw46c2`, `rw46c3`.

| Stage | Run | Starts from | Samples | Rollout | Spin-up | lr_max |
|---|---|---|---|---|---|---|
| 1 | `rw46s1` | scratch | 32 x 8192 | 4 days | none | 5e-5 |
| 2 | `rw46s2` | `rw46s1` final | 12 x 2048 | 15 days | none | 1e-5 |
| 3, cycle 1 | `rw46c1` | `rw46s2` final | 10 x 2048 | 15 days | 0-90 days | 1e-5 |
| 3, cycle 2 | `rw46c2` | `rw46c1` final | 10 x 2048 | 15 days | 0-90 days | 1e-5 |
| 3, cycle 3 | `rw46c3` | `rw46c2` final | 10 x 2048 | 15 days | 0-365 days, 25 % of steps without | 1e-5 |

Every stage after the first starts a new learning-rate schedule and optimizer
(`general.istep=0`) and gets a new `data_loading.rng_seed`.

Departures from the documented recipe, all on purpose:

1. **New model, noise and dropout off from the start.** The recipe starts from `fvars002`,
   which was trained with both. Untested for stages 1 and 2; the upstream RoPE runs also train
   without the noise.
2. **Stage 1 as 32 x 8192 instead of 64 x 4096 samples.** Same number of steps and the same
   schedule; half as many validations, checkpoints and loader starts.
3. **Learning-rate fix.** `lr_scheduler.py` passes `final_div_factor = lr_final_decay / lr_start`
   to OneCycleLR; it must be the inverse. Today the cosine decay ends at 1.8e-7 instead of the
   configured 5.7e-6 (at world size 8), so the last fifth of every stage-3 cycle trains at
   under 1 % of the peak. All recipe runs had this bug.
4. **New data seed per stage.** A fork inherits the seed, so cycle 2 would replay cycle 1's
   start dates and spin-up lengths exactly.
5. **Surface loss weight decided later.** The recipe gives `temp_2` and `salt_2` a static weight
   of 5 in stage 3, tuned for 31 levels. The 15 new deep levels will probably sit at the top
   of the dynamic weights (cap 20x) and pull weight off the upper ocean (estimate: `salt_2`
   -30 %, levels above 2035 m about half of the SALT loss). Stage 1 runs with the dynamic
   weights only. After its first 12-hour job, read the logged weights and decide with the user
   whether stage 3 needs a larger surface weight or a lower cap for the deep levels.

Decision points with the user:

- **After the first stage-1 job:** the dynamic loss weights (departure 5), the measured step
  time and an updated cost, and the training loss against `fvars001` (the old stage 1) at the
  same step. A loss spike when the warm-up ends, as the RoPE graft showed, is reported too.
- **After stage 1, go or no-go for stage 2.** Reference: `fvars001`'s final validation losses
  (stream means TEMP 0.00095, SALT 0.00202, SURFACE 0.00172, U 0.0452, V 0.0650). Compare on
  the 31 shared levels only, channel by channel, averaged per stream. No-go if any stream is
  more than 15 % worse; then stop and discuss instead of starting stage 2. The comparison is
  approximate: the old validation used the first 256 days of 2200, the new one 64 dates
  spread over the year. The 15 % is my call.

## Making training faster

Measured on the old runs (metrics timestamps): the time between the last training step of a
mini-epoch and the first of the next is 13 % of the training time in stage 1 and 15 % in stage
3. Validation is most of it (about 11 min per mini-epoch in stage 1, 28 min in stage 3). The
rest is the data loaders: every loader start forks new worker processes, and each worker
re-initialises its readers, which takes about 4 minutes; this happens for the training and the
validation loader in every mini-epoch.

1. **Persistent readers.** `data_loading.persistent_workers: True` for the training and the
   validation loader, so the worker processes and their initialised readers live for the whole
   job. Today the sampler relies on fresh workers: each is a copy of the main-process sampler,
   taken when the loader starts, and it reads the mini-epoch number and the resume position
   from that copy and multiplies its seed in place. With persistent workers the worker must
   keep its own mini-epoch counter (advance by one per completed iteration, clear the resume
   position after the first) and derive the seed from the unmodified base seed each time.
   Requirement: with the option on, the batches of every mini-epoch, also after a resume, are
   identical to those with the option off. The default stays off.
2. **Smaller validation.** 64 samples instead of 256. So that 64 samples still cover the year
   and are the same dates in every mini-epoch, add `validation_config.sample_stride` (default
   1): with shuffle off, take every n-th start date. Use 5 (64 dates from January to
   mid-November 2200). `time_window_step` cannot be used for this: forecast offsets are
   computed by integer division with it.
3. **Half as many mini-epochs in stage 1** (departure 2 above).

Expected: boundary time falls from 13-15 % to about 3-5 % of the training time. The smoke test
measures the loader start time with and without persistent workers.

## Other code changes

- **Atomic run-config write.** `config.save` writes the JSON in place; a kill during the write
  leaves a truncated file that blocks every later job. Write to a temporary file and rename.
- **Launcher** `sbatch_fvars46_train.sh` (new; the old `sbatch_fesom_vars_train.sh` hard-codes
  the 31-level stage-1 command and stays untouched). Environment: `WG_RUN_ID` (required),
  `WG_CONFIGS` (colon-joined config files for a fresh start), `WG_FORK_FROM="<run> <mini_epoch>"`
  with `WG_OVERLAY` (colon-joined) and `WG_EXTRA_OPTIONS`. Behaviour: continue the run in place
  if its `_latest` checkpoint exists; exit at once if its final checkpoint exists; before a
  fork, check that the parent's checkpoint file exists and exit non-zero otherwise; exit
  non-zero if the log holds a traceback from the training itself. Chain jobs with `afterany`.
- **Configs**: `config/config_fvars46.yml` (base with the model changes) and
  `config/fvars46_stage1.yml`, `fvars46_stage2.yml`, `fvars46_stage3.yml`,
  `fvars46_stage3_mix.yml` (schedules; the stage-3 files carry no surface weights until that
  decision is made).

## Verification before launch

1. Unit tests: the scheduler ends the decay at `lr_final_decay` x sqrt(world size); data
   order with persistent workers equals the order without, over three mini-epochs and after a
   mid-epoch resume; `sample_stride`; the atomic write. The existing model tests keep their
   state (13 known failures in `tests/test_encoder_teacher.py`).
2. Smoke test on 2 nodes with the real configs shrunk to a few steps (extend
   `smoke46/sbatch_smoke46.sh`): train from scratch, kill mid mini-epoch as the wall-time limit
   does, resume, fork into the stage-3 settings with a spin-up, 6-day rollout. Pass means: no
   traceback; finite losses; the resume log shows the restored optimizer and dynamic-loss
   state and the right step; peak GPU memory below 75 GB in the 15-day stage; all five streams
   written with 46 channels and finite values at wet points. Record step time and loader start
   time and update the cost estimate.
3. Report the smoke results to the user before submitting stage 1.

## Operation

- Check logs, not job states (failed jobs exit 0; the virtualenv prints a harmless traceback
  at every Python start).
- Up to 64 steps are lost at each wall-time kill. Accepted.
- Disk: the project quota on `/work` had 54 TiB free on 2026-11-17 (the 98 % is the whole file
  system). Unpruned, stage 1 would write about 450 GB. Prune while a stage runs: after each
  12-hour job, delete that run's per-mini-epoch checkpoints and training-state files except
  the last two, every eighth mini-epoch, and the final one. `_latest` is never touched. When a
  stage's successor has trained its first mini-epoch, keep only the finished stage's final
  checkpoint with its training state (and every eighth of stage 1). Checkpoints stay on
  `/work`.
- Cost estimate before the smoke test: about 490 node-hours and 21 or more 12-hour jobs, less
  the boundary savings, plus whatever 46 levels, the wider embeddings and RoPE in all modules
  add per step (unknown until measured). The user approved one full model. If the smoke test
  shows a step more than 30 % slower than the 31-level model (10 s in stage 1, 38 s in stage
  3), report it and ask before submitting stage 1; the fallback is the recipe's untested
  shortcut (mixed spin-up already in cycle 2, no cycle 3).
- Queue: the account's fair-share factor is 0, so jobs in the `gpu` partition wait at
  priority 1. The full smoke test needs that partition (8 A100 80 GB, for the memory and step
  time numbers); its wait is part of the schedule. Levante also has `gpu-devel` (3 nodes with
  2 A100 40 GB each, 30 min limit). It cannot hold the full smoke test, but a cut-down one
  (world size 4, a few 4-day training steps and one validation with RoPE in all modules) may
  fit and would show early whether the encoder-side RoPE runs under FSDP. Whether the model
  fits in 40 GB is not known.

## After training (outline only)

Rollout gate (120 days), then multi-year rollouts as in the recipe, compared with `fvnn003` /
`fvnn10y` on the 31 shared levels. Scoring the deep levels needs the climatology extension.

## Left to the builder

File layout inside the configs, test names, the exact form of the worker-side mini-epoch
counter, how the launcher detects a real traceback.
