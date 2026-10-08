# Long-horizon stability training for the IFS-forced FESOM emulator

Date: 2026-10-08. Branch: `kacpnowak/develop/forcings` (experimental, never merged; useful
changes are extracted into a separate PR later).

## Goal

The model is meant to become a **forced ocean emulator**: FESOM driven by the AWI-CM3 IFS
atmosphere, running freely for decades. The first milestone is a good **one-year** forced run;
decadal runs and a held-out decade come later.

The 180-day evaluation from 2209-01-01 (f180add1 = frcadd04, forced) showed:

- the forcing works: surface currents keep an anomaly correlation above 0.5 for 180 days,
  SST and SSH errors stay far below the un-forced runs;
- the deep tracers drift: at day 180 the NODE channel-mean RMSE is 2.4x climatology (a
  realistic but unsynchronised ocean would sit near 1.4x), `salt_450` RMSE is 0.176 against
  persistence 0.039; the global-mean bias is under 2 % of the MSE, so the drift is in the
  spatial pattern, not a uniform offset;
- the model trains on 15-day rollouts only, so drift that builds up over 30-180 days is never
  penalised.

This spec covers sub-project B, **long-horizon stability training** with a truncated window,
plus sub-project A, the **one-year acceptance test** it is judged by.

## Success criteria

Measured on the one-year forced run (below), against the same test for frcadd04:

1. **Drift.** The channel-mean RMSE relative to climatology at days 180 and 363 is clearly
   below frcadd04's for both streams; for FESOM_NODE the aim is the ~1.4x floor of an
   unsynchronised but realistic ocean.
2. **Deep fields.** For `temp_450`, `salt_450` and `salt_107` the RMSE closes most of the gap
   to persistence.
3. **No short-range regression.** The 15-step validation MSE and the first 30 days of the
   one-year run are at most ~5 % worse than frcadd04.
4. **Realistic variability.** The variability ratio (below) stays within about 0.7-1.3.

## Acceptance test: one-year forced run (sub-project A)

- **Forecast.** One free-running forced forecast from 2209-01-01 for **363 daily steps**
  (through 2209-12-30; 2209-12-31 has no forcing because its 00 UTC stamp would be in 2210).
  Same inference mode as the 180-day runs: one sample, lazy targets, `chunk_size` 30,
  `resume: true`, `compute_loss: false`, output streams FESOM_NODE and FESOM_ELEM.
  `sbatch_fesom_forced_180d.sh` is generalised into one script that takes the source run,
  mini-epoch, output run id and number of steps.
- **Scores.** paper_eval with a config per run (copy of `f180add1.yaml` with `leads.max: 363`,
  `map_leads: [90, 180, 363]`), scored on the `shared` partition (1 h, 64 GB was enough for
  180 days; the one-year run needs about twice the time).
- **Diagnostics script** (new, untracked helper next to `compare_fesom12h.py`), reading the
  inference output and the paper_eval climatology:
  - **regional means against the truth** over the year for `sst`, `ssh`, `temp_107`,
    `salt_107`, `temp_450`, `salt_450`: global and three latitude bands (90S-30S, 30S-30N,
    30N-90N), area-weighted with the mesh's node areas and wet masks (as paper_eval does);
  - **variability ratio** per field and lead: area-weighted spatial standard deviation of the
    forecast anomaly (forecast minus climatology) over that of the true anomaly; below 1 means
    blurred, above 1 means spurious variability.
- **Reference runs.** The same test for frcadd04 (current best) and ykwx7bl2 (the original).

## Mechanism (sub-project B)

### Configuration

```yaml
training_config:
  forecast:
    num_steps: 15            # supervised steps, unchanged
    policy: fixed
    prefix_steps: [0, 90]    # new: unsupervised spin-up length, drawn uniformly per step
```

- `prefix_steps` is `[min, max]`, inclusive. Absent or `[0, 0]` means today's behaviour.
- It applies to the **training stage only**. Validation and test configs inherit training
  settings through `get_active_stage_config`, so the sampler must ignore `prefix_steps`
  whenever its stage is not training: validation stays a 15-step rollout comparable with
  earlier runs, and the one-year inference has no prefix.

### Sampler (`MultiStreamDataSampler`)

- **Draw.** For each training step, draw P uniformly from `prefix_steps`. The generator is
  seeded from the run seed, the mini-epoch and the step's position in the rank's iteration,
  never from the rank or the loader worker, so all GPUs use the same P for the same step.
  (The existing sampler generator is reseeded per rank and worker, so it cannot be used: the
  slowest GPU would set every step's time and the FSDP call pattern would differ per GPU.)
- **Batch.** A sample spans P + 15 forecast steps after the offset. The input side is built as
  today. Targets (target coordinates in the source samples, target values in the target
  samples) are built only for the last 15 steps, through the existing step-range builders
  (`_build_batch_skeleton` and `_materialize_target_chunk(step_range)`, written for lazy
  inference). Forcing and conditions are built for all P + 15 steps, since the model needs
  them every step. The batch records the first supervised step (offset + P).
- **Valid start dates.** Sample start indices leave room for `max(prefix_steps) + num_steps`
  forecast steps, so every window, prefix included, lies inside the training period.

### Model (`Model.forward` rollout loop)

- Steps before the batch's first supervised step run the forcing injection and the forecast
  engine under `torch.no_grad()` and are neither decoded nor given latent predictions,
  exactly like today's pushforward steps.
- The existing `pushforward` option keeps its current behaviour (all steps but the last
  without gradients); the prefix is a separate rule.
- FSDP: the forcing module is part of the root FSDP unit and the forecast-engine units are not
  resharded during the rollout, so prefix steps add no collectives.

### Loss and logging

- The loss is unchanged: `LossPhysical.compute_loss` already skips forecast steps that have no
  predictions. The prefix steps' target slots exist but stay empty.
- P is printed with each training log line and its mean over the logging window goes into the
  training metrics, so the loss can be read against the prefix length.

### Cost

- **Memory:** as today (about 72 GiB peak on an A100-80 with a trainable forecast engine), since
  prefix steps keep no activations.
- **Compute:** on average 45 extra no-gradient forecast-engine and forcing steps per training
  step (today ~23 s per step for the 15 supervised steps).
- **I/O:** forcing reads grow up to 6x (IFS `atmos_all`, 6-hourly, nine channels).
- The smoke run measures all three. If a step gets too slow, lower `max(prefix_steps)`, or add
  a curriculum that raises it over the mini-epochs (30, 60, 90).

## Experiment

1. **Smoke run.** 2 nodes, a few dozen steps, prefix `[0, 90]`, from frcadd04 mini-epoch 12.
   Checks: training runs, peak memory near today's, P varies in the log, loss finite, step time
   and forcing I/O acceptable. The measured step time sets how many mini-epochs fit in 12 h.
2. **Two 12 h runs** from frcadd04 mini-epoch 12, same recipe as frcadd03 (fresh warm-up and
   decay with `general.istep=0`, forecast engine trainable, same freezing), new run ids:
   - **window**: `prefix_steps: [0, 90]`;
   - **control**: `prefix_steps: [0, 0]`, i.e. training exactly as now, which separates the
     effect of the prefix from simply training 12 h longer.
3. **Acceptance.** One-year runs for window, control, frcadd04 and ykwx7bl2; paper_eval to day
   363 plus the diagnostics; judged against the success criteria.

## Testing

Unit tests on CPU, written before the code:

- **Sampler:** P is identical across different rank and worker seeds for the same step and lies
  within `[min, max]`; targets exist only for the 15 supervised steps; forcing exists for all
  P + 15 steps; the batch's first supervised step is offset + P; start dates leave room for
  `max(prefix_steps) + num_steps`; validation and test stages ignore `prefix_steps`.
- **Model:** prefix steps run without gradients and produce no predictions; supervised steps
  produce predictions with gradients; with P = 0 the outputs match today's behaviour exactly.
- **Loss:** prefix steps contribute nothing (regression test for the existing skip).
- **Unchanged:** the existing forcing, forecast-chunk, rollout-step and decoder tests pass.

## Risks

- Training-side helpers that walk every forecast step's targets (for example
  `target_aux.update_state_pre_backward` or the output-target builder) may not expect empty
  target slots. The smoke run covers the full training path; any such helper is fixed to skip
  empty steps.
- The data-loader throughput with 6x forcing reads is unknown until the smoke run.
- Training from drifted states may cost short-range skill; criterion 3 and the control run
  catch that.

## Out of scope

- Sub-project C (more forcing variables such as `tcc` and `ci`; a stronger injection with
  `adaln_local` or a separate learning rate for the gate) and sub-project D (decoders that
  predict daily changes): only if this does not meet the targets.
- Decadal runs and a held-out decade.
- Never use the archive's `sst` as forcing: it is the ocean model's own SST passed to the
  atmosphere, which would leak the target.
