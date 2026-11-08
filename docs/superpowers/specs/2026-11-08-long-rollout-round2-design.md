# Long-rollout round 2: design

The user called round 1 a success and asked to continue improving the model without further
questions, so every choice below is my call.

## What round 1 taught

All seven 9-year rollouts (2201-2209) were stable. Years 2-9 SST RMSE: A2 0.464, C2 0.470,
B 0.474, C 0.485, A 0.488, D 0.505, fvnn003 0.526 °C (climatology 0.70-0.77).

1. **More training helped most.** A second stage-3 cycle (C2, A2) beat the first on every
   measure, and removed A's deep-salinity error (1400-2000 m at ~3x climatology).
2. **The 15-day validation loss does not track long-range skill.** A2's final validation loss
   (U 0.0568) equals A's (0.0567), yet its 9-year rollout is clearly better. Rollouts are the
   only measure that counts.
3. **Long spin-up (B, prefix 0-365) holds the state best** (best TEMP 0.98 and SALT 1.06
   climatology ratios, flattest SST curve) **but is the worst at short lead** (SST 0.493 °C at
   day 1). It rarely trains from an analysis: a prefix-0 step is 1 in 366, and the encoder
   only gets a gradient on those steps.
4. **The mean-error loss (D) made things worse** and is the only drifting run. Dropped.
5. **Dropout off is harmless**: A2 is marginally the best model. Kept.
6. **One start date cannot resolve differences of ~0.01 °C.**

## Goal

A model that keeps B's long-range stability and A2's short-range skill, judged on two start
dates.

Non-goals: architecture or data changes; beating climatology on every channel.
Anti-goal: a model that looks better in 2201 only because of the one start date.

## Runs

All fork from A2 (`fvdrop02` chkpt00010) with `general.istep=0`: a fresh optimizer and lr
cycle, the stage-3 recipe (10 mini-epochs x 2048 samples), latent noise and dropout off as in
A2. Equal training, so differences come from the spin-up alone.

| id | prefix_steps | prefix-0 share | tests |
|---|---|---|---|
| `fvr2ctl` | [0, 90] | 1/91 | a third cycle: does more training still help? (control) |
| `fvr2spn` | [0, 365] | 1/366 | B's spin-up on the best model |
| `fvr2mix` | [0, 365] | 25 % | spin-up plus guaranteed steps from the analysis: fixes B's short lead, trains the encoder |

`fvr2mix` needs a new option, `forecast.prefix_zero_fraction` (default 0): the share of
training steps drawn with no spin-up; the others draw from the range as before. Drawn from the
same rank-independent seed, so all ranks still agree. With the default the draws are
unchanged, so existing runs and their resume replay are untouched.

## Evaluation

- 9-year rollouts from **2201-01-01** (as round 1) and from **2001-01-01** (the 2000-2009
  held-out decade; 3285 steps end 2009-12-30) for the three new runs and for A2, C2, B and
  fvnn003, so round 1's ranking also gets a second sample.
- Scored with paper_eval. The 2001 rollouts are scored against a 2000-2009 climatology built
  from the JUPITER archive (one `FESOM_NODE_2000s` and one `FESOM_ELEM_2000s` source holding
  every channel), since the 2180-2209 one is two centuries away.
- Stability: all outputs finite at every lead and SST RMSE < 3 °C, from the score files.
- Ranking: mean over both start dates of years 2-9 SST RMSE and the per-stream climatology
  ratios; a difference counts only if both start dates agree on its sign.

## Builder's calls

- The rollout script takes the end date from `WG_ROLLOUT_END_DATE` (default 2210-01-02).
- 25 % for the prefix-0 share: enough to train the encoder (~640 steps of 2560 against ~7)
  while keeping three quarters of the steps on spun-up states.
- Run ids, link counts and scoring jobs as in round 1.
