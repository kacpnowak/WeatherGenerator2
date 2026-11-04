#!/bin/bash
# Request-9 experiments of the 2209 review: branch rollouts from the saved latent states of the
# deterministic rerun fvnnre (= fvnn10y) before the Aug-Sep 2209 step and change one thing.
# Waits until the needed state exists, branches it (branch_from_state.py) and submits
# sbatch_instab_rollout.sh to step $END (default 3600 = 2209-11-10), TEMP/SALT/SURFACE.
#
#   bash scripts/instab/launch_trigger_experiments.sh [names...]   # default: all
# Run ids fvtr_<name>. C (forcing_split constant) from sbatch_forcing_decomp.sh for fvnn003.
cd /work/ab0995/a270225/WeatherGenerator2
SRC=${SRC:-fvnnre}
END=${END:-3600}
C=${C:-/home/a/a270225/claude_api_tools/fable_diag/D1/data/C_fvnn003_2209-01-01.pt}
PY="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src:packages/common/src .venv/bin/python"
F="test_config.forecast"
SPLIT="{\"const_path\": \"$C\", \"alpha_const\": 1.0, \"alpha_anom\": 0.0}"

# name | start step | branch_from_state args | --options overrides
EXPS=(
  "fp32|3450||with_mixed_precision=False attention_dtype=fp32"
  "fs08|3450|--forcing-scale 0.8|$F.forcing_scale=0.8"
  "fs12|3450|--forcing-scale 1.2|$F.forcing_scale=1.2"
  "nz3s0|3450|--fe-noise {\"mode\":\"token_rms\",\"std\":1e-3,\"dropout\":false,\"seed\":0}|$F.fe_noise.mode=token_rms $F.fe_noise.std=0.001 $F.fe_noise.dropout=false $F.fe_noise.seed=0"
  "nz3s1|3450|--fe-noise {\"mode\":\"token_rms\",\"std\":1e-3,\"dropout\":false,\"seed\":1}|$F.fe_noise.mode=token_rms $F.fe_noise.std=0.001 $F.fe_noise.dropout=false $F.fe_noise.seed=1"
  "nz2s0|3450|--fe-noise {\"mode\":\"token_rms\",\"std\":1e-2,\"dropout\":false,\"seed\":0}|$F.fe_noise.mode=token_rms $F.fe_noise.std=0.01 $F.fe_noise.dropout=false $F.fe_noise.seed=0"
  "anom0|3450|--forcing-split SPLIT|$F.forcing_split.const_path=$C $F.forcing_split.alpha_const=1.0 $F.forcing_split.alpha_anom=0.0"
  "f2208|3450||streams.IFS_ATMO.window_shift=-8754h"
  "anom0a|3390|--forcing-split SPLIT|$F.forcing_split.const_path=$C $F.forcing_split.alpha_const=1.0 $F.forcing_split.alpha_anom=0.0"
  "f2208a|3390||streams.IFS_ATMO.window_shift=-8754h"
  # same as anom0a with the constant of another year: run with C=<.../C_fvnn003_2208-01-01.pt>
  "anom08a|3390|--forcing-split SPLIT|$F.forcing_split.const_path=$C $F.forcing_split.alpha_const=1.0 $F.forcing_split.alpha_anom=0.0"
)

want=" $* "
for e in "${EXPS[@]}"; do
  IFS='|' read -r name step bargs opts <<< "$e"
  [ $# -gt 0 ] && [[ "$want" != *" $name "* ]] && continue
  run=fvtr_$name
  if [ -d results/$run ]; then echo "$run exists, skipped"; continue; fi
  # wait until the source run has passed the step (reads only its current state)
  until [ "$($PY -c "import torch;print(torch.load('results/$SRC/rollout_state.pt',map_location='cpu',weights_only=False)['done_steps'])" 2>/dev/null || echo 0)" -ge "$step" ]; do
    sleep 600
  done
  sleep 30  # let the keep-states watcher copy the state
  if [ "$bargs" = "--forcing-split SPLIT" ]; then
    $PY scripts/instab/branch_from_state.py $SRC $step $run --forcing-split "$SPLIT" || exit 1
  elif [ -n "$bargs" ]; then
    read -r flag val <<< "$bargs"
    $PY scripts/instab/branch_from_state.py $SRC $step $run $flag "$val" || exit 1
  else
    $PY scripts/instab/branch_from_state.py $SRC $step $run || exit 1
  fi
  # the GPU account allows at most 5 running jobs; queueing more is fine
  WG_START=2200-01-01T00:00 WG_END=2210-01-02T00:00 WG_STREAMS=TEMP,SALT,SURFACE WG_EXTRA_OPTS="$opts" \
    sbatch --parsable --job-name=$run -A ab0995_gpu --time=04:00:00 \
    scripts/instab/sbatch_instab_rollout.sh fvnn003 10 $run $END
done
