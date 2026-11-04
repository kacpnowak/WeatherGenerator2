#!/bin/bash -x
#SBATCH --account=ab0995_gpu
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3800M
#SBATCH --gres=gpu:a100_80:1
#SBATCH --chdir=/work/ab0995/a270225/WeatherGenerator2
#SBATCH --output=logs/keep-states-%x.%j.out
#SBATCH --error=logs/keep-states-%x.%j.err

# sbatch_instab_rollout.sh with every saved rollout state kept: the trainer overwrites
# results/RUN_ID/rollout_state.pt after each chunk; a watcher copies each new version to
# results/RUN_ID/states/state_<done_steps-order>.pt so later runs can start from any chunk
# boundary. Same arguments and environment as sbatch_instab_rollout.sh.

RUN_ID=${3:?usage: sbatch_rollout_keep_states.sh FROM_RUN FROM_EPOCH RUN_ID NUM_STEPS}
STATE=results/$RUN_ID/rollout_state.pt
mkdir -p results/$RUN_ID/states
watch_states() {
  last=""
  while true; do
    if [ -f "$STATE" ]; then
      m=$(stat -c %Y.%s "$STATE")
      if [ "$m" != "$last" ]; then
        cp "$STATE" "results/$RUN_ID/states/state_$(date +%s%N).pt" && last=$m
      fi
    fi
    sleep 3
  done
}
watch_states &
WATCHER=$!
bash scripts/instab/sbatch_instab_rollout.sh "$@"
rc=$?
sleep 5
kill $WATCHER
exit $rc
