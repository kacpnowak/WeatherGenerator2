#!/bin/bash
#SBATCH --job-name=rsync_levante
#SBATCH --output=/e/scratch/hclimrep/nowak2/WeatherGenerator2/rsync_levante_%j.out
#SBATCH --error=/e/scratch/hclimrep/nowak2/WeatherGenerator2/rsync_levante_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G

DEST="/e/data1/climateai/hclimrep/data/ifs-fesom/"
SRC_BASE="a270225@levante.dkrz.de:/work/ab0995/a270088/Kacper/weathergenertor/AWICM3"

# Function to handle password injection for a single directory
run_sync() {
    DIR=$1
    echo "Starting sync for $DIR..."
    expect -c "
        set timeout -1
        spawn bash -c \"rsync -avz ${SRC_BASE}/${DIR} ${DEST}\"
        expect {
            \"*yes/no*\" { send \"yes\r\"; exp_continue }
            \"*assword:*\" { send \"rYmF7iYYgz5sztv\r\" }
        }
        expect eof
    "
    echo "Finished sync for $DIR."
}

# Launch the three directories in parallel background jobs
run_sync "atmos_all" &
run_sync "ocean_elem" &
run_sync "ocean_node" &

# Wait for all three parallel jobs to complete
wait
echo "All transfers completed successfully!"
