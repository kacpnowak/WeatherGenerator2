#!/bin/bash
# Parallel high-speed rsync script

DEST="/e/data1/climateai/hclimrep/data/ifs-fesom/"
SRC_BASE="a270225@levante.dkrz.de:/work/ab0995/a270088/Kacper/weathergenertor/AWICM3"

# Function to handle password injection and high-speed transfer for a single directory
run_sync() {
    DIR=$1
    echo "Starting high-speed sync for $DIR..."
    expect -c "
        set timeout -1
        spawn bash -c \"rsync -av --rsh='ssh -c aes128-gcm@openssh.com' ${SRC_BASE}/${DIR} ${DEST}\"
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
