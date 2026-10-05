#!/usr/bin/expect -f
set timeout -1
spawn bash -c "rsync -avz a270225@levante.dkrz.de:/work/ab0995/a270088/Kacper/weathergenertor/AWICM3/{atmos_all,ocean_elem,ocean_node} /e/data1/climateai/hclimrep/data/ifs-fesom/"
expect {
    "*yes/no*" { send "yes\r"; exp_continue }
    "*assword:*" { send "rYmF7iYYgz5sztv\r" }
}
expect eof
