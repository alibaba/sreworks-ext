#!/bin/bash
last_load=$1
active=$2
exp=1884
last_load_l=$(awk 'BEGIN{print int(2048*'$last_load')}')
active_l=$((2048*active))
load=$((last_load_l*exp))
load=$((load+active_l*$((2048-exp))))
load=$((load+1024))
load=$((load/2048))
load=$((load+10))
echo $(awk 'BEGIN{printf "%.1f",'$load'/2048}')
