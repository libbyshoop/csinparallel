#!/bin/bash
#
# This script helps me send commands to all nodes in the system
DHOSTS="tegra6 tegra5 tegra4 tegra3 tegra2";

for DHOST in $DHOSTS;
do
    echo "Sending '"$@"' to $DHOST";
    ssh $DHOST "$@";
done
