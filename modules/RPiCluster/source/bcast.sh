#!/bin/bash
#
# This script helps me send commands to all nodes in the system
DHOSTS="node3 node2 node1 head";

for DHOST in $DHOSTS;
do
	echo "Sending '"$@"' to $DHOST"; 
	ssh $DHOST "$@";
done
