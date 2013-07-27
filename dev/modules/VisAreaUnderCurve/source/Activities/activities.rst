**********
Activities
**********

ACTIVITY 2 - Striping in MPI
############################

Write a loop that will result in striping among the nodes. Note: compile using: **make area.c-mpi STRIPING=1**.

Download the solution: :download:`area_MPI_striping <area_MPI_striping.tgz>`.

ACTIVITY 3 - Striping in OpenMP
###############################

Write a loop that will result in striping among the threads on the master node. 

Note: compile using:
**make area.c-openmp,** and/or **make area.c-mpi-openmp**

The outcome should match this result:

.. figure:: omp_striping.png
	:width: 400px
	:align: center
	:height: 400px
	:alt: omp-striping
	:figclass: align-center

	Screenshot taken from a machine with 4 cores.

ACTIVITY 4a - Striping in MPI+OpenMP (blocking+striping)
########################################################

Write a loop that will result in blocking among the nodes and striping among the threads within each node. 

Note: compile using **make area.c-mpi-openmp STRIPING=1**

Download the solution: :download:`area_hybrid_blocking_striping <area_hybrid_blocking_striping.tgz>`.

ACTIVITY 4b - Striping in MPI+OpenMP (striping+striping)
########################################################

Write a loop that will result in striping among the nodes and the threads within each node. 
Note: compile using **make area.c-mpi-openmp STRIPING=1**

Download the solution: :download:`area_hybrid_striping <area_hybrid_striping.tgz>`.