Typical ways to run your MPI jobs on a cluster
----------------------------------------------

There are two variations of the MPI (Message Passing Interface) standard that
are typically installed on most clusters:

- MPICH
- OpenMPI

Though developed to match the same standard, they are slightly different.
This means that there will be some variation in just exactly how to run
your code on your cluster (or your multicore machine).
This document provides some basic instructions for unix machines.

You should note that you will need information specific to your cluster-
a file that lists the names of the nodes in your particular cluster, and that its format
will be different depending on whether you have MPICH or OpenMPI installed.
You can also run MPI on a multicore machine; in this case you will not need
the aforementioned file of nodes, since you are not running on a cluster.

Basic command for running MPI programs : multicore system
----------------------------------------------------------

The program to run MPI programs is called either `mpirun` or `mpiexec`.
On most installations, these two programs are the same- one is an alias to the other.
We will use `mpirun` in our examples below.

On a *multicore machine*, you can run `your_program`, an executable file created from
the mpicc compiler, as follows:

::

    mpirun -np <number of processes> ./your_program


Running MPI on a cluster of computers
--------------------------------------

Running MPI jobs on a cluster of computers requires an additional command-line flag
so that you can specify the names of the nodes to run the processes on. The format
for running your MPI executable called `your_program` becomes:

::

    mpirun -np <number of processes> -hostfile <path to file of hosts> ./your_program

The hostfile: MPICH
********************

If you have MPICH installed, the format of the hostfile listing your nodes
most commonly will look like this, with simply the names of the machines
in your cluster that you want to use:

::

    head
    node1
    node2
    node3

Where each line contains the name of a machine in your cluster.
In this example were are assuming we have a cluster with a machine called head, and three more
called node1, node2, and node3.

Using a file like this with the -hostfile option, mpirun will start processes
in a round-robin fashion, one
process per node, using each node once, then starting again, until a total of the
number of processes given with the -np flag have been started.

The hostfile: OpenMPI
**********************
