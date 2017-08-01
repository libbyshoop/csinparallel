********************************************
Hybrid Program Structure: SPMD
********************************************

0.Program Structure Implementation Strategy: Single Program, multiple data¶
*****************************************************************************

*file: hybrid-MPI+OpenMP/00.spmd/spmd.c*

*Build inside 00.spmd directory:*
::

  make spmd

*Execute on the command line inside 00.spmd directory:*
::

  mpirun -np <number of processes> ./spmd

.. note::

  This command is going to run all processes on the machine on which you
  type it.
  See :doc:`../MessagePassing/RunningMPI` for more information about running the code
  on a cluster of machines. This note applies for all the examples below.

This is a simple example of the single program, multiple data (SPMD) pattern.
The MPI program creates the MPI execution environment, defines the size
of the MPI_COMM_WORLD and gives a unique rank to each process. The program
then enters the OpenMP threaded portion of the code. The *thread_num* and
*get_num_threads* functions from the OpenMP program are called. The MPI program
then prints the thread number, number of threads, process rank, number of processes and
hostname of each process. Lastly, the MPI execution environment is terminated
by all processes.

.. topic:: To do:

  Compile and run the program varying the number of processes. How many threads
  are working within each process? Uncomment the #pragma directive, recompile and rerun
  the program, varying the number of processes as before. Can you explain the
  behavior of the program in terms of processes and threads?

.. literalinclude:: ../patternlets/hybrid-MPI+OpenMP/00.spmd/spmd.c
    :language: c
    :linenos:


1.Program Structure Implementation Strategy: Single Program, multiple data with user-defined number of threads¶
****************************************************************************************************************

*file: hybrid-MPI+OpenMP/01.spmd2/spmd.c*

*Build inside 01.spmd2 directory:*
::

  make spmd2

*Execute on the command line inside 01.spmd2 directory:*
::

  mpirun -np <number of processes> ./spmd2 [numThreads]

Here is a second SPMD example with user-defined number of threads.
We enter the number of threads to use on the command line.
This way you can use as many threads as you would like.

.. topic:: To do:

  Compile and run the program varying the number of processes and number of threads.
  Compare the behavior of the program to the source code.

.. literalinclude:: ../patternlets/hybrid-MPI+OpenMP/01.spmd2/spmd2.c
    :language: c
    :linenos:
