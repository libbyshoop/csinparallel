**********************************************************
Program Structure: SPMD, Master Worker, and Parallel Loops
**********************************************************

This initial set of MPI pattern examples illustrates how
many distributed processing programs are **structured**.
For this examples it is useful to look at the overall
organization of the program and become comfortable with the idea
that multiple processes are all running this code simultaneously,
in no particular guaranteed order.

00. Single Program, Multiple Data
*********************************

*file: patternlets/MPI/00.spmd/spmd.c*

*Build inside 00.spmd directory:*
::

  make spmd

*Execute on the command line inside 00.spmd directory:*
::

  mpirun -np <number of processes> ./spmd

.. note::

  This command is going to run all processes on the machine on which you
  type it.
  See :doc:`RunningMPI` for more information about running the code
  on a cluster of machines. This note applies for all the examples below.

First let us illustrate the basic components of an MPI program,
which by its nature uses a single program that runs on each process.
Note what gets printed is different for each process, thus the
processes using this one single program can have different data values
for its variables.  This is why we call it single program, multiple data.

On the command line, *mpirun* tells the system to start <number of processes>
instances of the program. The call to *MPI_INIT* on line 25 tells the MPI
system to setup. This includes allocating storage for message buffers and
deciding the rank each process receives. *MPI_INIT* also defines a communicator
called *MPI_COMM_WORLD*. A communicator is a group of processes that can
communicate with each other by sending messages. The *MPI_Finalize* command
tells the MPI system that we are finished and it deallocates MPI resources.

.. topic:: To do:

  Can you determine the purpose of the *MPI_Comm_rank* function and
  *MPI_Comm_size* function? How is the communicator related to these functions?

.. literalinclude:: ../patternlets/MPI/00.spmd/spmd.c
    :language: c
    :linenos:


01. The Master-Worker Implementation Strategy Pattern
*****************************************************

*file: patternlets/MPI/01.masterWorker/masterWorker.c*

*Build inside 01.masterWorker directory:*
::

  make masterWorker

*Execute on the command line inside 01.masterWorker directory:*
::

  mpirun -np <number of processes> ./masterWorker

The master worker pattern is illustrated in this simple example.  The pattern
consists of one process, called the master, executing one block of code while
the rest of the processes, called workers, are executing a different block of code.

.. literalinclude:: ../patternlets/MPI/01.masterWorker/masterWorker.c
    :language: c
    :linenos:

02. Data Decomposition: on *equal-sized chunks* using parallel-for
************************************************************************

*file: patternlets/MPI/02.parallelLoop-equalChunks/parallelLoopEqualChunks.c*

*Build inside 02.parallelLoop-equalChunks directory:*
::

  make parallelLoopEqualChunks

*Execute on the command line inside 02.parallelLoop-equalChunks directory:*
::

  mpirun -np <number of processes> ./parallelLoopEqualChunks

The **data decomposition** pattern occurs in code in two ways:

  1. a for loop that traverses  many data elements stored in an array
  (1-dimensional or more). If each element in an array needs some sort of
  computation to be done on it, that work could be split between processes.
  This classic data decomposition pattern divides the array into equal-sized
  pieces, where each process works on a subset of the array assigned to it.

  2. a for loop that simply has a total of N independent iterations to perform
  a data calculation of some type. The work can be split into N/P 'chunks' of
  work, which can be performed on each of P processes.

In this example, we illustrate the second of these two. The total iterations to
perform are numbered from 0 to REPS in the code below. Each process will
complete REPS / numProcesses iterations, and will `start` and `stop` on its
chunk from 0 to, but not including REPS. Since each process receives REPS /
numProcesses consecutive iterations to perform, we declare this an *equal-sized
chunks* decomposition pattern. This type of decomposition is commonly used when
accessing data that is stored in consecutive memory locations (such as an
array).

.. topic:: To do:

  Verify that the program behavior is as follows for 4 processes:

  .. image:: MPIImages/EqualChunks.png
    :width: 800

  Run it more than once- what do you observe about the order in which the
  processes print their iterations? Try it for other numbers of processes from
  1 through 8. As you can guess, we cannot always get equal-sized chunks for
  all processes, but we can distribute chunks that differ by no more than one
  in size. When the REPS are equally divisible by the number of processes (e.g.
  2, 4, or 8 processes), the work is equally distributed among the processes.
  What happens when this is not the case (3, 5, 6, 7 processes)?

.. literalinclude:: ../patternlets/MPI/02.parallelLoop-equalChunks/parallelLoopEqualChunks.c
    :language: c
    :linenos:

03. Data Decomposition: on *chunks of size 1* using parallel-for
**************************************************************************

*file: patternlets/MPI/03.parallelLoop-chunksOf1/parallelLoopChunksOf1.c*

*Build inside 03.parallelLoop-chunksOf1 directory:*
::

  make parallelLoopChunksOf1

*Execute on the command line inside 03.parallelLoop-chunksOf1 directory:*
::

  mpirun -np <number of processes> ./parallelLoopChunksOf1

A simple decomposition sometimes used when your loop is not accessing consecutive
memory locations would be to let each process do one iteration, up to N processes,
then start again with process 0 taking the next iteration. A for loop on line 29
is used to implement this type of data decomposition.

.. image:: MPIImages/ChunksOf1.png
  :width: 800

This is a basic example that does not yet include a data array, though
it would typically be used when each process would be working on a portion
of an array that could have been looped over in a sequential solution.

.. literalinclude:: ../patternlets/MPI/03.parallelLoop-chunksOf1/parallelLoopChunksOf1.c
    :language: c
    :linenos:
