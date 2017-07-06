*********************************************************
Master Worker Pattern and Message Passing
*********************************************************

00. Single Program, Multiple Data
*********************************

*file: patternlets/MPI/00.spmd/spmd.c*

*Build inside 00.spmd directory:*
::

  make spmd

*Execute on the command line inside 00.spmd directory:*
::

  mpirun -np <number of processes> ./spmd

First let us illustrate the basic components of an MPI program,
which by its nature uses a single program that runs on each process.
Note what gets printed is different for each process, thus the
processes using this one single program can have different data values
for its variables.  This is why we call it single program, multiple data.

On the command line, *mpirun* tells the system to start <number of processes>
instances of the program. The call to *MPI_INIT* on line 25 tells the MPI
system to setup. This includes allocating storage for message buffers and
deciding the rank each process receives. *MPI_INIT* also defines a communicator
called *MPI_COMM_WORLD*. The function *MPI_Comm_rank* returns in its second
argument the rank of the calling process in the communicator. Similarly, the
function *MPI_Comm_size* returns as its second argument the number of processes
in *MPI_COMM_WORLD*. A communicator is a group of processes that can
communicate with each other by sending messages. The *MPI_Finalize* command
tells the MPI system that we are finished and it deallocates MPI resources.

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
consists of one thread, called the master, executing one block of code while
the rest of the threads, called workers, are executing a different block of code.

.. literalinclude:: ../patternlets/MPI/01.masterWorker/masterWorker.c
    :language: c
    :linenos:

02. Message passing 1, using Send-Receive of a single value
***********************************************************

*file: patternlets/MPI/02.messagePassing/messagePassing.c*

*Build inside 02.messagePassing directory:*
::

  make messagePassing

*Execute on the command line inside 02.messagePassing directory:*
::

  mpirun -np <number of processes> ./0messagePassing

Note: The number of processes must be positive and even.

This example shows the pattern of sending and receiving messages between
various processes. The following code displays message passing between pairs
of odd and even rank processes.

(rank 0, rank 1), (rank 2, rank 3), (rank 4, rank5), ... ,

The message that is being passed is the value of the square root of
the process rank. Conceptually, the running code is executing like this:

.. image:: MessagePassing.png
	:width: 800

.. literalinclude:: ../patternlets/MPI/02.messagePassing/messagePassing.c
    :language: c
    :linenos:

03. Message passing 2,  using Send-Receive of an array of values
****************************************************************

*file: patternlets/MPI/03.messagePassing2/messagePassing2.c*

*Build inside 03.messagePassing2 directory:*
::

  make messagePassing2

*Execute on the command line inside 03.messagePassing2 directory:*
::

  mpirun -np <number of processes> ./messagePassing2

Note: The number of processes must be positive and even.

The messages sent and received by processes can be of types other than
floats. Here the message that is being passed is a string (array of chars).
The *sprintf* function is similar to *printf* except that it writes to a string
instead of stdout. This example follows the previous messagePassing example in that it
passes strings between pairs of odd and even rank processes.

.. literalinclude:: ../patternlets/MPI/03.messagePassing2/messagePassing2.c
    :language: c
    :linenos:

04. Message passing 3,  using Send-Receive with master-worker pattern
*********************************************************************

*file: patternlets/MPI/04.messagePassing3/messagePassing3.c*

*Build inside 04.messagePassing3 directory:*
::

  make messagePassing3

*Execute on the command line inside 04.messagePassing3 directory:*
::

  mpirun -np <number of processes> ./messagePassing3

.. literalinclude:: ../patternlets/MPI/04.messagePassing3/messagePassing3.c
    :language: c
    :linenos:
