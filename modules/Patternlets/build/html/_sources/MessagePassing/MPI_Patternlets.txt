*************************************
Message Passing Parallel Patternlets
*************************************

Parallel programs contain *patterns*:  code that recurs over and over again
in solutions to many problems.  The following examples show very simple
examples of small portions of
these patterns that can be combined to solve a problem.  These C code examples use the
Message Passing Interface (MPI) library, which is suitable for use on either a
single multiprocessor machine or a cluster
of machines.

Source Code
************

Please download all examples from this tarball: 
:download:`MPI.tgz <../patternlets/MPI.tgz>`

A C code file for each example below can be found in subdirectories of the MPI directory,
along with a makefile and an example of how to execute the program.

00. Single Program, Multiple Data
*********************************

First let us illustrate the basic components of an MPI program,
which by its nature uses a single program that runs on each process.
Note what gets printed is different for each process, thus the
processes using this one single program can have different data values
for its variables.  This is why we call it single program, multiple data.

.. literalinclude:: ../patternlets/MPI/00.spmd/spmd.c
    :language: c

.. comment
    :lines: 36-51

*file: patternlets/MPI/00.spmd/spmd.c*



01. The Master-Worker Implementation Strategy Pattern
*****************************************************

.. literalinclude:: ../patternlets/MPI/01.masterWorker/masterWorker.c
    :language: c


*file: patternlets/MPI/01.masterWorker/masterWorker.c*

02. Message passing 1, using Send-Receive of a single value
***********************************************************

.. literalinclude:: ../patternlets/MPI/02.messagePassing/messagePassing.c
    :language: c

*file: patternlets/MPI/02.messagePassing/messagePassing.c*

03. Message passing 2,  using Send-Receive of an array of values
****************************************************************

.. literalinclude:: ../patternlets/MPI/03.messagePassing2/messagePassing2.c
    :language: c

*file: patternlets/MPI/03.messagePassing2/messagePassing2.c*

04. Message passing 3,  using Send-Receive with master-worker pattern
*********************************************************************

.. literalinclude:: ../patternlets/MPI/04.messagePassing3/messagePassing3.c
    :language: c

*file: patternlets/MPI/04.messagePassing3/messagePassing3.c*

05. Data Decomposition: on *equal-sized chunks* using parallel-for 
************************************************************************

In this example, the data being decomposed is simply the set of integers 
from zero to REPS * numProcesses, which are used in the for loop.

.. literalinclude:: ../patternlets/MPI/05.parallelLoop-equalChunks/parallelLoopEqualChunks.c
    :language: c

*file: patternlets/MPI/05.parallelLoop-equalChunks/parallelLoopEqualChunks.c*


06. Data Decomposition: on *chunks of size 1* using parallel-for 
**************************************************************************

This is a basic example that does not yet include a data array, though
it would typically be used when each process would be working on a portion
of an array that could have been looped over in a sequential solution.

.. literalinclude:: ../patternlets/MPI/06.parallelLoop-chunksOf1/parallelLoopChunksOf1.c
    :language: c

*file: patternlets/MPI/06.parallelLoop-chunksOf1/parallelLoopChunksOf1.c*


07. Broadcast: a special form of message passing
**************************************************

This example shows how a data item read from a file can be sent to all the processes.

.. literalinclude:: ../patternlets/MPI/07.broadcast/broadcast.c
    :language: c

*file: patternlets/MPI/07.broadcast/broadcast.c*

08. Broadcast: send data to all processes
**************************************************

This example shows how to ensure that all processes have a copy of an array
created by a single *master* node.

.. literalinclude:: ../patternlets/MPI/08.broadcast2/broadcast2.c
    :language: c

*file: patternlets/MPI/08.broadcast2/broadcast2.c*

09. Collective Communication: Reduction
***************************************

Once processes have performed independent concurrent computations, possibly
on some portion of decomposed data, it is quite common to then *reduce*
those individual computations into one value.  This example shows a simple
calculation done by each process being reduced to a sum and a maximum.
In this example, MPI, has built-in computations, indicated by MPI_SUM and
MPI_MAX in the following code.

.. literalinclude:: ../patternlets/MPI/09.reduction/reduction.c
    :language: c

*file: patternlets/MPI/09.reduction/reduction.c*

10. Collective Communication: Reduction
****************************************

Here is a second reduction example using arrays of data.

.. literalinclude:: ../patternlets/MPI/10.reduction2/reduction2.c
    :language: c

*file: patternlets/MPI/10.reduction2/reduction2.c*

11. Collective communication: Scatter for message-passing data decomposition
****************************************************************************

If processes can independently work on portions of a larger data array
using the geometric data decomposition pattern,
the scatter pattern can be used to ensure that each process receives
a copy of its portion of the array.

.. literalinclude:: ../patternlets/MPI/11.scatter/scatter.c
    :language: c

*file: patternlets/MPI/11.scatter/scatter.c*

12. Collective communication: Gather for message-passing data decomposition
***************************************************************************

If processes can independently work on portions of a larger data array
using the geometric data decomposition pattern,
the gather pattern can be used to ensure that each process sends
a copy of its portion of the array back to the root, or master process.

.. literalinclude:: ../patternlets/MPI/12.gather/gather.c
    :language: c

*file: patternlets/MPI/12.gather/gather.c*

13. The Barrier Coordination Pattern
*****************************************************

A barrier is used when you want all the processes to complete a portion of
code before continuing. Use this exercise to verify that is is ocurring when
you add the call to the MPI_Barrier funtion.

.. literalinclude:: ../patternlets/MPI/13.barrier/barrier.c
    :language: c


*file: patternlets/MPI/13.barrier/barrier.c*

14. Timing code using the Barrier Coordination Pattern
******************************************************

In this example you can run the code several times and determine the average, median, and minimum
execution time when the code has a barrier and when it does not. The primary purpose of this exercise
is to illustrate that one of the most useful uses of a barrier is to ensure that you are getting legitimate
timings for your code examples. By using a barrier, you ensure that all processes have finished before
recording the time using the master node.

.. literalinclude:: ../patternlets/MPI/14.barrier+Timing/barrier+timing.c
    :language: c


*file: patternlets/MPI/14.barrier+Timing/barrier+timing.c*

15. Sequence Numbers
*****************************************************

Tags can be placed on messages that are sent from a non-master node and received by the master node.
Using tags is an alternative form of simulating the barrier example in example 13 above.

.. literalinclude:: ../patternlets/MPI/15.sequenceNumbers/sequenceNumbers.c
    :language: c


*file: patternlets/MPI/15.sequenceNumbers/sequenceNumbers.c*

