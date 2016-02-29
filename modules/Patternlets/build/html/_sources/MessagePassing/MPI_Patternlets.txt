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

1. The Barrier Coordination Pattern
*****************************************************

.. literalinclude:: ../patternlets/MPI/01.barrier/barrier.c
    :language: c


*file: patternlets/MPI/01.barrier/masterWorker.c*

2. The Master-Worker Implementation Strategy Pattern
*****************************************************

.. literalinclude:: ../patternlets/MPI/02.masterWorker/masterWorker.c
    :language: c


*file: patternlets/MPI/02.masterWorker/masterWorker.c*

3. Message passing 1, using Send-Receive of a single value
**********************************************************

.. literalinclude:: ../patternlets/MPI/03.messagePassing/messagePassing.c
    :language: c

*file: patternlets/MPI/03.messagePassing/messagePassing.c*

4. Message passing 2,  using Send-Receive of an array of values
***************************************************************

.. literalinclude:: ../patternlets/MPI/04.messagePassing2/messagePassing2.c
    :language: c

*file: patternlets/MPI/04.messagePassing2/messagePassing2.c*

5. Message passing 3,  using Send-Receive with master-worker pattern
********************************************************************

.. literalinclude:: ../patternlets/MPI/05.messagePassing3/messagePassing3.c
    :language: c

*file: patternlets/MPI/05.messagePassing3/messagePassing3.c*

6 (text). Data Decomposition: on *equal-sized chunks* using parallel-for 
************************************************************************

In this example, the data being decomposed is simply the set of integers 
from zero to REPS * numProcesses, which are used in the for loop.

.. literalinclude:: ../patternlets/MPI/06.parallelLoop-equalChunks/textual/parallelLoopEqualChunks.c
    :language: c

*file: patternlets/MPI/06.parallelLoop-equalChunks/textual/parallelLoopEqualChunks.c*

6 (visual). Data Decomposition: on *equal-sized chunks* using parallel-for
**************************************************************************

In this example, we can visually see how the slicing of data used in iterations
of a nested for loop is working.  Run it to see the effect!

.. literalinclude:: ../patternlets/MPI/06.parallelLoop-equalChunks/visual/parallelForBlocks.c
    :language: c


*file: patternlets/MPI/06.parallelLoop-equalChunks/visual/parallelForBlocks.c*

7 (text). Data Decomposition: on *chunks of size 1* using parallel-for 
**************************************************************************

This is a basic example that does not yet include a data array, though
it would typically be used when each process would be working on a portion
of an array that could have been looped over in a sequential solution.

.. literalinclude:: ../patternlets/MPI/07.parallelLoop-chunksOf1/textual/parallelLoopChunksOf1.c
    :language: c

*file: patternlets/MPI/07.parallelLoop-chunksOf1/textual/parallelLoopChunksOf1.c*

7 (visual). Data Decomposition: on *chunks of size 1* using parallel-for 
**************************************************************************

In this example you can see how blocks of values within a matrix might be
assigned to each process.  Run it to see the effect!

.. literalinclude:: ../patternlets/MPI/07.parallelLoop-chunksOf1/visual/parallelForSlices.c
    :language: c

*file: atternlets/MPI/07.parallelLoop-chunksOf1/visual/parallelForSlices.c*

8. Broadcast: a special form of message passing
**************************************************

This example shows how to ensure that all processes have a copy of an array
created by a single *master* node.

.. literalinclude:: ../patternlets/MPI/08.broadcast/broadcast.c
    :language: c

*file: patternlets/MPI/08.broadcast/broadcast.c*

9. Collective Communication: Reduction
**************************************

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


