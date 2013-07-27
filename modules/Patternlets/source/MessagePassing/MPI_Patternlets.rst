*************************************
Message Passing Parallel Patternlets
*************************************

Parallel programs contain *patterns*:  code that recurs over and over again
in solutions to many problems.  The following examples show very simple
examples of small portions of
these patterns that can be combined to solve a problem.  These C code examples use the
Message Passing Interface (MPI) library, which is suitable for use on either a
single pultiprocessor machine or a cluster
of machines.

Source Code
************

Please download all examples from this tarball: 
:download:`patternlets.tgz <../patternlets.tgz>`

A C code file for each example below can be found in subdirectories of the MPI directory,
along with a makefile and an example of how to execute the program.

0. Single Program, Multiple Data
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

1. The Master-Worker Implementation Strategy Pattern
*****************************************************

.. literalinclude:: ../patternlets/MPI/01.masterWorker/masterWorker.c
    :language: c


*file: patternlets/MPI/01.masterWorker/masterWorker.c*

2. Message passing 1, using Send-Receive of a single value
**********************************************************

.. literalinclude:: ../patternlets/MPI/02.messagePassing/messagePassing.c
    :language: c

*file: patternlets/MPI/02.messagePassing/messagePassing.c*

3. Message passing 2,  using Send-Receive of an array of values
***************************************************************

.. literalinclude:: ../patternlets/MPI/03.messagePassing2/messagePassing2.c
    :language: c

*file: patternlets/MPI/03.messagePassing2/messagePassing2.c*

4. A. Data Decomposition: on *slices* using parallel-for (textual version)
**************************************************************************

In this example, the data being decomposed is simply the set of integers 
from zero to REPS * numProcesses, which are used in the for loop.

.. literalinclude:: ../patternlets/MPI/04.parallelForLoop-slices/textual/parallelForSlices.c
    :language: c

*file: patternlets/MPI/04.parallelForLoop-slices/textual/parallelForSlices.c*

4. B. Data Decomposition: on *slices* using parallel-for (visual version)
**************************************************************************

In this example, we can visually see how the slicing of data used in iterations
of a nested for loop is working.  Run it to see the effect!

.. literalinclude:: ../patternlets/MPI/04.parallelForLoop-slices/visual/parallelForSlices.c
    :language: c


*file: patternlets/MPI/04.parallelForLoop-slices/visual/parallelForSlices.c*

5. A. Data Decomposition: on *blocks* using parallel-for (textual version)
**************************************************************************

This is a basic example that does not yet include a data array, though
it would typically be used when each process would be working on a portion
of an array that could have been looped over in a sequential solution.

.. literalinclude:: ../patternlets/MPI/05.parallelForLoop-blocks/textual/parallelForBlocks.c
    :language: c

*file: patternlets/MPI/05.parallelForLoop-blocks/textual/parallelForBlocks.c*

5. B. Data Decomposition: on *blocks* using parallel-for (visual version)
**************************************************************************

In this example you can see how blocks of values within a matrix might be
assigned to each process.  Run it to see the effect!

.. literalinclude:: ../patternlets/MPI/05.parallelForLoop-blocks/visual/parallelForBlocks.c
    :language: c

*file: atternlets/MPI/05.parallelForLoop-blocks/visual/parallelForBlocks.c*

6. Broadcast: a special form of message passing
**************************************************

This example shows how to ensure that all processes have a copy of an array
created by a single *master* node.

.. literalinclude:: ../patternlets/MPI/06.broadcast/broadcast.c
    :language: c

*file: patternlets/MPI/06.broadcast/broadcast.c*

7. Collective Communication: Reduction
**************************************

Once processes have performed independent concurrent computations, possibly
on some portion of decomposed data, it is quite commen to then *reduce*
those individual computations into one value.  This example shows a simple
calculation done by each process being reduced to a sum and a maximum.
In this example, MPI, has built-in computations, indicated by MPI_SUM and
MPI_MAX in the following code.

.. literalinclude:: ../patternlets/MPI/07.reduction/reduction.c
    :language: c

*file: patternlets/MPI/07.reduction/reduction.c*

8. Collective communication: Scatter for message-passing data decomposition
****************************************************************************

If processes can independently work on portions of a larger data array
using the geometric data decomposition pattern,
the scatter patternlet can be used to ensure that each process receives
a copy of its portion of the array.

.. literalinclude:: ../patternlets/MPI/08.scatter/scatter.c
    :language: c

*file: patternlets/MPI/08.scatter/scatter.c*

9. Collective communication: Gather for message-passing data decomposition
***************************************************************************

If processes can independently work on portions of a larger data array
using the geometric data decomposition pattern,
the gather patternlet can be used to ensure that each process sends
a copy of its portion of the array back to the root, or master process.

.. literalinclude:: ../patternlets/MPI/09.gather/gather.c
    :language: c

*file: patternlets/MPI/09.gather/gather.c*

10. Collective Communication: Barrier
****************************************

This simple example shows the use of a barrier: a point at which all processes
must complete the code above it before moving on and running the code below it.

.. literalinclude:: ../patternlets/MPI/10.barrier/barrier.c
    :language: c

*file: patternlets/MPI/10.barrier/barrier.c*

