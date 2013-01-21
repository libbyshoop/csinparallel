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

0. Hello, World
***************

First let us illustrate the basic components of an MPI program.

.. literalinclude:: ../patternlets/MPI/00.hello/hello.c
    :language: c

.. comment
    :lines: 36-51

1. The Master-Worker Implementation Strategy Pattern
*****************************************************

.. literalinclude:: ../patternlets/MPI/01.masterWorker/masterWorker.c
    :language: c


2. Send-Receive (basic message passing pattern)
************************************************

.. literalinclude:: ../patternlets/MPI/02.sendRecv/sendRecv.c
    :language: c

3. Data Decomposition: on *slices* using parallel-for
*********************************************************************

In this example, the data being decomposed in simply the set of integeres from zero to 15, inclusive.

.. literalinclude:: ../patternlets/MPI/03.parallelForSlices/parallelForSlices.c
    :language: c

4. Data Decomposition: on *blocks* using parallel-for
*******************************************************

This is a basic example that does not yet include a data array, though
it would typically be used when each process would be working on a portion
of an array that could have been looped over in a sequential solution.

.. literalinclude:: ../patternlets/MPI/04.parallelForBlocks/parallelForBlocks.c
    :language: c

5. Broadcast: a special form of message passing
************************************************

This example shows how to ensure that all processes have a copy of an array
created by a single *master* node.

.. literalinclude:: ../patternlets/MPI/05.bcast/bcast.c
    :language: c

6. Collective Communication: Reduction
**************************************

Once processes have performed independent concurrent computations, possibly
on some portion of decomposed data, it is quite commen to then *reduce*
those individual computations into one value.  This example shows a simple
calculation done by each process being reduced to a sum and a maximum.
In this example, MPI, has built-in computations, indicated by MPI_SUM and
MPI_MAX in the following code.

.. literalinclude:: ../patternlets/MPI/06.reduce/reduce.c
    :language: c

7. Collective communication: Scatter for message-passing data decomposition
****************************************************************************

If processes can independently work on portions of a larger data array
using the geometric data decomposition pattern,
the scatter patternlet can be used to ensure that each process receives
a copy of its portion of the array.

.. literalinclude:: ../patternlets/MPI/07.scatter/scatter.c
    :language: c

8. Collective communication: Gather for message-passing data decomposition
***************************************************************************

If processes can independently work on portions of a larger data array
using the geometric data decomposition pattern,
the gather patternlet can be used to ensure that each process sends
a copy of its portion of the array back to the root, or master process.

.. literalinclude:: ../patternlets/MPI/08.gather/gather.c
    :language: c


