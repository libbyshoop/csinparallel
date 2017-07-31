*********************************************************
Advanced Topics
*********************************************************

20. Advanced Data Decomposition: on *equal-sized chunks* using parallel-for
***************************************************************************

*file: patternlets/MPI/20.parallelLoopAdvanced/parallelLoopChunks.c*

*Build inside 20.parallelLoopAdvanced directory:*
::

  make parallelLoopChunks

*Execute on the command line inside 20.parallelLoopAdvanced directory:*
::

  mpirun -np <number of processes> ./parallelLoopChunks

This example is a continuation of example 2 which showed data decomposition on
*equal-sized chunks* using parallel-for. Recall that the program only ran
correctly when the work was equally divisible among the processes.
We will delve into how to approach situations in which the chunks of work are not
always the same size. We are able to do this by equally distributing chunks of work
that differ by no more than one in size among the processes. The diagram below
illustrates this concept with 8 iterations assigned to 3 processes.

.. image:: MPIImages/AdvancedParallelLoop.png
  :width: 600

.. topic:: To do:

  Compile and run the code varying the number of processes from 1 through 8.
  What do you notice about the how the iterations of the loop are divided among
  the processes? Can you explain this in terms of chunkSize1 and chunkSize2?

.. literalinclude:: ../patternlets/MPI/20.parallelLoopAdvanced/parallelLoopChunks.c
    :language: c
    :linenos:

21. Advanced Broadcast and Data Decomposition
***********************************************

*file: patternlets/MPI/21.broadcast+ParallelLoop/broadcastLoop.c*

*Build inside 21.broadcast+ParallelLoop directory:*
::

  make broadcastLoop

*Execute on the command line inside 21.broadcast+ParallelLoop directory:*
::

  mpirun -np <number of processes> ./broadcastLoop


We once again expand upon example 2 on data decomposition using parallel-for
loop with *equal-sized chunks* to incorporate broadcast and gather.
We begin by filling an array with values and broadcasting this array to all processes.
Afterwards, each process works on their portion of the array which has been determined
by the equal sized chunks data decomposition pattern. Lastly, all of the worked
on portions of the array are gathered into an array containing the final result.
Below is a diagram of the code executing using 4 processes. The diagram assumes that
we have already broadcast the filled array to all processes.

.. image:: MPIImages/AdvancedBroadcastParallelLoop.png
  :width: 700


Note that we chose to keep the original array, *array*, intact.
Each process allocates memory, *myChunk* to store their worked on portion of
the array. Later, the worked on portions from all processes are gathered into a
final result array, *gatherArray*. This way of working on array is useful in
instances in which we want to be able to access the initial array after working on it.

.. literalinclude:: ../patternlets/MPI/21.broadcast+ParallelLoop/broadcastLoop.c
    :language: c
    :linenos:
