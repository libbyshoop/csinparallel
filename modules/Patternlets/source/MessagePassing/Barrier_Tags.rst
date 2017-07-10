*************************************
Barrier Synchronization and Tags
*************************************

15. The Barrier Coordination Pattern
*****************************************************

*file: patternlets/MPI/15.barrier/barrier.c*

*Build inside 15.barrier directory:*
::

  make barrier

*Execute on the command line inside 15.barrier directory:*
::

  mpirun -np <number of processes> ./barrier

A barrier is used when you want all the processes to complete a portion of
code before continuing. Use this exercise to verify that it is occurring when
you add the call to the MPI_Barrier function.

.. literalinclude:: ../patternlets/MPI/15.barrier/barrier.c
    :language: c
    :linenos:

16. Timing code using the Barrier Coordination Pattern
******************************************************

*file: patternlets/MPI/16.barrier+Timing/barrier+timing.c*

*Build inside 16.barrier+Timing directory:*
::

  make marrier+timing

*Execute on the command line inside 16.barrier+Timing directory:*
::

  mpirun -np <number of processes> ./barrier+timing

In this example you can run the code several times and determine the average, median, and minimum
execution time when the code has a barrier and when it does not. The primary purpose of this exercise
is to illustrate that one of the most useful uses of a barrier is to ensure that you are getting legitimate
timings for your code examples. By using a barrier, you ensure that all processes have finished before
recording the time using the master process.

.. literalinclude:: ../patternlets/MPI/16.barrier+Timing/barrier+timing.c
    :language: c
    :linenos:

17. Sequence Numbers
*****************************************************

*file: patternlets/MPI/17.sequenceNumbers/sequenceNumbers.c*

*Build inside 17.sequenceNumbers directory:*
::

  make sequenceNumbers

*Execute on the command line inside 17.sequenceNumbers directory:*
::

  mpirun -np <number of processes> ./sequenceNumbers

Tags can be placed on messages that are sent from a non-master process and received
by the master process. Using tags is an alternative form of simulating the barrier
in example 15 above.

.. literalinclude:: ../patternlets/MPI/17.sequenceNumbers/sequenceNumbers.c
    :language: c
    :linenos:
