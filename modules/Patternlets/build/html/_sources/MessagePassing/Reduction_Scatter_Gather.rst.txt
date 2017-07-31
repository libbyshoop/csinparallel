*********************************************************
Collective Communication
*********************************************************

With independent, often distributed processes, there is a need in many
program situations to have all the processes communicating with each other,
usually by sharing data, either before or after independent simultaneous
computations that each process performs. Here we see simple examples of these
collective communication patterns.

12. Collective Communication: Reduction
***************************************

*file: patternlets/MPI/12.reduction/reduction.c*

*Build inside 12.reduction directory:*
::

  make reduction

*Execute on the command line inside 12.reduction directory:*
::

  mpirun -np <number of processes> ./reduction

Once processes have performed independent concurrent computations, possibly
on some portion of decomposed data, it is quite common to then *reduce*
those individual computations into one value. This example shows a simple
calculation done by each process being reduced to a sum and a maximum.
In this example, MPI, has built-in computations, indicated by MPI_SUM and
MPI_MAX in the following code. With four processes, the code is implemented
like this:

.. image:: MPIImages/Reduction.png
	:width: 800

.. literalinclude:: ../patternlets/MPI/12.reduction/reduction.c
    :language: c
    :linenos:

13. Collective Communication: Reduction
****************************************

*file: patternlets/MPI/13.reduction2/reduction2.c*

*Build inside 13.reduction2 directory:*
::

  make reduction2

*Execute on the command line inside 13.reduction2 directory:*
::

  mpirun -np <number of processes> ./reduction2

Here is a second reduction example using arrays of data.

.. topic:: To do:

  Can you explain the reduction, `MPI_reduce`, in terms of srcArr and destArr?

.. literalinclude:: ../patternlets/MPI/13.reduction2/reduction2.c
    :language: c
    :linenos:

.. topic:: Further Exploration:

  This useful `MPI tutorial
  <http://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/>`_ explains other
  reduction operations that can be performed. You could use the above code or
  the previous examples to experiment with some of these.


14. Collective communication: Scatter for message-passing data decomposition
****************************************************************************

*file: patternlets/MPI/14.scatter/scatter.c*

*Build inside 14.scatter directory:*
::

  make scatter

*Execute on the command line inside 14.scatter directory:*
::

  mpirun -np <number of processes> ./scatter

If processes can independently work on portions of a larger data array
using the geometric data decomposition pattern, the scatter pattern can be
used to ensure that each process receives a copy of its portion of the array.
Process 0 gets the first chunk, process 1 gets the second chunk and so on until
the entire array has been distributed.

.. image:: MPIImages/Scatter.png
	:width: 700

.. topic:: To do:

  What previous data decomposition pattern is this similar to?

.. literalinclude:: ../patternlets/MPI/14.scatter/scatter.c
    :language: c
    :linenos:


15. Collective communication: Gather for message-passing data decomposition
***************************************************************************

*file: patternlets/MPI/15.gather/gather.c*

*Build inside 15.gather directory:*
::

  make gather

*Execute on the command line inside 15.gather directory:*
::

  mpirun -np <number of processes> ./gather

If processes can independently work on portions of a larger data array
using the geometric data decomposition pattern,
the gather pattern can be used to ensure that each process sends
a copy of its portion of the array back to the root, or master process.
Thus, gather is the reverse of scatter. Here is the idea:

.. image:: MPIImages/Gather.png
	:width: 750

.. topic:: To do:

  Find documentation for the MPI function MPI_Gather.
  Make sure that you know what each parameter is for.
  Why are the second and fourth parameters in our example
  both SIZE? Can you explain what this means in terms of
  MPI_Gather?

.. literalinclude:: ../patternlets/MPI/15.gather/gather.c
    :language: c
    :linenos:
