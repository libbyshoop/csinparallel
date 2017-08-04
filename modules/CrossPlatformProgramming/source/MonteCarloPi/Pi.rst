****************************************
Estimating Pi using Monte Carlo Method
****************************************

This example demonstrates the Monte Carlo method for estimating the value of
:math:`\pi`. Monte Carlo methods rely on repeated independent and random sampling.
Such methods work well with parallel shared systems and distributed systems as
the work can be split among many threads or processes.

The problem can be imagined in terms of playing darts. Let the
dartboard consist of a square target with a circular target inside of it.
To solve this by means of using a 'Monte Carlo Simulation', you would simply
throw a bunch of darts at the target and record the percentage
that land in the inner circular target.

We can extend this idea to approximate :math:`\pi` quite easily.
Suppose the square target has a length of two feet and the circular target
has a radius of one foot.

.. image:: PiBoard.png
	:width: 400

Based on the dimensions of the board, we have that the ratio of the area
of the circle to the area of the square is

.. math::

    \frac{\pi {1}^2}{2^2} = \frac{\pi}{4}


As it happens, we can calculate a value for the ratio of the area of the circle
to the area of the square with a Monte Carlo simulation. We pick random
points in the square and find the ratio of the number of points inside the circle
to the total number of points. This ratio should approach :math:`\frac{\pi}{4}`.
We multiply this by 4 to get our estimate of :math:`\pi`.

This can be simplified by using only a quarter of the board. The ratio of the
area of the circle to the area of the square is still :math:`\pi`/4. To
simulate the throw of a dart, we generate a number of random points with
coordinates (x,y). These coordinates are uniformly distributed random numbers
between 0 and 1. Then, we determine how many of these points fall inside of
the circle and take the ratio of the areas.

Sequential Code
===============

*file: cross_platform_examples/monteCarloPi/calcPiSeq/calcPiSeq.C*


*Build inside calcPiSeq directory:*
::

  make calcPiSeq

*Execute on the command line inside calcPiSeq directory:*
::

   ./calcPiSeq <number of tosses>

The code follows from the description of the problem. One thing to point out is
the use of random generator *rand_r*. *rand_r* is a reentrant and thread-safe
function that allows us to get reproducible behavior. This is the reason behind
using C++ for coding this problem.


.. topic:: To do:

		Run and compile the code experimenting with the number of tosses. Compare
		the source code to the output. What do you notice about the accuracy of
		our estimation of pi as the number of tosses increase?

		Record execution times using 16 million, 32 million, 64 million, 128 million,
		and 256 million for the number of tosses.

.. literalinclude:: ../cross_platform_examples/monteCarloPi/calcPiSeq/calcPiSeq.C
	:language: C++
	:linenos:


OpenMP Code
============

*file: cross_platform_examples/monteCarloPi/calcPiOMP/calcPiOMP.C*


*Build inside calcPiOMP directory:*
::

  make calcPiOMP

*Execute on the command line inside calcPiOMP directory:*
::

   ./calcPiOMP <number of threads> <number of tosses>

The shared memory version of the code begins by setting the number of threads for
the program. We then calculate the number of tosses each thread will simulate.
The block of code beneath the #pragma omp parallel is run by each individual thread.
To get the total number of tosses that land in the circle, we reduce each thread's
relevant tosses to a single value.

.. literalinclude:: ../cross_platform_examples/monteCarloPi/calcPiOMP/calcPiOMP.C
	 :language: C++
	 :lines: 30-41

.. topic:: To do:

		Find the speedup and efficiency of this program. To do so, you will
	 	need your execution times above from the sequential version of calculating pi
		using the Monte Carlo method.

		Use 2, 4, 8, 12, 14, and 16 for the number of processes and 16 million,
		32 million, 64 million, 128 million, and 256 million for the number of tosses.

		Make a copy of the template provided 
		`here <https://docs.google.com/spreadsheets/d/1GBgyDzKhQIh_BVFJOi1LHbEtputyEh5rQ5ETn1ZRi9U/edit?usp=sharing.>`_ and
		record the execution times from each combination in the execution time table.
		The speedup and efficiency of each combination will automatically be calculated
		and corresponding speedup and efficiency graphs will be made.



MPI Code
=========

*file: cross_platform_examples/monteCarloPi/calcPiMPI/calcPiMPI.C*

*Build inside calcPiMPI directory:*
::

  make calcPiMPI

*Execute on the command line inside calcPiMPI directory:*
::

  mpirun -np <N> ./calcPiMPI <number of tosses>

.. note::

	 This command is going to run all processes on the machine on which you
	 type it. You will need a separate machines file for running the code
	 on a cluster of machines. This note applies for all examples utilizing MPI.

The distributed memory version starts with initializing the execution environment
and assigning a unique rank to each process. Next, we calculate the number of
tosses that each process will sample. All processes sample their predetermined
number of tosses and determine whether or not they fall inside the circle. The
local values for tosses that land inside the circle are reduced to a single value.

.. literalinclude:: ../cross_platform_examples/monteCarloPi/calcPiMPI/calcPiMPI.C
	:language: C++
	:lines: 34-43

.. topic:: To do:

		Find the speedup and efficiency of this program the same way you did previously
		for the OpenMP version. To do so, you will need your execution times from the sequential
		version of calculating pi using the Monte Carlo method above.

		Use 2, 4, 8, 12, 14, and 16 for the number of processes and 16 million,
		32 million, 64 million, 128 million, and 256 million for the number of tosses.

		Make a copy of the template provided at
		`this link <https://docs.google.com/spreadsheets/d/1ff1yFkz4cMheYPaZIiA29J_GzCFNW4tKAMoDjPQu130/edit?usp=sharing.>`_ and
		record the execution times from each combination in the execution time table.
		The speedup and efficiency of each combination will automatically be calculated
		and corresponding speedup and efficiency graphs will be made.

		Compare the speedup and efficiency of this program to the speedup and efficiency
		of the OpenMP program. What do you observe?

MPI+OpenMP Hybrid Code
========================

*file: cross_platform_examples/monteCarloPi/calcPiHybrid/calcPiHybrid.C*

*Build inside calcPiHybrid directory:*
::

  make calcPiHybrid

*Execute on the command line inside calcPiHybrid directory:*
::

  mpirun -np <N> ./calcPiHybrid <number of threads> <number of tosses>

This hybrid version relies on both distributed memory from MPI and shared
memory from OpenMP. A programming model that combines MPI and OpenMP uses
MPI to distribute work among processes, each of which uses OpenMP to assign
threads to its task. First, we calculate the number of tosses that each process'
threads will sample. Then, each process initializes its threads. Every thread
simulates the predetermined number of tosses and counts how many land in circle.
The local values for tosses that land inside the circle from threads are reduced
before each process' local values are reduced.

.. literalinclude:: ../cross_platform_examples/monteCarloPi/calcPiHybrid/calcPiHybrid.C
	:language: C++
	:lines: 35-51

.. topic:: To do:

		Try the hybrid program with different number of processes and different number
		of threads. What combinations of processes and threads seem to run faster?
		Why might this be the case?

		Run the program using 4 processes, 4 threads and 64,000,000 tosses.
		Compare the execution time to the time it took to run the MPI program using
		4 processes and 16,000,000 tosses. How do the times compare?

		Run the program using 4 processes, 4 threads and 256,000,000 tosses.
		Compare the execution time to the time it took to run the MPI program using
		4 processes and 64,000,000 tosses. Can you explain this behavior?
