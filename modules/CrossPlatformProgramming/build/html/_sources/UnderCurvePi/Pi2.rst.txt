***************************************************
Estimating Pi by Calculating Area Under the Curve
***************************************************

Problem Write-up

Sequential Code
===============

*file: cross_platform_examples/piIntegration/seq_pi_done/seq_pi_done.c*


*Build inside seq_pi_done directory:*
::

  make seq_pi_done

*Execute on the command line inside seq_pi_done directory:*
::

   ./seq_pi_done <number of bins>


.. topic:: To do:

		Run and compile the code experimenting with the number of bins. Compare
		the source code to the output. What do you notice about the accuracy of
		our estimation of pi as the number of bins increase?

		Record execution times using 125 million, 250 million, 500 million, 1 billion,
		and 2 billion for the number of bins.

.. literalinclude:: ../cross_platform_examples/piIntegration/seq_pi_done/seq_pi_done.c
	:language: c
	:linenos:


OpenMP Code
============

*file: cross_platform_examples/piIntegration/omp_pi_done/omp_pi_done.c*


*Build inside omp_pi_done directory:*
::

  make omp_pi_done

*Execute on the command line inside omp_pi_done directory:*
::

   ./omp_pi_done <number of threads> <number of bins>


.. topic:: To do:

		Find the speedup and efficiency of this program. To do so, you will
	 	need your execution times above from the sequential version of calculating pi
		using the Monte Carlo method.

		Use 2, 4, 8, 12, 14, and 16 for the number of processes and 125 million, 250 million,
		500 million, 1 billion, and 2 billion for the number of bins.

		Make a copy of the template provided at
		`this link <https://docs.google.com/spreadsheets/d/1GBgyDzKhQIh_BVFJOi1LHbEtputyEh5rQ5ETn1ZRi9U/edit?usp=sharing.>`_ and
		record the execution times from each combination in the execution time table.
		The speedup and efficiency of each combination will automatically be calculated
		and corresponding speedup and efficiency graphs will be made.



MPI Code
=========

*file: cross_platform_examples/piIntegration/mpi_pi_done/mpi_pi_done.c*

*Build inside mpi_pi_done directory:*
::

  make mpi_pi_done

*Execute on the command line inside mpi_pi_done directory:*
::

  mpirun -np <N> ./mpi_pi_done <number of bins>

.. note::

	 This command is going to run all processes on the machine on which you
	 type it. You will need a separate machines file for running the code
	 on a cluster of machines. This note applies for all examples utilizing MPI.

.. topic:: To do:

		Find the speedup and efficiency of this program the same way you did previously
		for the OpenMP version. To do so, you will need your execution times from the sequential
		version of calculating pi using the Monte Carlo method above.

		Use 2, 4, 8, 12, 14, and 16 for the number of processes and 125 million, 250 million,
		500 million, 1 billion, and 2 billion for the number of bins.

		Make a copy of the template provided
		`here <https://docs.google.com/spreadsheets/d/1ff1yFkz4cMheYPaZIiA29J_GzCFNW4tKAMoDjPQu130/edit?usp=sharing.>`_ and
		record the execution times from each combination in the execution time table.
		The speedup and efficiency of each combination will automatically be calculated
		and corresponding speedup and efficiency graphs will be made.

		Compare the speedup and efficiency of this program to the speedup and efficiency
		of the OpenMP program. What do you observe?

MPI+OpenMP Hybrid Code
========================

*file: cross_platform_examples/piIntegration/hybrid_pi_done/hybrid_pi_done.c*

*Build inside hybrid_pi_done directory:*
::

  make hybrid_pi_done

*Execute on the command line inside hybrid_pi_done directory:*
::

  mpirun -np <N> ./hybrid_pi_done <number of threads> <number of bins>


.. topic:: To do:

		Try the hybrid program with different number of processes and different number
		of threads. What combinations of processes and threads seem to run faster?
		Why might this be the case?

		Run the program using 4 processes, 4 threads and 500,000,000 bins.
		Compare the execution time to the time it took to run the MPI program using
		4 processes and 125,000,000 bins. How do the times compare?

		Run the program using 4 processes, 4 threads and 2,000,000,000 bins.
		Compare the execution time to the time it took to run the MPI program using
		4 processes and 500,000,000 bins. Can you explain this behavior?
