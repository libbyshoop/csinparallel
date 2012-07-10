=======================
Compiling and Activites
=======================

Compiling MPI program on LittleFe
---------------------------------

To compile a MPI program on our LittleFe, we can use the following command in the terminal:

*mpicc -o name of executable file filename.c*

*mpirun -np #processes ./mpi* or *mpirun -machinefile machines -np #processes ./mpi*

**machines**: is the instruction for running the executable file on which node, and how many times on which node.

Activity 1: PI Computation
--------------------------

In this activity, we are going to compute PI using integral techniques. We have formula:

Formula


Therefore, we can calculating the sum of area under the curve to get the value of the integral. We can split the sum into bins. The idea is we want to split the bins into smaller chunks, and so we can use each process to calculate each chunk, and then combine the result into one value. Remember, that we can get a more accurate result if you split the area under the curve to more number of bins.

In this activity, we also want to time our computation by using MPI_Wtime() function. We provide you some parts of the codes, and would like you to write some at TO DO, and experiment with the number of bins you are using. Moreover, you want you to run with different number of processes, and compare your timings.

.. highlight:: c

.. literalinclude:: mpi_pi.c
	:linenos:


Activity 2: Vector Matrix Multiplication
----------------------------------------	

In this activity, we will compute vector matrix multiplication. This multiplication will produces a vector of length 
as same as that of the input vector. In this activity, we will illustrate the use of MPI_Bcast, MPI_Scatter, and MPI_Gather to do this multiplication. First, we would want you to complete this MPI program by filling codes at *TO DO*. After having completed, try to run this MPI program on LittleFe by using different number of processes. 


.. literalinclude:: vector_matrix_buggy.c
	:linenos:


