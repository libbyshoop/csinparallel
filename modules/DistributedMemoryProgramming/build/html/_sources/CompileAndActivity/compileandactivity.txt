=======================
Compiling and Activites
=======================

Compiling MPI program on LittleFe
---------------------------------

To compile a MPI program on our LittleFe, we can use the following command in the terminal:

First we need to make an object from the MPI program: ::

	mpicc -o name_executable_file filename.c

Then we are able to compile that object using **mpirun** : ::

	mpirun -machinefile machines -np #processes ./name_executable_file

.. note:: 
	**machines**: is the instruction for running the executable file on which node, and how many times on which node.

Moreover, you can also compile a MPI program without using **machines**, you can use the following command to run only on the master node: ::

	mpirun -np #processes ./name_executable_file

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

I will explain how the vector matrix multiplication works. First let's say we have a matrix *A*, and a vector *x* as below:  

.. image:: images/vector_matrix_multi.png
	:width: 500px
	:align: center
	:height: 300px
	:alt: MPI Structure

.. centered:: Figure 4: vector matrix multiplication

This multiplication produces a new vector whose length is the number of rows of *A*. The multiplication is very simple, we just need to take a row of *A* dot product with *x*, and this produces an element of result vector. For example, the first row of *A* dot products with *x* will produce the first element in vector *y*. 


http://cms.uni-konstanz.de/fileadmin/informatik/ag-saupe/Webpages/lehre/na_08/Lab1/1_Preliminaries/html/matrixVectorProduct.html

I will step you through the source code for this MPI program. Since this is a MPI program, we need to create MPI execution environment. You are asked to completed this part of the source code.

After having initialized the MPI environment, we want to ask the master to initialize the vector and matrix we are going to multiply. In order to do that, first we check if the process is master, if so we just need to initialize the matrix and vector. ::

	if (rank == 0) {

        /* Initialize Matrix and Vector */
        for(i=0; i < WIDTH; i++) {
            vector[i] = 1;
            for(j = 0; j < WIDTH; j++) {
                matrix[i][j] = 1;
            }
        }
    }

Since the vector is not very large here and all processes need to have vector to do the multiplication, we will broadcast entire vector to all processes. We do that by using MPI_Bcast. Also, we want to distribute the matrix to each process in the MPI_COMM_WORLD. We would do this using MPI_Scatter. You are asked to complete this part. 

When all processes are able to see the vector and part of matrix, they are now able to do the multiplication. We need to store the result in result matrix. ::

	for(i = 0; i < chunk_size; i++) {
        result[i] = 0;
    	for(j = 0; j < WIDTH; j++) {
        	result[i] += local_matrix[i][j] * vector[j];
        }
    }

The last part you need to complete is to gather all result vectors in all processes, and store them in the output vector, which is called global_result vector. This will be our result vector. Moreover, we can print out the value of each element in the global_result vector, and then terminate the MPI execution environment.    

Below is the entire source code for vector matrix multiplication :    

.. literalinclude:: vector_matrix_buggy.c
	:linenos:


