=======================
Compiling and Activites
=======================

.. highlight:: c

Compiling MPI program on LittleFe
---------------------------------

To compile a MPI program on our LittleFe, we can use the following commands in the terminal:

First we need to make an object from the MPI program: ::

	mpicc -o name_executable_file filename.c

Then we are able to compile that object using **mpirun** : ::

	mpirun -machinefile machines -np #processes ./name_executable_file

.. note:: 
	**machines**: is the instruction for running the executable file on which node, and how many times on which node. It has the structure as follow: 

        - node000.bccd.net    slots = 1
        - node011.bccd.net    slots = 1
        - node012.bccd.net    slots = 1
        - node013.bccd.net    slots = 1 
        - node014.bccd.net    slots = 1
        - node015.bccd.net    slots = 1

Moreover, you can also compile a MPI program without using **machines**, you can use the following command to run only on the master node: ::

	mpirun -np #processes ./name_executable_file

Activity 1: PI Computation
--------------------------

In this activity, we are going to compute PI using integration. We have formula:

.. math::

    \int_0^1 \frac{1}{1 + x^2} dx = \frac {\pi}{4}

Therefore, we can compute the area under the curve to get the value of the integral. 

.. image:: images/graph.png
    :width: 300px
    :align: center
    :height: 150px
    :alt: graph

.. centered:: Figure 4: Graph for function


We can split the sum into bins. The idea is to split the bins into smaller chunks, and so we can use each process to calculate each chunk, and then combine the result into one value. Remember, that we can get a more accurate result if you split the area under the curve to more number of bins.

In this activity, we also want to time our computation by using MPI_Wtime() function. We provide you some parts of the code, and would like you to complete **TO DO**, and then you can experiment with the different number of bins you are using. Moreover,  we want you to execute with different number of processes, and compare your timings. I will walk you through the code step by step.

First, you need to initialize the MPI execution environment, define the size of communicator, and define the rank of each process. This should be straight forward for you. You are asked to complete this part.

Then we want to let each process know the number of bins we are using. Therefore, we can broadcast the number of bin to all processes in our MPI_COMM_WORLD. You should use MPI_Bcast to broadcast from master node. You are asked to complete this part.

After that we are ready to ask each process compute their task. This can be done by using the following piece of code: ::

    /* Calculating for each process */
    step = 1.0 / (double) n;
    sum = 0.0;
    for (i = rank + 1; i <= n; i += nprocs) {
        x = step * ((double)i - 0.5);
        sum += (4.0/(1.0 + x*x));
    }

    mypi = step * sum;

When all processes have finished their computations, their results are stored in *mypi*. Therefore, we can reduce all their results into one result. You task is to complete this part by using MPI_Reduce. 

Below is the complete source code for mpi_pi.c[1]:

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

.. centered:: Figure 5: vector matrix multiplication [2]

This multiplication produces a new vector whose length is the number of rows of *A*. The multiplication is very simple, we just need to take a row of *A* dot product with *x*, and this produces an element of result vector. For example, the first row of *A* dot products with *x* will produce the first element in vector *y*. 

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

Below is the entire source code for vector matrix multiplication [3]:    

.. literalinclude:: vector_matrix_buggy.c
	:linenos:


.. rubric:: Footnotes
.. [1] http://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples/simplempi/main.htm
.. [2] http://cms.uni-konstanz.de/fileadmin/informatik/ag-saupe/Webpages/lehre/na_08/Lab1/1_Preliminaries/html/matrixVectorProduct.html
.. [3] http://www.public.asu.edu/~dstanzoi/matvec.c
