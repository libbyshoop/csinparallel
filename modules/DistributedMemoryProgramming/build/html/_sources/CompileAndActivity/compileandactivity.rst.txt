==================================
Compiling and First Two Activites
==================================

.. highlight:: c

Compiling an MPI program
------------------------

To compile an MPI program on your local cluster, you can enter the following commands in the terminal:

First, we need to make an executable file from the MPI program by using **mpicc**:
::

    mpicc -o filename filename.c

Then you are able to execute it using **mpirun** :
::

    mpirun -machinefile machines -np #processes ./filename

.. note::
	**machines**: is the instruction for running the executable file on a cluster. It tells the executable file to run on which nodes and how many times on those nodes. For example, machines on LittleFe has structure as follow:

        - node000.bccd.net    slots = 1
        - node011.bccd.net    slots = 1
        - node012.bccd.net    slots = 1
        - node013.bccd.net    slots = 1
        - node014.bccd.net    slots = 1
        - node015.bccd.net    slots = 1

Moreover, you can also compile an MPI program without using **machines**, you can use the following command to run only on the master node:
::

    mpirun -np #processes ./filename

.. note:: Please ask your instructor for instructions on how to log in onto your local machine.

Try the simple examples
------------------------

Log into your cluster and make sure that you have the following example files in your account:

Simplest Hello World Example:
:download:`download hellompi.c <hellompi.c>`

Hello World by Sending messages:
:download:`download example2.c <example2.c>`

You can specifically compile the simplest example like this: ::

    mpicc -o hellompi hellompi.c

To run it on your cluster using 3 processes, you can do this, assuming you also have a
file called machines that describes your system: ::

    mpirun -machinefile machines -np 3 ./hellompi

Depending on how many other people are using your cluster and how big it is, you can try
different numbers of processes by changing the 3 above.

Once that seems to be working for you, try the example 2, which uses sends and receives.

.. note::

    The procedure for compiling and running all of your mpi code will follow this format.



Activity 1: Computing :math:`{\pi}` ?
---------------------------------------

Download the source code to do this activity (originally from www.mcs.anl.gov [1]):
:download: :download:`download seq_pi_done.c <seq_pi_done.c>` and `download mpi_pi_done.c <mpi_pi_done.c>`

Have it open in an editor so that you can follow along.  Or at a minimum use your browser and right-click on the link above and choose to open it in a new browser window.

In this activity, we are going to analyze speedup and efficiency of a program that
computes :math:`{\pi}` using integration. We have formula:

.. math::

    \int_0^1 \frac{4}{1 + x^2} dx = {\pi}

Therefore, we can compute the area under the curve to get the value of the integral.

.. image:: images/graph.png
    :width: 300px
    :align: center
    :height: 150px
    :alt: graph

.. centered:: Figure 4: Graph for function

:Comments:

    * We can split the area under the curve into bins. The idea is to group the bins into smaller chunks so we can use each process to calculate each chunk, and then combine the result into one value. Remember, we can get a more accurate result if you split the area under the curve into more number of bins. We will walk you through the code step by step.

    * First, we initialize the MPI execution environment, define the size of communicator, and define the rank of each process.

    * Then we let each process know the **number of bins** we are using. Therefore, we broadcast the **number of bins** to all processes in our MPI_COMM_WORLD.

    * Now we are ready to ask each process compute their task. We want to evaluate the integral of :math:`\frac {4}{1 + x^2}` from *0* to *1*, and we can do so by finding the sum of all bins from *0* to *1*. Each bin is approximately :math:`\frac {1}{n} * \frac {4}{1 + x^2}` (**n** is the number of bins). We are iterating over the number of bins, and we start from *0*; therefore, to find the center of each bin, we need to add *+ 0.5* to variable *i*. Moreover, in the **for loop**, we ask the rank *0* to compute the first bin, the (nprocs) bin, and so on, rank *1* to compute the second bin, the (nprocs + 1) bin, and so on, ..., as long as value of *i* is less than **n**. Suppose that there are *p* processes, then these *p* processes will take the first *p* bins, where process whose rank is 0 takes *1st* bin, process whose rank is 1 takes *2nd* bin, and so on. If there are any bins left, process whose rank is 1 takes *pth* bin, and so on. This can be done by using the following piece of code: ::

        /* Calculating for each process */
        step = 1.0 / (double) n;
        sum = 0.0;
        for (i = rank; i < n; i += nprocs) {
            x = step * ((double)i + 0.5);
            sum += (4.0/(1.0 + x*x));
        }

        mypi = step * sum;

    * When all processes have finished their computations, their results are stored in **mypi**. Therefore, we can reduce all their results into one result, which is the value of :math:`{\pi}`.

.. topic:: To do:

	In this activity, we want to time the computation. This is done using
	the MPI_Wtime() function. You can experiment using different number of
	bins and different number of processes.

	We want to time several combinations of bin sizes and number of processes
	to calculate the speedup and efficiency of the program. Compile and run the
	sequential program using 125 million, 250 million, 500 million, 1 billion and
	2 billion for the number of bins. Make a copy of the template provided at
	`this link <https://docs.google.com/spreadsheets/d/1ff1yFkz4cMheYPaZIiA29J_GzCFNW4tKAMoDjPQu130/edit?usp=sharing.>`_ and
	record the execution times.

	Next, compile and run the MPI program using 2, 4, 8, 12 and 16 processes
	with the number of bins listed above. Continue to record these execution times
	in same table. The speedup and efficiency for each combination is computed for
	you. You should now see graphs corresponding to speedup and efficiency. At what
	number of threads do you notice both speedup and efficiency begin to decrease?
	What does this tell us about the number of processes we should use for this
	parallel program?

Activity 2: Vector Matrix Multiplication
----------------------------------------

To download the source code to do this activity:
:download:`download vector_matrix_todo.c <vector_matrix_buggy_todo.c>`


Have it open in an editor so that you can work on it.  Or at a minimum use your browser and right-click on the link above and choose to open it in a new browser window.

In this activity, we are going to compute vector matrix multiplication. This activity illustrates the use of MPI_Bcast, MPI_Scatter, and MPI_Gather to do this multiplication. First, we want you to complete this MPI program by filling codes at **TO DO**. After having completed this task, try to run this MPI program by using different number of processes. Try to explain to yourself what is happening !

Here is how the vector matrix multiplication works. First, let's say we have a matrix *A*, and a vector *x* as below:

.. image:: images/vector_matrix_multi.png
	:width: 500px
	:align: center
	:height: 300px
	:alt: MPI Structure

.. centered:: Figure 5: vector matrix multiplication Obtained from cms.uni-konstanz.de [2]

This multiplication produces a new vector whose length is the number of rows of matrix *A*. The multiplication is very simple. We take a row of matrix *A* dot product with vector *x*, and this produces an element of the result vector. For instance, the first row of matrix *A* dot products with vector *x* will produce the first element in vector *y*.

:Comments:

    * We will step you through this activity step by step. Since this is an MPI program, we need to create the MPI execution environment, define the size of the communicator, and give each process a unique rank. You are asked to completed this part of the code.

    * After having initialized the MPI environment, we want to ask the master to initialize the vector and matrix we are going to multiply. In order to do that, we check if the process is master. If so, we initialize the matrix and vector. We initialize every entry to 1 because of its simplicity. ::

        if (rank == 0) {
            /* Initialize the matrix and vector */
            for(i=0; i < WIDTH; i++) {
                vector[i] = 1;
                for(j = 0; j < WIDTH; j++) {
                    matrix[i][j] = 1;
                }
            }
        }

    * Since the vector is not very large and all processes must have this vector to do the multiplication, we will broadcast the entire vector to all processes. We do this by using MPI_Bcast. In addition, we want to distribute the matrix to each process in the MPI_COMM_WORLD. We would do this using MPI_Scatter. You are asked to complete this task.

    * When all processes can see the vector and some rows of matrix, they are now able to do the multiplication. We need to store their result in the result vector. ::

        for(i = 0; i < chunk_size; i++) {
            result[i] = 0;
            for(j = 0; j < WIDTH; j++) {
                result[i] += local_matrix[i][j] * vector[j];
            }
        }

    * The last part you need to complete is to gather all result vectors in all processes, and store them in the output vector, called global_result vector. This will be our result vector. Moreover, we can print out the value of each element in the global_result vector, and then terminate the MPI execution environment.


If you get stuck and wanto move on, download the entire source code (originally from www.public.asu.edu/~dstanzoi [3]):
:download:`download vector_matrix_done.c <vector_matrix_buggy_done.c>`


.. rubric:: References

.. [1] http://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples/simplempi/main.htm
.. [2] http://cms.uni-konstanz.de/fileadmin/informatik/ag-saupe/Webpages/lehre/na_08/Lab1/1_Preliminaries/html/matrixVectorProduct.html
.. [3] http://www.public.asu.edu/~dstanzoi/matvec.c
