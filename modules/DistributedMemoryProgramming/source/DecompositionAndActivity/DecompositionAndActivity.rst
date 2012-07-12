============================
Decomposition and Activities
============================

Decomposition is a very important aspect to optimize the performance of parallel programmind models. Decomposition is a way to divide up the task fairly; thus, each task can be distributed to each process. There are many ways to break up a task, and you should choose the way that the best suits your code. For instance, you might want to split your matrix by row, rather than by column if you want to compute matrix multiplication. You often will see this decomposition technique in your activities.

Example 3: Decompose the matrix by row
**************************************
.. highlight:: c

::

      averow = ROW/numworkers;
      extra = ROW%numworkers;
      offset = 0;
      mtype = FROM_MASTER;

      for (dest = 1; dest <= numworkers; dest++) {
          rows = (dest <= extra) ? averow + 1 : averow;
          Then send to each worker the number of rows

.. note:: **rows = (dest <= extra) ? averow + 1 : averow** means if **dest <= extra**, we have **rows = averow + 1**. Otherwise, we have **rows = averow**. This is a shorter version of if and else statement.

In this example, **ROW** is the number of rows of matrix, so each process will get at least **averow** rows. The **extra** is the extra rows when **ROW** is not divisible by number of workers. In order to send each task to each worker, we need to iterate over the number of workers. Then if we have extra rows, we know that number of extra rows must be less than the number of workers, so we can give one more row to workers whose ranks are less than extra.


Activity 3: Vector Matrix Multiplication Improved Version
---------------------------------------------------------

In this activity, we will be using decomposition technique, MPI_Send, and MPI_Recv to 
improve the efficiency and accuracy of vector matrix multiplication. We already seen that by using 
MPI_Scatter, we do not get the right result if the length of vector is not divisible by
the number of workers. Thus, we want to use the decomposition technique to help us divide 
the task fairly among each worker. Then, we can send each task to each worker by using MPI_Send. 
After the workers having received their tasks, they will compute each task, and send their results 
back to master, and master will check if they are right.

I will walk you through the code step by step. First, we will need to initialize the MPI execution 
environment. This should be straight forward to you because you have seen this many times already. ::

	/* Initialize MPI execution environment */
    MPI_Init( &argc,&argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs);

Then we want to initialize the vector and matrix in master node. This can be done by: ::

	if (rank == 0) {
    /* Initialize Matrix and Vector */
        for(i=0; i < WIDTH; i++) {
            vector[i] = 1;
            for(j = 0; j < WIDTH; j++) {
                matrix[i][j] = 1;
            }
        }
    }        

Since we have seen that using the collective communication without decomposition is not the best way 
of doing this problem. Here is better way that will work for any number of processes. We will be using 
the decomposition technique above to split the task for each process. Then, we will be sending each process 
the number of rows (rows) of matrix, and send the vector to all processes. You are asked to complete this part 
of the code. 

.. note:: You should use MPI_Send to send the starting rows, and number of rows, and part of matrix, and the vector to the processes whose ranks are less than number of workers. Moreover, when you send part of matrix, you should use MPI_Send(&matrix[starting row][0], number of elements, ...). This will send the rows of matrix, which contain the number of elements.

::

	averow = ROW/numworkers;
    	extra = ROW%numworkers;
    	offset = 0;
    	mtype = FROM_MASTER;

    	for (dest = 1; dest <= numworkers; dest++) {
       		rows = (dest <= extra) ? averow + 1 : averow;
        	// TO DO

After having sent the messages to all workers, we need to ask the workers to receive what being sent from the master. We need to check if the process is not the master, we should set the receive function, MPI_Recv. You are asked complete this part. ::

	if (rank > 0) {
        mtype = FROM_MASTER;

        /* Receive the task from master */
        // TO DO:
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix, rows*WIDTH, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&vector, WIDTH, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        // end TO DO

Then each worker will compute their part of calculation, and we need to send the result back to the master. You are asked to complete this part. ::

	/* Each worker works on their computation */
        for(i = 0; i < rows; i++) {
            result[i] = 0;
            for(j = 0; j < WIDTH; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }

        /* send the result back to the master */
        mtype = FROM_WORKER;
        // TO DO:
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&result, rows, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        //end TO DO


The last part is you need to receive the results sent from worker in the master node. You are asked to complete this part. ::

	/* Receiving the work from each worker */
        mtype = FROM_WORKER;
        //TO DO:
        for (i = 1; i <= numworkers; i++) {
             source = i;
             MPI_Recv(&offset, 1, MPI_INT, source,mtype, MPI_COMM_WORLD, &status);
             MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
             MPI_Recv(&result[offset], rows, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
             printf("Received results from task %d\n", source);
        }
        //end TO DO

Below is the complete source code:

.. literalinclude:: vector_matrix_imp.c
	:linenos:


Activity 4: Matrix Multiplication
---------------------------------

In this activity, we want you to use the techniques you have since in the previous activities to 
complete the matrix multiplication program. Please fill your code at TO DO.

.. literalinclude:: matrix_multiplication.c
	:linenos: