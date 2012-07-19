============================
Decomposition and Activities
============================

Decomposition is a very important aspect to optimize the performance of parallel programming models. Decomposition is a way to divide up the task fairly; thus, each task can be distributed to each process. There are many ways to break up a task, and you should choose the way that the best suits your code. For instance, you should split your matrix by row, rather than by column if you want to compute matrix multiplication. Below is an example of decomposition method. You will see this method very often in your activities.

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
        // Then send to each worker the number of rows

:Comments:
    * **rows = (dest <= extra) ? averow + 1 : averow** means if **dest <= extra**, we have **rows = averow + 1**. Otherwise, we have **rows = averow**. This is a shorter version of if and else statement.

    * In this example, **ROW** is the number of rows of the matrix, so each process will get at least **averow** rows. The **extra** is the extra rows when **ROW** is not divisible by number of workers. In order to send each task to each worker, we need to iterate over the number of workers. Then if we have extra rows, we know that number of extra rows must be less than the number of workers, so we can give one more row to workers whose ranks are less than or equal to extra.


Activity 3: Vector Matrix Multiplication Improved Version
---------------------------------------------------------

In this activity, we will be using decomposition technique, MPI_Send, and MPI_Recv to 
improve the efficiency and accuracy of vector matrix multiplication. We already seen that by using 
MPI_Scatter, we do not get the right result if the length of vector is not divisible by
the number of workers. Thus, we want to use the decomposition technique to help us divide 
the task fairly among each worker. Then, we can send each task to each worker by using MPI_Send. 
After the workers having received their tasks, they will compute their own task, and send their results 
back to the master. Finally the master will receive results from workers, and combine them into a result vector.

:Comments: 
    * I will walk you through the code step by step. First, we will need to initialize the MPI environment, define the size of the communicator, and give a rank to each process. This should be straight forward to you because you have seen this many times already. ::

        /* Initialize MPI execution environment */
        MPI_Init( &argc,&argv);
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        MPI_Comm_size( MPI_COMM_WORLD, &nprocs);

    * Then we want to initialize the vector and matrix on master node. This can be done by: ::

        if (rank == 0) {
            /* Initialize Matrix and Vector */
            for(i = 0; i < WIDTH; i++) {
            vector[i] = 1;
                for(j = 0; j < WIDTH; j++) {
                    matrix[i][j] = 1;
                }
            }
        }        

    * We have seen that using the collective communications without decomposition is not the best way of doing this problem. Here is a better way that will work for any number of processes. We will be using the decomposition technique above to split the task for each process. Then, the master will be sending each process the number of rows (**rows**) of matrix, and the vector. You are asked to complete this part of the code. 

    * .. note:: You should use MPI_Send to send the starting rows, and number of rows, and some rows of matrix, and the vector to the workers. Moreover, when you send some rows of matrix, you should use MPI_Send(&matrix[starting row][0], number of elements, ...). This will send the rows of matrix, which contain the number of elements, and it starts from the first element in that row.
    
    * ::

        averow = ROW/numworkers;
        extra = ROW%numworkers;
        offset = 0;
        mtype = FROM_MASTER;

        for (dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            // TO DO
            ..............
            // end TO DO
        }

    * After having sent the messages to all workers, we need to ask workers to receive the messages from the master. We check if the process is not the master, we will use MPI_Recv to receive the messages sent from the master. You are asked complete this task. ::

        if (rank > 0) {
            mtype = FROM_MASTER;
            /* Receive the task from master */
            // TO DO:
            ..............
            // end TO DO
        }    

    * Each worker now can compute their task, and then we have to send the results back to the master. Sending results back to master should not be difficult. Since result is just a vector, we can send the starting rows, number of rows, and entire result vector to the master. You are asked to complete this part. ::

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
        //............
        //end TO DO


    * Finally, the master has to receive the results from all workers. You are asked to complete this task. ::

    	/* Receiving the work from each worker */
        mtype = FROM_WORKER;
        for (i = 1; i <= numworkers; i++) {
            source = i;
            // TO DO
            //..............
            // end TO DO
            printf("Received results from task %d\n", source);
        }


To download the source code to do your activity: 
:download:`download vector_matrix_todo.c <vector_matrix_imp_todo.c>`

To download the entire source code:
:download:`download vector_matrix_done.c <vector_matrix_imp.c>`

Activity 4: Matrix Multiplication
---------------------------------

In this activity, we want you to use decomposition technique, MPI_Send, and MPI_Recv in previous activities to complete the matrix multiplication program. If you have not seen matrix multiplication before, please click on `matrix multiplication <http://mathworld.wolfram.com/MatrixMultiplication.html>`_ to read how matrix multiplication works.

:Comments:

    * This activity is not much different from the previous activity. First, we use the decomposition method in the previous activity. Then we want to send some rows from the first matrix, and entire second matrix to each worker. Note that this is not the most efficient method of doing matrix multiplication because when the second matrix gets really large, we might not be able to send entire matrix to each worker. We use this method because of its simplicity. ::

        /* Computing the row and extra row */
        averow = ROWA/numworkers;
        extra = ROWA%numworkers;
        offset = 0;
        mtype = FROM_MASTER;

        /* Distributing the task to each worker */
        for (dest = 1; dest <= numworkers; dest++) {
            /*If the rank of a process <= extra row, then add one more row to process*/
            rows = (dest <= extra) ? averow+1 : averow;
            printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);
            // TO DO:
            // ............
            // end TO DO
        }

    * Next we want each worker to receive messages sent from the master, and then we can use these matrices to do matrix multiplication on each worker. The result is then stored in matrix **c**. Your task is to receive the messages sent from the master. ::

        if (taskid > MASTER) {
            mtype = FROM_MASTER;

            /* Each worker receive task from master */
            // TO DO
            // ...............
            // end TO DO

            /* Each worker works on their matrix multiplication */
            for (k = 0; k < COLB; k++){
                for (i = 0; i < rows; i++) {
                    c[i][k] = 0.0;
                    for (j = 0; j < COLA; j++)
                        c[i][k] = c[i][k] + a[i][j] * b[j][k];
                }
            }
        }    

    * After each worker has computed the matrix multiplication, all workers have to send the results back to the master. Each worker needs to send their matrix **c** to master. You are asked to complete this task. ::

        /* Each worker sends the output back to master */
        mtype = FROM_WORKER;
        // TO DO
        // ...........
        // end TO DO

    * Then master can receive the results from all workers, and combine them into a single result matrix. You are asked to complete this task. ::

        /* Receive results from worker tasks */
        mtype = FROM_WORKER; /* message comes from workers */
        for (i = 1; i <= numworkers; i++) {
            source = i; /* Specifying where it is coming from */
            // TO DO
            // ...............
            // end TO DO
            printf("Received results from task %d\n",source);
        }

To download the source code to do your activity: 
:download:`download matrix_multiplication_todo.c <matrix_multiplication_todo.c>`

To download the entire source code from computing.llnl.gov [1]:
:download:`download matrix_multiplication.c <matrix_multiplication.c>`

.. rubric:: References
.. [1] https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_mm.c
