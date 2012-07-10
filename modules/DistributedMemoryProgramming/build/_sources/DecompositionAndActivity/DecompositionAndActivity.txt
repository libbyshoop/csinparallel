============================
Decomposition and Activities
============================

Decomposition is a very important aspect to optimize the performance of MPI. Decomposition is a way to divide up the task fairly; thus, each task can be distributed to each process. There are many ways to break up a task, and you should choose the way that the best suits your code. For instance, you might want to split your matrix by row, rather than by column if you want to compute matrix multiplication. You often will see this decomposition technique in your activities.

**Example 3** : Decompose the matrix by row

.. highlight:: c

::

      averow = ROW_A/numworkers;
      extra = ROW_A%numworkers;
      offset = 0;
      mtype = FROM_MASTER;

      for (dest = 1; dest <= numworkers; dest++) {
          rows = (dest <= extra) ? averow + 1 : averow;
          Then send to each worker the number of rows


Activity 3: Vector Matrix Multiplication Improved Version
---------------------------------------------------------

In this activity, we will be using decomposition technique, MPI_Send, and MPI_Recv to 
improve the efficiency of vector matrix multiplication. We already seen that by using 
MPI_Scatter, we do not get the right result if the length of vector is not divisible by
the number of workers. Thus, we want to use the decomposition technique to help us divide 
the task fairly among each worker. Then, we can send each task to each worker via MPI_Send. 
After the workers having received their tasks, they will compute each task, and send their results 
back to master, and master will check if they are right.

.. literalinclude:: vector_matrix_imp.c
	:linenos:


Activity 4: Matrix Multiplication
---------------------------------

In this activity, we want you to use the techniques you have since in the previous activities to 
complete the matrix multiplication program. 

.. literalinclude:: matrix_multiplication.c
	:linenos: