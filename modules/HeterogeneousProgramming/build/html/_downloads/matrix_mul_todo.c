/*******************************************************************
* File: mpi.c (The Heterogeneous cuda and mpi)
* Description:
*       This program is the hybrid of cuda and mpi to calculate
*       matrix multiplication. This is very similar to matrix
*       multiplication mpi version. But it has to call function
*       from cuda.cu and then each node uses that function for each
*       computation.
* NOTE: This code is a modified version of mpi_mm.c by Blaise Barney.
*******************************************************************/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define ROW_A 32               /* number of rows in matrix A */
#define COL_A 12               /* number of columns in matrix A */
#define COL_B 32               /* number of columns in matrix B */
#define MASTER 0               /* taskid of the master node */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */
#define WIDTH 32               /* width of the matrix */

/* declaring function from cuda*/
void MatrixMul(float *dM, float *dN, float *dP, int width, int block_size);

int main (int argc, char *argv[]) {

    int numtasks,              /* number of tasks in partition */
        rank,                  /* a task identifier */
        numworkers,            /* number of worker tasks */
        source,                /* task id of message source */
        dest,                  /* task id of message destination */
        mtype,                 /* message type */
        rows,                  /* rows of matrix A sent to each worker */
        averow, extra, offset, /* used to determine rows sent to each worker */
        i, j, k, rc;           /* variables for loops */

    float a[ROW_A][COL_A],           /* matrix A for multiplication */
          b[COL_A][COL_B],           /* matrix B for multiplication */
          c[ROW_A][COL_B];           /* result matrix C */

    MPI_Status status;         /* the status for receiving the result */


    /*Initializing MPI execution environment*/
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    numworkers = numtasks - 1;    /* number of workers */

    /**************************** master task ************************************/
    if (taskid == MASTER) {

        /* Initializing both matrices on master node */
        for (i = 0; i < ROW_A; i++)
            for (j = 0; j < COL_A; j++)
                a[i][j]= 1;
        for (i = 0; i < COL_A; i++)
            for (j = 0; j < COL_B; j++)
                b[i][j]= 1;

        /* Computing the average row and extra row for each process */
        averow = ROW_A/numworkers;
        extra = ROW_A%numworkers;
        offset = 0;
        mtype = FROM_MASTER;

        /* Distributing the task to each worker */
        for (dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow+1 : averow;
            printf("Sending %d rows to task %d offset = %d\n", rows, dest, offset);
            
            // TO DO
            // send some rows of first matrix and entire second matrix to each worker
            // end TO DO

            offset = offset + rows;
        }

        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*COL_B, MPI_FLOAT, source, mtype, MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n",source);
        }

        /* Master prints results */
        printf("******************************************************\n");
        printf("Result Matrix:\n");
        for (i = 0; i < ROW_A; i++) {
            printf("\n");
            for (j = 0; j < COL_B; j++)
                 printf("%6.2f   ", c[i][j]);
        }
        printf("\n******************************************************\n");
        printf ("Done.\n");
    }

    /**************************** worker task ************************************/
    if (taskid > MASTER) {

        /* Each receives task from master*/
        mtype = FROM_MASTER;
        
        // TO DO
        // receive the matrices sent from master
        // end TO DO

        /* Calling function from CUDA. Each worker computes on their GPU */
        // TO DO
        // call CUDA function 
        // end TO DO

        /* Each worker sends result back to the master */
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c, rows*COL_B, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
     }

     /* Terminate MPI execution environment */
     MPI_Finalize();
}
