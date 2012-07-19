/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 04/13/05
******************************************************************************/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define ROWA 10                 /* number of rows in matrix A */
#define COLA 10                 /* number of columns in matrix A */
#define COLB 10                 /* number of columns in matrix B */
#define MASTER 0                /* taskid of first task */
#define FROM_MASTER 1           /* setting a message type */
#define FROM_WORKER 2           /* setting a message type */

int main (int argc, char *argv[]) {

    int numtasks,              /* number of tasks in partition */
        taskid,                /* a task identifier */
        numworkers,            /* number of worker tasks */
        source,                /* task id of message source */
        dest,                  /* task id of message destination */
        mtype,                 /* message type */
        rows,                  /* rows of matrix A sent to each worker */
		averow, extra, offset, /* used to determine rows sent to each worker */
        i, j, k, rc;           /* misc */

    double      a[ROWA][COLA],           /* matrix A to be multiplied */
                b[COLA][COLB],           /* matrix B to be multiplied */
                c[ROWA][COLB];           /* result matrix C */

    MPI_Status status; /* status for receiving */


    /* Initializing MPI execution environment */
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    /* Need at least two processes, a master and a worker */
    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    /* if the number of processes is less than number of nodes,
    * master node will not do the multiplication. If the number
    * of processes is more than the number of nodes, then
    * master will take part in the multiplication.
    */
    numworkers = numtasks - 1;

    /**************************** master task ************************************/
    if (taskid == MASTER) {

        /* Initializing the matrices A and B */
        for (i = 0; i < ROWA; i++) {
            for (j = 0; j < COLA; j++) {
                a[i][j]= 1;
            }
        }
        for (i = 0; i < COLA; i++) {
            for (j = 0; j < COLB; j++) {
                b[i][j]= 1;
            }
        }

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
            /* Send value offset to all processes */
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD); 
            /* Send value of rows to all processes */
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);   
            /* Sending some rows of matrix A to some processes. Offset is the starting row, 
            * and it starts at 0 column.*/
            MPI_Send(&a[offset][0], rows*COLA, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            /* Send entire B matrix to all processes*/
            MPI_Send(&b, COLA*COLB, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD); 
            /* the first process gets row 0 to some rows, and so on */
            offset = offset + rows; 
        }

        /* Receive results from worker tasks */
        mtype = FROM_WORKER; /* message comes from workers */
        for (i = 1; i <= numworkers; i++) {
            source = i; /* Specifying where it is coming from */
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows*COLB, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n",source);
        }

        /* Printing results */
        printf("******************************************************\n");
        printf("Result Matrix:\n");
        for (i = 0; i < ROWA; i++) {
            printf("\n");
            for (j = 0; j < COLB; j++)
                printf("%6.2f   ", c[i][j]);
        }
        printf("\n******************************************************\n");
        printf ("Done.\n");
    }

    /**************************** worker task ************************************/

    /* Each worker receives task from master, and work on their computation, and
     * send their outputs back to master.
     */

     if (taskid > MASTER) {
        mtype = FROM_MASTER;

        /* Each worker receive task from master */
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows*COLA, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, COLA*COLB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

        /* Each worker works on their matrix multiplication */
        for (k = 0; k < COLB; k++){
           for (i = 0; i < rows; i++) {
               c[i][k] = 0.0;
               for (j = 0; j < COLA; j++)
                   c[i][k] = c[i][k] + a[i][j] * b[j][k];
           }
        }

        /* Each worker sends the output back to master */
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c, rows*COLB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }
    /* Terminate MPI environment */
    MPI_Finalize();
}




