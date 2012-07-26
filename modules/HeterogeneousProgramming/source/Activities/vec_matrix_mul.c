#include "mpi.h"
#include <stdio.h>

#define COLS 24         /* number of columns of the matrix */
#define ROWS 24         /* number of rows of the matrix */
#define MASTER 0        /* master has rank 0 */
#define FROM_MASTER 1   /* message sent from master */
#define FROM_WORKER 2   /* message sent from worker */
#define BLOCK_SIZE 12   /* number of threads in a block */
#define MAX_NAME 80     /* length of character for naming a node */

/* Declaring CUDA function for vector-matrix multiplication */
void run_kernel(int *A, int *x, int *y, int width, int block_size);

int main(int argc, char *argv[] ) {

    int nprocs,         /* number of processes */
        rank,           /* rank of a process */
        len, i, j;

    int averow,         /* average rows to be sent to each worker */
        extra,          /* extra rows to be sent to some workers */
        offset,         /* starting position of row to be sent */
        dest,           /* rank of the destination process */
        numworkers,     /* number of workers */
        mtype,          /* message type */
        source,         /* rank of the source process */
        rows;           /* number of rows to be sent to a worker */

    int matrix[ROWS][COLS];     /* matrix for the multiplication */
    int vector[ROWS];           /* vector for the multiplication */
    int result[ROWS];           /* output vector */

    char name[MAX_NAME];        /* char array for storing name of a process */

    MPI_Status status;          /* status for receiving message from MPI_Send */

    /* Initialize MPI execution environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Get_processor_name(name, &len);

    /******************************* Master ***************************/
    if (rank == 0) {
    /* Initialize Matrix and Vector */
        for(i = 0; i < ROWS; i++) {
            // change here to use random integer
            vector[i] = 1;
            for(j = 0; j < COLS; j++) {
                // change here to use random integer
                matrix[i][j] = 1;
            }
        }

        numworkers = nprocs - 1;

        /* divide the number of rows for each worker */
        averow = ROWS/numworkers;
        extra = ROWS%numworkers;
        offset = 0;
        mtype = FROM_MASTER;

        /* Master sends smaller task to each worker */
        for(dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&matrix[offset][0], rows * COLS, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&vector, ROWS, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            printf("Master sent elements %d to %d to rank %d\n", offset, offset + rows, dest);
            offset += rows;
        }

        /* Master receives the output from each worker*/
        mtype = FROM_WORKER;
        for (i = 1; i <= numworkers; i++) {
             source = i;
             MPI_Recv(&offset, 1, MPI_INT, source,mtype, MPI_COMM_WORLD, &status);
             MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
             MPI_Recv(&result[offset], rows, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
             printf("Received results from task %d\n", source);
        }

        /* Master prints results */
        for (i = 0; i < ROWS; i++) {
            printf("The element of output vector is: %d\n", result[i]);
        }
    }

    /************************************** Workers *************************************/
    if (rank > 0) {
        mtype = FROM_MASTER;
        /* Each worker receives messages sent from the master*/
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix, rows*COLS, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&vector, ROWS, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        printf("Worker rank %d, %s receives the messages\n", rank, name);

        /* use CUDA function to compute the the vector-matrix multiplication for each worker */
        run_kernel(*matrix, vector, result, ROWS, BLOCK_SIZE);

        /* Each worker sends the result back to the master */
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&result, rows, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        printf("Worker rank %d, %s sends the result to master \n", rank, name);
    }

    /* Terminate the MPI execution environment */
    MPI_Finalize();
    return 0;
}
