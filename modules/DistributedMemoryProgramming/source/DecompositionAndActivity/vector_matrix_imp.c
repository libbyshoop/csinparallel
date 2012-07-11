#include "mpi.h"
#include <stdio.h>

#define WIDTH 10          /* the size of vector */
#define FROM_MASTER 1     /* message tag */
#define FROM_WORKER 2     /* message tag */
#define ROW 10            /* number of rows of matrix */
#define MASTER 0          /* master has rank of 0 */

int main(int argc, char *argv[] ) {

    int nprocs,         /* number of processes in MPI_COMM_WORLD */
        rank,           /* the rank of each process */
        i, j;

    int averow,         /* average rows to be sent to each worker */
        extra,          /* extra rows to be sent to some workers */
        offset,
        dest,           /* rank of the destination process */
        numworkers,     /* storing number of workers */
        mtype,          /* message type */
        source,         /* rank of the source process */
        rows;           /* rows to be sent to each process */

    int matrix[WIDTH][WIDTH];   /* matrix for multiplication */
    int vector[WIDTH];          /* vector for multiplication */
    int result[WIDTH];          /* vector we get after multiplication */

    MPI_Status status;          /* status for receiving */

    /* Initialize MPI execution environment */
    MPI_Init( &argc,&argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs);

    printf("Hello from process %d of %d \n",rank,nprocs);

    /************************* Master ************************/
    if (rank == 0) {
    /* Initialize Matrix and Vector */
        for(i=0; i < WIDTH; i++) {
            vector[i] = 1;
            for(j = 0; j < WIDTH; j++) {
                matrix[i][j] = 1;
            }
        }
         /* number of workers in the communicator MPI_COMM_WORLD */
        numworkers = nprocs - 1;       

        /* decomposing the rows of matrix for each worker */
        averow = ROW/numworkers;
        extra = ROW%numworkers;
        offset = 0;
        mtype = FROM_MASTER;

        /* sending each task to each worker, some workers will get more others
         * if number of rows is not divisible by number of processes
         */

         // TO DO:
        for(dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            printf("Sending %d rows to task %d offset = %d\n", rows, dest, offset);
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&matrix[offset][0], rows * WIDTH, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&vector, WIDTH, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            offset += rows;
        }
        // end TO DO

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

        /* Printing results */
        for (i = 0; i < WIDTH; i++) {
            printf("%d\n ", result[i]);
        }
    }

    /* the workers receive task from master, and will do the computations */
    if (rank > 0) {
        mtype = FROM_MASTER;

        /* Receive the task from master */
        // TO DO:
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix, rows*WIDTH, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&vector, WIDTH, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        // end TO DO

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
    }

    /* Terminate the MPI execution environment */
    MPI_Finalize();
    return 0;
}


