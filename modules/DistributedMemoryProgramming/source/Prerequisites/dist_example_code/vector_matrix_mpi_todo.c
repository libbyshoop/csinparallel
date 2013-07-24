/*********************************************************
* Program: Vector Matrix Multiplication
*     The decomposition technique, and the use of MPI_Send
*     are adapted from matrix matrix multiplication by 
*     Blaise Barney. 
**********************************************************/

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
            // Change here to use random integer
            vector[i] = 1;
            for(j = 0; j < WIDTH; j++) {
                // Change here to use random integer
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

        for(dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            printf("Sending %d rows to task %d offset = %d\n", rows, dest, offset);
            // TO DO
            //..............
            // end TO DO
            offset += rows;
        }


        /* Receiving the work from each worker */
        mtype = FROM_WORKER;
        for (i = 1; i <= numworkers; i++) {
             source = i;
             // TO DO
             //..................
             // end TO DO
             printf("Received results from task %d\n", source);
        }


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
        //................
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
        // ..................
        //end TO DO
    }

    /* Terminate the MPI execution environment */
    MPI_Finalize();
    return 0;
}


