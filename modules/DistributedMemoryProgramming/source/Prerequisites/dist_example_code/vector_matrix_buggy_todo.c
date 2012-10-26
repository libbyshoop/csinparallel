/*****************************************************************************
* This activity illustrates the use of MPI_Bcast, MPI_Scatter, and MPI_Gather. 
******************************************************************************/

#include "mpi.h"
#include <stdio.h>

#define WIDTH 10        /* the size of the vector */

int main(int argc, char *argv[]) {

    int nprocs,         /* number of processes */
        rank,           /* rank of each process */
        chunk_size,     /* number of rows to be sent to a process */
        i, j;

    int matrix[WIDTH][WIDTH];           /* matrix for multiplication */
    int vector[WIDTH];                  /* vector for multiplication */
    int local_matrix[WIDTH][WIDTH];     /* for storing piece of matrix in each process */
    int result[WIDTH];                  /* for storing result in each process */
    int global_result[WIDTH];           /* vector result after all calculations */

    MPI_Status status;                  /* status for receiving */

    /* Initialize MPI execution environment */
    // TO DO:
    // ...............
    // end TO DO
    
    printf("Hello from process %d of %d \n",rank,nprocs);

    chunk_size = WIDTH/nprocs;          /* number of rows to be sent to each process */

    /* master doing part of the work here */
    if (rank == 0) {

        /* Initialize Matrix and Vector */
        for(i=0; i < WIDTH; i++) {
            // Change here if you want to use random integer
            vector[i] = 1;
            for(j = 0; j < WIDTH; j++) {
                // Change here if you want to use random integer
                matrix[i][j] = 1;
            }
        }
    }

    /* Distribute Vector */
    /* All processes need the input vector for multiplication, so we can broadcast it to all processes */
    // TO DO:
    // ............
    // end TO DO

    /* Distribute Matrix */
    /* We can broadcast entire matrix to all processes, but matrix might be too big,
     * and it is not an efficient method. Instead we will split matrix by row.
     * Scatter will do that for us. It will take the matrix, and send WIDTH * chunk_size
     * to each process, and each piece get stored in local_matrix.
     */
    // TO DO: 
    // ..............
    // end TO DO

    /* Each processor has some rows of matrix, and the vector. Each process works on their multiplication
     * and storing the result in result vector.
     */
     for(i = 0; i < chunk_size; i++) {
        result[i] = 0;
        for(j = 0; j < WIDTH; j++) {
            result[i] += local_matrix[i][j] * vector[j];
        }
    }

    /* Each process sends result back to master, and get stored in global_result */
    // TO DO:
    // ..............
    // end TO DO

    /* master prints elements of the result */
    if(rank == 0) {
        for(i = 0; i < WIDTH; i++) {
            printf("%d\n", global_result[i]);
        }
    }
    /* Terminate MPI execution environment */
    MPI_Finalize();
    return 0;
}

     

