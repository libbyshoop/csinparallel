#include <stdio.h>
#include "mpi.h"

int main(int argc, char ** argv[]) {

    int rank, ntag = 100;
    char message[12] = "Hello, world";
    
    /* status for MPI_Recv */
    MPI_Status status;
    
    /* Initialize MPI execution environment */
    MPI_Init(&argc, &argv);
    /* Give each process a unique rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
    /* If the process is the master */
    if ( rank == 0 )
        /* Send message to process whose rank is 1 in the MPI_COMM_WORLD */
        MPI_Send(&message, 12, MPI_CHAR, 1, ntag, MPI_COMM_WORLD);

    /* If the process has rank of 1 */
    else if ( rank == 1 ) {
        /* Receive message sent from master */
        MPI_Recv(&message, 12, MPI_CHAR, 0, ntag, MPI_COMM_WORLD, &status);
        /* Print the rank and message */
        printf("Node %d : %s\n", rank, message); 
    }
    
    /* Terminate MPI execution environment */
    MPI_Finalize();
}