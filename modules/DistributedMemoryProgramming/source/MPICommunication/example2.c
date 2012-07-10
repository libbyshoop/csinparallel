#include <stdio.h>
#include "mpi.h"

int main(int argc, char ** argv[]) {

    int rank, ntag = 100;
    char message[12] = "Hello, world";
    
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
    if ( rank == 0 )
        MPI_Send(&message, 12, MPI_CHAR, 1, ntag, MPI_COMM_WORLD);
    else if ( rank == 1 ) {
        MPI_Recv(&message, 12, MPI_CHAR, 0, ntag, MPI_COMM_WORLD, &status);
        printf("Node %d : %s\n", rank, message); 
    }

    MPI_Finalize();
}