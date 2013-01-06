/* ring.c
 * ... illustrates the use of MPI_Send() and MPI_Recv()...
 * Joel Adams, Calvin College, at November 2009.
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
	int id = -1, numProcesses = -1, receivedID = -1, tag = 0;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	if (id == 0) {
		MPI_Send( &id, 1, MPI_INT, id+1, tag, MPI_COMM_WORLD);
		MPI_Recv( &receivedID, 1, MPI_INT, numProcesses-1, 
                            tag, MPI_COMM_WORLD, &status);
		printf("%d - message successfully sent around ring of %d processes\n",
			id, numProcesses);
	} else {
		MPI_Recv( &receivedID, 1, MPI_INT, id-1, 
                            tag, MPI_COMM_WORLD, &status);
		MPI_Send( &id, 1, MPI_INT, (id+1) % numProcesses, tag, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}

