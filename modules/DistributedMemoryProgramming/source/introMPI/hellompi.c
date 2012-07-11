#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[]) {
	int rank, nprocs;

	MPI_Init (&argc, &argv);	/* creates MPI execution environment */
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* get current process rank */
	MPI_Comm_size (MPI_COMM_WORLD, &nprocs);/* get number of processes */
	printf("Hello world from process %d of %d\n", rank, nprocs);
	MPI_Finalize();			/* terminates the MPI execution environment */
	return 0;
}