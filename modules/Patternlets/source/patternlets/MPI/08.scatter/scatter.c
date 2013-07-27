/* scatter.c
 * ... illustrates the use of MPI_Scatter()...
 * Joel Adams, Calvin College, November 2009.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
        const int MAX = 8;
	int aSend[] = {11, 22, 33, 44, 55, 66, 77, 88};
	int* aRcv;
        int i, numProcs, myRank, numSent;

	MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
     
        numSent = MAX / numProcs;
	aRcv = (int*) malloc( numSent * sizeof(int) );
        MPI_Scatter(aSend, numSent, MPI_INT, aRcv, numSent, MPI_INT, 0, MPI_COMM_WORLD);
	printf("Process %d: ", myRank);
	for (i = 0; i < numSent; i++) {
		printf(" %d", aRcv[i]);
	}
	printf("\n");

	free(aRcv);
 	MPI_Finalize();

	return 0;
}

