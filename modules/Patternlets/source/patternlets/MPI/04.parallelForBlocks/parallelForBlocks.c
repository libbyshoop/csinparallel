/* parallelForBlocks.c
 * ... illustrates the parallel for loop pattern in MPI 
 *	in which processes perform the loop's iterations in 'blocks' 
 *	(preferable when loop iterations do access memory/cache locations) ...
 * Joel Adams, Calvin College, at November 2009.
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
	const int ITERS = 16;
	int id = -1, numProcesses = -1, i = -1,
		start = -1, stop = -1, blockSize = -1;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

	blockSize = ITERS / numProcesses;     // integer division
	start = id * blockSize;               // find starting index
                                              // find stopping index
	if ( id < numProcesses - 1 ) {        // if not the last process
		stop = (id + 1) * blockSize;  //  stop where next process starts
	} else {                              // else
		stop = ITERS;                 //  last process does leftovers
	}

	for (i = start; i < stop; i++) {      // iterate through your range
		printf("Process %d is performing iteration %d\n",
			id, i);
	}

	MPI_Finalize();
	return 0;
}

