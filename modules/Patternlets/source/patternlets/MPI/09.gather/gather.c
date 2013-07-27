/* gather.c
 * ... illustrates the use of MPI_Gather()...
 * Joel Adams, Calvin College, November 2009.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX 3

int main(int argc, char** argv) {
   int computeArray[MAX];
   int* gatherArray = NULL;
   int i, numProcs, myRank, totalGatheredVals;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   for (i = 0; i < MAX; i++) {
      computeArray[i] = myRank * 10 + i;
   }
     
   totalGatheredVals = MAX * numProcs;
   gatherArray = (int*) malloc( totalGatheredVals * sizeof(int) );

   MPI_Gather(computeArray, MAX, MPI_INT,
               gatherArray, MAX, MPI_INT, 0, MPI_COMM_WORLD);

   if (myRank == 0) {
      for (i = 0; i < totalGatheredVals; i++) {
         printf(" %d", gatherArray[i]);
      }
      printf("\n");
   }

   free(gatherArray);
   MPI_Finalize();

   return 0;
}

