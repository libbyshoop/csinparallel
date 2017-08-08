/*
 * Hannah Sonsalla, Macalester College, 2017
 *
 *  calcPiHybrid.C
 *
 *   ...program uses MPI and OpenMP to calculate the value of Pi
 *
 * Usage:  mpirun -np N ./calcPiHybrid <number of threads> <number of tosses>
 *
 */

#include <mpi.h>     // MPI commands
#include <omp.h>     // OpenMP commands
#include <math.h>    // fabs
#include <stdio.h>   // printf()
#include <stdlib.h>  // atoi()

void Get_input(int argc, char* argv[], int myRank, int* numThreads_p, long* totalNumTosses_p);
long Toss(long numProcessThreadTosses, int myRank, int threadID);

int main(int argc, char** argv) {
  int myRank, numProcs, numThreads, length;
  long totalNumTosses, localNumTosses, processThreadNumberInCircle = 0, totalNumberInCircle;
  double start, finish, loc_elapsed, elapsed, piEstimate;
  double PI25DT = 3.141592653589793238462643;         /* 25-digit-PI*/
  char hostName[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Get_processor_name (hostName, &length);

  Get_input(argc, argv, myRank, &numThreads, &totalNumTosses);  // Read total number of tosses from command line

  localNumTosses = totalNumTosses/(numProcs * numThreads);
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  #pragma omp parallel num_threads(numThreads) reduction(+:processThreadNumberInCircle)
  {
    int threadID = omp_get_thread_num();
    processThreadNumberInCircle = Toss(localNumTosses,myRank, threadID);
    printf("Thread %d of %d from process %d of %d on %s has %ld in circle\n", threadID, numThreads,
    myRank, numProcs, hostName, processThreadNumberInCircle);
  }

  finish = MPI_Wtime();
  loc_elapsed = finish-start;
  MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Reduce(&processThreadNumberInCircle, &totalNumberInCircle, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (myRank == 0) {
    printf("Got %ld out of %ld in circle target!\n", totalNumberInCircle, totalNumTosses);
    piEstimate = (4*totalNumberInCircle)/((double) totalNumTosses);
    printf("Elapsed time = %f seconds \n", elapsed);
    printf("Pi is approximately %.16f, Error is %.16f\n", piEstimate, fabs(piEstimate - PI25DT));
  }
  MPI_Finalize();
  return 0;
}

/* Function gets input from command line for totalNumTosses */
void Get_input(int argc, char* argv[], int myRank, int* numThreads_p, long* totalNumTosses_p){
if (myRank == 0) {
	if (argc!= 3){
	    fprintf(stderr, "usage: mpirun -np <N> %s <number of threads> <number of tosses> \n", argv[0]);
          fflush(stderr);
          *totalNumTosses_p = 0;
	} else {
		*numThreads_p = atoi(argv[1]);
		*totalNumTosses_p = atoi(argv[2]);
	}
}
// Broadcasts value of numThreads & totalNumTosses to each process
MPI_Bcast(numThreads_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(totalNumTosses_p, 1, MPI_LONG, 0, MPI_COMM_WORLD);

// 0 totalNumTosses ends the program
  if (*totalNumTosses_p == 0) {
      MPI_Finalize();
      exit(-1);
  }
}

/* Function implements Monte Carlo version of tossing darts at a board */
long Toss (long processThreadTosses, int myRank, int threadID){
  long toss, numberInCircle = 0;
	double x,y,seed;
	seed = (myRank * threadID) + myRank + threadID + 1;
	srand(seed);
	for (toss = 0; toss < processThreadTosses; toss++) {
	   x = rand()/(double)RAND_MAX;
	   y = rand()/(double)RAND_MAX;
	   if((x*x+y*y) <= 1.0 ) numberInCircle++;
    }
    return numberInCircle;
}
