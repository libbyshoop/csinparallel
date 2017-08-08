/*
 * Hannah Sonsalla, Macalester College, 2017
 *
 *  calcPiOMP.C
 *
 *   ... OpenMP program to calculate the value of Pi using
 *       Monte Carlo Method. OpenMP pragma in main.
 *
 * Usage:  ./calcPiOMP <number of threads> <number of tosses>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

void Get_input(int argc, char* argv[], int* numThreads, long* totalNumTosses_p);
long Toss (long numProcessTosses, int my_thread_id);

int main(int argc, char** argv) {
	int my_thread_id, numThreads;
  long totalNumTosses, numThreadTosses, numberInCircle = 0;
  double piEstimate;
  double start, finish, elapsed;  // holds wall clock time
  double PI25DT = 3.141592653589793238462643;         /* 25-digit-PI*/

  Get_input(argc, argv, &numThreads, &totalNumTosses);  // Read total number of tosses from command line
  omp_set_num_threads(numThreads); // Set number of threads
  numThreadTosses = totalNumTosses/numThreads;  // Calculate number of tosses per thread

  start = omp_get_wtime();

  #pragma omp parallel reduction(+:numberInCircle)
	{
		my_thread_id = omp_get_thread_num();
		numberInCircle = Toss(numThreadTosses, my_thread_id);
  }

	finish = omp_get_wtime();
  elapsed = (double)(finish - start);

  piEstimate = (4*numberInCircle)/((double) totalNumTosses);
  printf("Elapsed time = %f seconds \n", elapsed);
	printf("Pi is approximately %.16f, Error is %.16f\n", piEstimate, fabs(piEstimate - PI25DT));

  return 0;
}

/* Function gets input from command line for numThreads and numTosses */
void Get_input(int argc, char* argv[], int* numThreads_p, long* numTosses_p){
	if (argc!= 3){
		fprintf(stderr, "usage:  %s <number of threads> <number of tosses> \n", argv[0]);
        fflush(stderr);
        *numTosses_p = -1;
    } else {
		*numThreads_p = atoi(argv[1]);
		*numTosses_p = atoi(argv[2]);
	}

	// negative totalNumTosses ends the program
    if (*numTosses_p <= 0) {
        exit(-1);
    }
}

/* Function implements Monte Carlo version of tossing darts at a board */
long Toss (long numTosses, int my_thread_id){
	long toss, numInCircle = 0;
	double x,y;
	unsigned int seed = (unsigned) time(NULL);
	srand(seed + my_thread_id);
	for (toss = 0; toss < numTosses; toss++) {
	  x = rand_r(&seed)/(double)RAND_MAX;
	  y = rand_r(&seed)/(double)RAND_MAX;
	  if((x*x+y*y) <= 1.0 ) numInCircle++;
  }
  return numInCircle;
}
