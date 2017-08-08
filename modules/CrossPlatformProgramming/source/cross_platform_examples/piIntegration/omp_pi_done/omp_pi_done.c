/****************************************************************************
* Program: Pi Calculation
* This PI program was taken from Argonne National Laboratory.
* OpenMP version by Hannah Sonsalla, Macalester College, 2017
****************************************************************************/

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void getInput(int argc, char* argv[], int* numThreads_p, int* n_p);

/* Function gets input from command line for numThreads and n */
void getInput(int argc, char* argv[], int* numThreads_p, int* n_p){
	if (argc!= 3){
		fprintf(stderr, "usage:  %s <number of threads> <number of bins> \n", argv[0]);
        fflush(stderr);
        *n_p = -1;
    } else {
		*numThreads_p = atoi(argv[1]);
		*n_p = atoi(argv[2]);
	}

	// negative n ends the program
    if (*n_p <= 0) {
        exit(-1);
    }
}

int main(int argc, char *argv[]) {

    int numThreads,                                     /* number of threads */
        i,
        n;                                              /* the number of bins */

    double PI25DT = 3.141592653589793238462643;         /* 25-digit-PI*/
    double pi,                                          /* value of PI in total*/
           step,                                        /* the step */
           sum,                                         /* sum of area under the curve */
           x;

    double start_time,          /* starting time */
           end_time,            /* ending time */
           computation_time;    /* time for computing value of PI */


    getInput(argc, argv, &numThreads, &n);
    omp_set_num_threads(numThreads);

    start_time = omp_get_wtime();

    step = 1.0 / (double) n;
    sum = 0.0;
    
    #pragma omp parallel for schedule(static,1) reduction(+:sum) private(i,x)
	for (i = 1; i <= n; i ++) {
		x = step * ((double)i - 0.5);
		sum += (4.0/(1.0 + x*x));
	}
   

    pi = step * sum;
    
    end_time = omp_get_wtime();

    printf("Pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
    computation_time = end_time - start_time;
    printf("Time of calculating PI is: %f seconds \n", computation_time);

    return 0;
}

