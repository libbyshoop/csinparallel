/* parallelForStripes.c
 * ... illustrates how to make OpenMP map threads to 
 *	parallel for-loop iterations in 'stripes' instead of blocks
 *	(use only when not accesssing memory).
 *
 * Joel Adams, Calvin College, November 2009.
 *
 * Usage: ./parallelForStripes [numThreads]
 */

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char** argv) {

    if (argc > 1) {
        omp_set_num_threads( atoi(argv[1]) );
    }

    #pragma omp parallel
    {
        int rank  = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        int i;
        for (i = rank; i < 8; i += numThreads) {
            printf("Thread %d performed iteration %d\n", 
                     rank, i);
        }
    }

    return 0;
}

