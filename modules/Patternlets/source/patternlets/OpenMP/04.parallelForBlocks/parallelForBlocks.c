/* parallelForBlocks.c
 * ... illustrates the use of OpenMP's default parallel for loop
 *  	in which threads iterate through blocks of the index range
 *	(cache-beneficial when accessing adjacent memory locations).
 *
 * Joel Adams, Calvin College, at November 2009.
 * Usage: ./parallelForBlocks [numThreads]
 * Exercise
 */

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, i;

    if (argc > 1) {
        omp_set_num_threads( atoi(argv[1]) );
    }

    #pragma omp parallel for private(rank, i) 
    for (i = 0; i < 8; i++) {
        rank  = omp_get_thread_num();
        printf("Thread %d performed iteration %d\n", 
                 rank, i);
    }

    return 0;
}

