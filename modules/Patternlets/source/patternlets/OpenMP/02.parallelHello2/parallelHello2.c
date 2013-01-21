/* parallelHello2.c
 * ... illustrates the use of two basic OpenMP commands
 * 	using the commandline to control the number of threads...
 *
 * Joel Adams, Calvin College, November 2009.
 *
 * Usage: ./parallelHello2 [numThreads]
 * Compile & run with no commandline arg, rerun with different commandline args
 */



#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, numThreads;

    if (argc > 1) {
        omp_set_num_threads( atoi(argv[1]) );
    }

    #pragma omp parallel 
    {
        rank  = omp_get_thread_num();
        numThreads = omp_get_num_threads();
        printf("Hello from thread %d of %d\n", rank, numThreads);
    }

    return 0;
}

