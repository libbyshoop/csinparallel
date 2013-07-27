/* parallelHello.c
 * ... illustrates the use of two basic OpenMP commands...
 *
 * Joel Adams, Calvin College, November 2009.
 *
 * Usage: ./parallelHello
 * Compile & run, uncomment pragma, recompile & run, compare results
 */

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char** argv) {

//    #pragma omp parallel 
    {
        int rank  = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        printf("Hello from thread %d of %d\n", rank, numThreads);
    }

    return 0;
}

