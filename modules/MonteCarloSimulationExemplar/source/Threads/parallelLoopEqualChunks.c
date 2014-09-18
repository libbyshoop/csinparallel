/* 
 * Parallel for loop using equal chunks per thread.
 */

#include <stdio.h>    // printf()
#include <stdlib.h>   // atoi()
#include <omp.h>      // OpenMP

int main(int argc, char** argv) {
    const int REPS = 16;

    omp_set_num_threads(4);

    #pragma omp parallel for  
    for (int i = 0; i < REPS; i++) {
        int id = omp_get_thread_num();
        printf("Thread %d performed iteration %d\n", 
                 id, i);
    }

    printf("Main thread 0 done.\n");
    return 0;
}

