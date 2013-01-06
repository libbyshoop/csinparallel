/* sections.c
 * ... illustrates the use of OpenMP's parallel section/sections directives...
 * Joel Adams, Calvin College, at November 2009.
 */

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char** argv) {

    printf("\nBefore...\n\n");

    #pragma omp parallel sections num_threads(4)
    {
        #pragma omp section 
        {
            printf("Section A performed by thread %d\n", 
                    omp_get_thread_num() ); 
        }
        #pragma omp section 
        {
            printf("Section B performed by thread %d\n",
                    omp_get_thread_num() ); 
        }
        #pragma omp section
        {
            printf("Section C performed by thread %d\n",
                    omp_get_thread_num() ); 
        }
        #pragma omp section 
        {
                printf("Section D performed by thread %d\n", 
                         omp_get_thread_num() ); 
        }
    }

    printf("\nAfter...\n\n");

    return 0;
}

