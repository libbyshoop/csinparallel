/* simpleParallel.c
 * ... illustrates the use of OpenMP's parallel directive...
 *
 * Joel Adams, Calvin College, at November 2009.
 *
 * Usage: ./simpleParallel 
 * Compile & run, uncomment the pragma, recompile & run, compare.
 */

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char** argv) {

    printf("\n\nBefore...\n");

//    #pragma omp parallel 
    printf("\nDuring...");

    printf("\n\nAfter...\n\n");

    return 0;
}

