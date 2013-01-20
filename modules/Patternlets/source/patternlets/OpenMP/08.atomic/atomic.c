/* atomic.c
 * ... illustrates a race condition when multiple threads write to a shared variable
 *  (and explores OpenMP private variables and atomic operations).
 *
 * Joel Adams (Calvin College), November 2009.
 * Usage: ./atomic
 * Exercise:
 *  - Compile and run 10 times; note that it always produces the final balance 0
 *  - To parallelize, uncomment A1+A2, recompile and rerun, compare results
 *  - Try 1: recomment A1+A2, uncomment B1+B2, recompile/run, compare
 *  - To fix: recomment B1+B2, uncomment A1+A2, C1+C2, recompile and rerun, compare
 */

#include<stdio.h>
#include<omp.h>

int main() {
    const int REPS = 1000000;
    int i;
    double balance = 0.0;
  
    printf("\nYour starting bank account balance is %0.2f\n", balance);

    // simulate many deposits
//    #pragma omp parallel for                      // A1
//    #pragma omp parallel for private(balance)     // B1
    for (i = 0; i < REPS; i++) {
//        #pragma omp atomic                        // C1
        balance += 10.0;
    }

    printf("\nAfter %d $10 deposits, your balance is %0.2f\n", 
		REPS, balance);

    // simulate the same number of withdrawals
//    #pragma omp parallel for                      // A2
//    #pragma omp parallel for private(balance)     // B2
    for (i = 0; i < REPS; i++) {
//        #pragma omp atomic                          // C2
        balance -= 10.0;
    }

    // balance should be zero
    printf("\nAfter %d $10 withdrawals, your balance is %0.2f\n\n", 
            REPS, balance);

    return 0;
}

