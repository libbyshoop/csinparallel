/* critical.c
 * ... fixes a race condition when multiple threads write to a shared variable
 *  	using the OpenMP critical directive.
 *
 * Joel Adams, Calvin College, November 2009.
 * Usage: ./critical
 * Exercise:
 *  - Compile and run several times; note that it always produces the final balance 0
 *  - Comment out A1+A2; recompile/run and note incorrect result
 *  - To fix: uncomment B1a+B1b+B1c, B2a+B2b+B2c, recompile and rerun, compare
 */

#include<stdio.h>
#include<omp.h>

int main() {
    const int REPS = 1000000;
    int i;
    double balance = 0.0;
  
    printf("\nYour starting bank account balance is %0.2f\n", balance);

    // simulate many deposits
    #pragma omp parallel for
    for (i = 0; i < REPS; i++) {
        #pragma omp atomic                          // A1
//        #pragma omp critical                      // B1a
//        {                                         // B1b
        balance += 10.0;
//        }                                         // B1c
    }

    printf("\nAfter %d $10 deposits, your balance is %0.2f\n", 
		REPS, balance);

    // simulate the same number of withdrawals
    #pragma omp parallel for
    for (i = 0; i < REPS; i++) {
        #pragma omp atomic                          // A2
//        #pragma omp critical                      // B2a
//        {                                         // B2b
        balance -= 10.0;
//        }                                         // B2c
  }

    // balance should be zero
    printf("\nAfter %d $10 withdrawals, your balance is %0.2f\n\n", 
		REPS, balance);

    return 0;
}

