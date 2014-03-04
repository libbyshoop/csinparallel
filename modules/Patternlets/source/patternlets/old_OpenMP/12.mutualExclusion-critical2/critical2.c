/* critical.c
 * ... compares the performance of OpenMP's critical and atomic directives
 *
 * Joel Adams, Calvin College, November 2009.
 * Usage: ./critical
 * Exercise:
 *  - Compile, run, compare times for critical vs. atomic
 *  - Compute how much more costly critical is than atomic, on average
 *  - Create an expression setting a shared variable that atomic cannot handle (research)
 */

#include<stdio.h>
#include<omp.h>

void print(char* label, int reps, double balance, double total, double average) {
    printf("\nAfter %d $10 deposits using '%s': \
            \n\t- balance = %0.2f, \
            \n\t- total time = %0.12f, \
            \n\t- average time per deposit = %0.12f\n\n", 
               reps, label, balance, total, average);
}

int main() {
    const int REPS = 1000000;
    int i;
    double balance = 0.0,
           startTime = 0.0, 
           stopTime = 0.0,
           totalTime = 0.0;
  
    printf("\nYour starting bank account balance is %0.2f\n", balance);

    // simulate many deposits using atomic
    startTime = omp_get_wtime();
    #pragma omp parallel for 
    for (i = 0; i < REPS; i++) {
        #pragma omp atomic
        balance += 10.0;
    }
    stopTime = omp_get_wtime();
    totalTime = stopTime - startTime;
    print("atomic", REPS, balance, totalTime, totalTime/REPS);


    // simulate the same number of deposits using critical
    balance = 0;
    startTime = omp_get_wtime();
    #pragma omp parallel for 
    for (i = 0; i < REPS; i++) {
         #pragma omp critical
         {
             balance += 10.0;
         }
    }
    stopTime = omp_get_wtime();
    totalTime = stopTime - startTime;
    print("critical", REPS, balance, totalTime, totalTime/REPS);

    return 0;
}

