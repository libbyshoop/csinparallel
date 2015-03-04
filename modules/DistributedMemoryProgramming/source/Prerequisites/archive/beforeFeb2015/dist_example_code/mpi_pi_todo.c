/************************************************************
* Program: Pi Calculation
* MPI_Bcast and MPI_Reduce
* This PI program was taken from Argonne National Laboratory.
*************************************************************/

#include "mpi.h"
#include <math.h>
#include <stdio.h>

#define MAX_NAME 80   /* length of characters for naming a process */
#define MASTER 0      /* rank of the master */

int main(int argc, char *argv[]) {

    int rank,                                           /* rank variable to identify the process */
        nprocs,                                         /* number of processes */
        i,
        len;                                            /* variable for storing name of processes */

    int n = 10000;                                      /* the number of bins */
    double PI25DT = 3.141592653589793238462643;         /* 25-digit-PI*/
    double mypi,                                        /* value from each process */
           pi,                                          /* value of PI in total*/
           step,                                        /* the step */
           sum,                                         /* sum of area under the curve */
           x;

    char name[MAX_NAME];        /* char array for storing the name of each process */

    double start_time,          /* starting time */
           end_time,            /* ending time */
           computation_time;    /* time for computing value of PI */

    /*Initialize MPI execution environment */
    // TO DO
    // ..............
    // end TO DO

    MPI_Get_processor_name(name, &len);

    start_time = MPI_Wtime();

    /* Broadcast the number of bins to all processes */
    /* This broadcasts an integer which is n, from the master to all processes
     * and
     */

    // TO DO
    // ..............
    // end TO DO

    /* Calculating for each process */
    step = 1.0 / (double) n;
    sum = 0.0;
    for (i = rank + 1; i <= n; i += nprocs) {
        x = step * ((double)i - 0.5);
        sum += (4.0/(1.0 + x*x));
    }

    mypi = step * sum;

    printf("This is my sum: %.16f from rank: %d name: %s\n", mypi, rank, name);

    /* Now we can reduce all those sums to one value which is Pi */
    // TO DO
    // ...............
    // end TO DO

    if (rank == 0) {
        printf("Pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
        end_time = MPI_Wtime();
        computation_time = end_time - start_time;
        printf("Time of calculating PI is: %f\n", computation_time);
    }
    /* Terminate MPI execution environment */
    MPI_Finalize();
    return 0;
}

