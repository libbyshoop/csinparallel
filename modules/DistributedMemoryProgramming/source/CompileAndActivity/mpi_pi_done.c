/****************************************************************************
* Program: Pi Calculation
* MPI_Bcast and MPI_Reduce
* This PI program was taken from Argonne National Laboratory.
* Modified for command line argument by Hannah Sonsalla, Macalester College
****************************************************************************/

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_NAME 80   /* length of characters for naming a process */

void getInput(int argc, char* argv[], int rank, int* n);

void getInput(int argc, char* argv[], int rank, int* n){
    if (rank == 0){
        if(argc != 2){
            fprintf(stderr, "usage: mpirun -n %s <number of bins> \n", argv[0]);
            fflush(stderr);
            *n = -1;
        } else {
            *n = atoi(argv[1]);
        }
    }
    
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (*n <= 0) {
        MPI_Finalize();
        exit(-1);
    }
}


int main(int argc, char *argv[]) {

    int rank,                                           /* rank variable to identify the process */
        nprocs,                                         /* number of processes */
        i,
        len,                                            /* variable for storing name of processes */
        n;                                              /* the number of bins */

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
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &len);

    getInput(argc, argv, rank, &n);

    start_time = MPI_Wtime();;

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
    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
        end_time = MPI_Wtime();
        computation_time = end_time - start_time;
        printf("Time of calculating PI is: %f seconds \n", computation_time);
    }
    /* Terminate MPI execution environment */
    MPI_Finalize();
    return 0;
}

