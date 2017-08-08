/****************************************************************************
* Program: Pi Calculation
* This PI program was taken from Argonne National Laboratory.
* MPI + OpenMP version by Hannah Sonsalla, Macalester College
****************************************************************************/

#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_NAME 80   /* length of characters for naming a process */

void getInput(int argc, char* argv[], int rank, int* numThreads, int* n);

void getInput(int argc, char* argv[], int rank, int* numThreads, int* n){
    if (rank == 0){
        if(argc != 3){
            fprintf(stderr, "usage: mpirun -n %s <number of threads> <number of bins> \n", argv[0]);
            fflush(stderr);
            *n = -1;
        } else {
			*numThreads = atoi(argv[1]);
            *n = atoi(argv[2]);
        }
    }
    
    MPI_Bcast(numThreads, 1, MPI_INT, 0, MPI_COMM_WORLD);
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
        numThreads,										/* number of threads per process */
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

    getInput(argc, argv, rank, &numThreads, &n);  
    omp_set_num_threads(numThreads);
    
    if (rank == 0) {
		printf("Number of processes: %d, Number of Threads per process: %d\n", nprocs, numThreads);
    }

    start_time = MPI_Wtime();

    /* Calculating for each process */
    step = 1.0 / (double) n;
    sum = 0.0;
    
    #pragma omp parallel for schedule(static,1) reduction(+:sum) private(i,x)
    for (i = rank + 1; i <= n; i += nprocs) {
        x = step * ((double)i - 0.5);
        sum += (4.0/(1.0 + x*x));
    }

    mypi = step * sum;

    printf("This is my sum: %.16f from process: %d name: %s\n", mypi, rank, name);

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

