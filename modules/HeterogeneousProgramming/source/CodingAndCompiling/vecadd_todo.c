#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 10        /* number of elements in each array */
#define MAX_NAME 80     /* lenght of character array for name of the nodes */
#define FROM_MASTER 1   /* message sent from master */
#define FROM_WORKER 2   /* message sent from workers*/
#define MASTER 0        /* master has rank 0 */
#define BLOCKS 30       /* Number of blocks per grid */
#define THREADS 128     /* Number of threads per block */

MPI_Status status;      /* used to return data in recieving */

/* Cuda function declared */
void run_kernel(int *a, int *b, int *c, int size, int nblocks, int nthreads);

int main(int argc, char* argv[]) {

    int rank,   /* rank for each process */
        size,   /* size of the communicator */
        len;    /* length of the name of a process */

    int i,
        source,         /* rank of the source */
        dest,           /* rank of the destination */
        num_worker,     /* number of workers */
        ave_size,       /* average size of task to be sent to each worker */
        extra,          /* extra task to be sent to some workers */
        mtype,          /* message type */
        offset,         /* starting position of element */
        eles;           /* number of elements to be sent */

    int arr_a[WIDTH], arr_b[WIDTH], arr_c[WIDTH]; /* vectors for addition */

    char name[MAX_NAME];        /* character array for storing name of process */

    /* Initialize MPI execution environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /******************** Master ***********************/
    if (rank == MASTER) {

        /* Initializing both vectors in master */
        int sum = 0;
        for (i = 0; i < WIDTH; i++) {
            arr_a[i] = i;
            arr_b[i] = i*i;
        }

        /* Decomposing the problem into smaller problems, and send each task
         * to each worker. Master not taking part in any computation.
         */
        num_worker = size - 1;
        ave_size = WIDTH/num_worker; /* finding the average size of task for a process */
        extra = WIDTH % num_worker;  /* finding extra task for some processes*/
        offset = 0;
        mtype = FROM_MASTER;    /* message sends from master */

        /* Master sends each task to each worker */
        for (dest = 1; dest <= num_worker; dest++) {
            eles = (dest <= extra) ? ave_size + 1: ave_size;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&eles, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            
            // TO DO
            // send a piece of each vector to each worker
            // end TO DO

            printf("Master sent elements %d to %d to rank %d\n", offset, offset + eles, dest);
            offset += eles;

        }

        /* Master receives the result from each worker */
        mtype = FROM_WORKER;
        for(i = 1; i <= num_worker; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&eles, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&arr_c[offset], eles, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n", source);
        }

        /* checking the result on master */
        for (i = 0; i < WIDTH; i ++) {
            if (arr_c[i] != arr_a[i] + arr_b[i]) {
                printf("Failure !");
                return 0;
            }
        }
        printf("Successful !\n");
    }

    /* The workers receive the task from master, and will each run run_kernel to
     * compute the sum of each element from vector a and vector b. After computation
     * each worker sends the result back to master node.
     */
    /******************************* Workers **************************/
    if (rank > MASTER) {
        mtype = FROM_MASTER;
        source = MASTER;
        /* Receive data from master */
        
        // TO DO
        // receive the vectors sent from master
        // end TO DO
        
        MPI_Get_processor_name(name, &len);

        /* Use kernel to compute the sum of element a and b */
        
        // TO DO
        // call run_kernel function here
        // end TO DO

        /* send result back to the master */
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&eles, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&arr_c, eles, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    }

    /* Terminate the MPI execution environment */
    MPI_Finalize();
    return 0;
}
    
