#include <mpi.h>

#define MAX 80    /* maximum characters for naming the node */

/* Declaring the CUDA function */
void hello();

int main(int argc, char *argv[]) {

   int rank, nprocs, len;
   char name[MAX];      /* char array for storing the name of each node */

   /* Initializing the MPI execution environment */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Get_processor_name(name, &len);

   /* Call CUDA function */
   hello();

   /* Print the rank, size, and name of each node */
   printf("I am %d of %d on %s\n", rank, size, name);

   /*Terminating the MPI environment*/
   MPI_Finalize();
   return 0;
}