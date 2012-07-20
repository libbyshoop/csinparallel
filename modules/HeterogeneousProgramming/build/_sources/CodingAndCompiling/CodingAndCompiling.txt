Coding and Compiling a Heterogeneous Program
============================================

Heterogeneous Program: Hello World
**********************************

Distributed memory computing and GPU computing are two different parallel programming models. In this section, you will learn how to put these two parallel models together, and that will speed up your running time. In order to introduce you to this new concept, we will look at the **Hello World** program using hybrid CUDA and MPI model. In order to combine CUDA and MPI, we need to get their codes to communicate to each other during the compilation. Let's look at the **Hello World** program below.

.. highlight:: c

**CUDA program**

.. literalinclude:: hello.cu
	:linenos:

**MPI program integrated with CUDA**

.. literalinclude:: hello_mpi.c
	:linenos:	

:Comments:
	
	* From source codes above, CUDA program creates a grid consisting a block, which has a thread. It will print “Hello World !”. The **hello** function in CUDA program uses the keyword **extern “C”**, so the MPI program is able to link to use **hello** function using a 'C' compatible header file that contains just the declaration of **hello** function. In addition, MPI program only creates the MPI execution environment, defines the size of the MPI_COMM_WORLD, gives the unique rank to each process, calls **hello** function from CUDA program to print "Hello World !", and prints the rank, size, and name of the process. Finally, all processes terminate the MPI execution environment. 	


Compiling a Heterogeneous Program
*********************************

The most common way of compiling a heterogeneous program MPI and Cuda is:

	1. Make a CUDA object from the CUDA program. This can be done by using this command on the terminal: ::

		nvcc -c cuda.cu -o cuda.o

	2. Make an MPI object from MPI program. This can be done by using this command on the terminal: ::

		mpicc -c mpi.c -o mpi.o

	3. Make an executable file from both objects. This can be done by using this command on the terminal: ::

		mpicc -o cudampi mpi.o cuda.o -L/usr/local/cuda/lib64 -lcudart

To execute the executable file, **cudampi**, we can enter the following command on the terminal: ::

	mpirun -machinefile machines -x LD_LIBRARY_PATH -np #processes ./cudampi


Activity 1: Vector Addition
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this activity, we are going to compute vector addition by using hybrid programming model, CUDA and MPI. Vector addition is very simple and easy. Suppose we have vector *A* and vector *B*, and both have the same length. To add vector *A* and *B*, we just add the corresponding element of *A* and *B*. This results in a new vector of the same length.

:Comments on CUDA Program:

	* We will walk you through this first activity step by step. First, let's look at the CUDA program for vector addition. We need to have a kernel function for vector addition. This should be straight forward to you. Each thread computes an element of the result matrix, where thread index is the index of that element. ::

		__global__ void kernel(int *a, int *b, int *c) {
			/* this thread index is the index of the vector */
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			// TO DO
			// add corresponding elements of a and b
			// end TO DO
		}

	* Another function in the CUDA program is **run_kernel**, which works on the host(CPU) and calls the kernel function on the device(GPU). This function allocates memory on the GPU for storing input vectors, copies input vectors onto the device, does the calculations on the device, copies output vector back to the host, and erases all those vectors on the device. This function will be called in the MPI program. ::
	
		/*
		* size is the number of elements in the vector
		* nblocks is the number of blocks per grid
		* nthreads is the number of threads per block
		*/
		extern "C" void run_kernel(int *a, int *b, int *c, int size, int nblocks, int nthreads) {

			/* pointers for storing each vector on the device*/
			int *dev_a, *dev_b, *dev_c; 

			/* Allocate memory on the device */
			cudaMalloc((void**)&dev_a, sizeof(int)*size);
			cudaMalloc((void**)&dev_b, sizeof(int)*size);
			cudaMalloc((void**)&dev_c, sizeof(int)*size);

			/* Copy vectors a and b from host to device */
			cudaMemcpy(dev_a, a, sizeof(int)*size, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_b, b, sizeof(int)*size, cudaMemcpyHostToDevice);

			/* Calling the kernel function to do calculation */
			
			// TO DO
			// Call kernel function here
			// end TO DO

			/* Copy the result vector from device to host*/
			cudaMemcpy(c, dev_c, sizeof(int)*size, cudaMemcpyDeviceToHost);

			/* Free memory on the device */
			cudaFree(dev_a);
			cudaFree(dev_b);
			cudaFree(dev_c);
		}


:Comments on MPI Program:

	* This MPI program is basically the MPI program with an addition of a function from CUDA program. It splits both input vectors into smaller pieces, and sends each piece of each vector to each worker. Then we will call the **run_kernel** function from CUDA program to calculate additions of the two vectors on each node. 

	* First we need to initialize the MPI execution environment, define the size of all processes, and give a unique rank to each process. Then we will ask the master to initialize the input vectors, split the input vectors into smaller chunks, and send these chunks to each process. Your task is to send the pieces of input vectors to each worker. :: 

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
			ave_size = WIDTH/num_worker;	/* finding the average size of task for a process */
			extra = WIDTH % num_worker;		/* finding extra task for some processes*/
			offset = 0;
			mtype = FROM_MASTER;			/* message sends from master */

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
		}

	* Then we want all workers to receive the messages sent from master, and we call the **run_kernel** function from CUDA program to compute the sum of both vectors on each worker. This function will call the kernel function and compute additions on the GPU of each worker. When they are done with computations, each worker needs to send its result vector to the master. Your task is to receive the vectors sent from master, and to call **run_kernel** function from the CUDA program. ::

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

	* We need to ask the master to receive the result vector sent from each worker. We then can check to see if they are correct. Verification part should not be included in your timing. ::

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

Download the source code to do your activity: 
	:download:`download CUDA program <vecadd_todo.cu>`

	:download:`download MPI program <vecadd_todo.c>`

Download the entire source code:
	:download:`download CUDA program <vecadd.cu>`

	:download:`download MPI program <vecadd.c>`
