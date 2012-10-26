Activities
==========

.. highlight:: c

In this chapter, we are going to look at two problems, one is vector-matrix multiplication, and the other is matrix-matrix multiplication. We chose these problems because they are relatively easy to decompose, and not complicated. 

Activity 2: Vector Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the source code to do your activity: 

	:download:`download CUDA program <vec_matrix_mul_todo.cu>`

	:download:`download MPI program <vec_matrix_mul_todo.c>`

In this activity, we are going to compute vector-matrix multiplication in hybrid environment MPI and CUDA. You might already see this problem before in the MPI module. If so, it will not be much different. The basic idea is we want to split the rows of the matrix, and ask the master to send some rows of the matrix and entire input vector to each worker. We then ask each worker to receive messages from the master, and each worker will call the CUDA function to do computation on their own GPU. Basically, on the GPU each thread will compute a new element of the vector. 

:Comments on CUDA program:
	
	* First let's look at the kernel function in the CUDA program. We use one thread to compute multiplications of a row of the matrix with the vector, and this produces a new element of the result vector. We also use our two-dimensional array, the matrix, as a one-dimensional array to ease the problem. We are using two-dimensional block and thread. We use *threadIdx.y* because we want each thread to work on the multiplications of each row of the matrix with the vector. ::

		/* kernel function for computation on the GPU */
		__global__ void kernel(int *A, int *x, int *y, int width, int block_size) {

			int tid = blockIdx.y * block_size + threadIdx.y;
			int entry = 0;
			for (int i = 0; i < width; i++) {
				entry += A[tid * width + i] * x[i];
			}
			y[tid] = entry;
		}


	* Then we need to have a function that calls the kernel function on the host. This function is to allocate memory for the matrix and vector on the device, copy matrix and vector from host to device, compute the vector matrix multiplication, and copy the result vector from device to host. This function will be called in the MPI program. Your task is to call kernel function at **TO DO**. ::

		/* function on the host, CPU */
		extern "C" void run_kernel(int *A, int *x, int *y, int width, int block_size) {

			/* the size of the matrix and the vector */
			int matrix_size = sizeof(int) * width * width;
			int vector_size = sizeof(int) * width;

			/* Pointes array on GPU */
			int *dev_A, *dev_x, *dev_y;

			/* Allocate memory on GPU */
			cudaMalloc((void**)&dev_A, matrix_size);
			cudaMalloc((void**)&dev_x, vector_size);
			cudaMalloc((void**)&dev_y, vector_size);

			/* Copy matrix and vector from CPU to GPU */
			cudaMemcpy(dev_A, A, matrix_size, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_x, x, vector_size, cudaMemcpyHostToDevice);

			/* Initializing the grid size and block size */
			dim3 dimGrid(width/block_size, width/block_size);
			dim3 dimBlock(block_size, block_size);

			/* Running the kernel function */
			// TO DO
			// call the kernel function
			// end TO DO

			/* Copy the output vector from GPU to CPU */
			cudaMemcpy(y, dev_y, vector_size, cudaMemcpyDeviceToHost);

			/* Free memory on GPU */
			cudaFree(dev_A);
			cudaFree(dev_x);
			cudaFree(dev_y);
		}

:Comments on MPI program:
	
	* Now we can look at our MPI program with an addition of a CUDA function. First we need to initialize the MPI execution environment, define the size of MPI_COMM_WORLD, and give a unique rank to each process. Then we ask the master to initialize the input matrix and vector, divide the matrix by row, and send their pieces and the entire vector to each worker. Your task is to complete code at **TO DO**. ::

		/* Initialize MPI execution environment */
		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
		
		MPI_Get_processor_name(name, &len);

		/******************************* Master ***************************/
		if (rank == 0) {
			/* Initialize Matrix and Vector */
			for(i = 0; i < ROWS; i++) {
				// change here to use random integer
				vector[i] = 1;
				for(j = 0; j < COLS; j++) {
					// change here to use random integer
					matrix[i][j] = 1;
				}
			}

			numworkers = nprocs - 1;

			/* divide the number of rows for each worker */
			averow = ROWS/numworkers;
			extra = ROWS%numworkers;
			offset = 0;
			mtype = FROM_MASTER;

			/* Master sends smaller task to each worker */
			for(dest = 1; dest <= numworkers; dest++) {
 				rows = (dest <= extra) ? averow + 1 : averow;

				// TO DO
				// send each piece of matrix and entire vector to each worker
				// end TO DO

				printf("Master sent elements %d to %d to rank %d\n", offset, offset + rows, dest);
				offset += rows;
			}
		}

	* We need to ask all workers to receive the messages sent from the master. Then we want each worker to call the CUDA function to compute their vector matrix multiplication. After having computed their multiplications, each worker needs to send their result back to the master. Please complete the following code at **TO DO**. ::

		/************************************** Workers *************************************/
		if (rank > 0) {
			mtype = FROM_MASTER;
			/* Each worker receives messages sent from the master*/

			// TO DO
			// receive each piece of the matrix and vector sent from master
			// end TO DO

			printf("Worker rank %d, %s receives the messages\n", rank, name);

			/* use CUDA function to compute the the vector-matrix multiplication for each worker */
			// TO DO
			// call a function from CUDA program
			// end TO DO

			/* Each worker sends the result back to the master */
			mtype = FROM_WORKER;
			MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
			MPI_Send(&result, rows, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
			printf("Worker rank %d, %s sends the result to master \n", rank, name);
		}

	* Finally, we need to ask the master to receive the result vector sent from each worker, and prints the result vector. ::

		/* Master receives the output from each worker*/
		mtype = FROM_WORKER;
		for (i = 1; i <= numworkers; i++) {
			source = i;
			MPI_Recv(&offset, 1, MPI_INT, source,mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&result[offset], rows, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			printf("Received results from task %d\n", source);
		}

		/* Master prints results */
		for (i = 0; i < ROWS; i++) {
			printf("The element of output vector is: %d\n", result[i]);
		}




If you get stuck, Download the entire source code:

	:download:`download CUDA program <vec_matrix_mul.cu>`

	:download:`download MPI program <vec_matrix_mul.c>`

Activity 3: Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the source code to do this activity: 
	:download:`download CUDA program <matrix_mul_todo.cu>`

	:download:`download MPI program <matrix_mul_todo.c>`


In this activity, we are going to compute matrix-matrix multiplication in hybrid environment MPI and CUDA. The basic idea is we want to split the rows of the first matrix, and ask the master to send some rows of the first matrix and the entire second matrix to each worker. We then ask each worker to receive messages sent from the master, and each worker will call the CUDA function to do computation on their own GPU. 

.. note:: This is not the most efficient method of computing matrix-matrix multiplication by using hybrid environment CUDA and MPI because when the second matrix gets too large, a progammer may not be able to send it to each worker. Furthermore, a programmer can improve the kernel function to be more effecient by using the shared memory archeticture in the GPU. 

:Comments on CUDA Program:

	* First let's look at the kernel function in the CUDA program. In this kernel function, we are using two different threads. One thread is to calculate the row index of the first matrix, and the other is to calculate the column index the second matrix. To calculate the row index, we use *threadIdx.y*, and to calculate the column index, we use *threadIdx.x*. Then we need to iterate over the width of the matrix, and multiply each corresponding elements of the two input matrices, and sum all of them to produce a new element of the result matrix. Your task to complete the code at **TO DO**. ::

		/* kernel function */
		__global__ void MatrixKernel(float *dM, float *dN, float *dP, int width) {

			/* calculate the row index of the dP element and M */
			// TO DO
			// int row = .........
			// end TO DO

			/* calculate the column index of dP element and N */
			// TO DO
			// int col = .........
			// end TO DO

			float pvalue = 0.0f;
			for (int k = 0; k < width; k++) {
				float M_elem = dM[row * width + k];
				float N_elem = dN[k * width + col];
				pvalue += M_elem * N_elem;
			}
			dP[row * width + col] = pvalue;
		}

	* Then we need to have a function that calls the kernel function on the host. This function is to allocate memory for the matrices on the device, copy matrices from host to device, compute the matrix multiplication, and copy the result matrix from device to host. This function will be called in the MPI program. Your task is to call kernel function at **TO DO**. ::

		/* function that you will call in mpi code */
		extern "C" void MatrixMul(float* M, float* N, float* P, int width, int block_size) {

			int matrix_size = width * width * sizeof(float);
			float *dM, *dN, *dP;

			// Allocate and Load M and N to device memory
			cudaMalloc(&dM, matrix_size);
			cudaMemcpy(dM, M, matrix_size, cudaMemcpyHostToDevice);

			cudaMalloc(&dN, matrix_size);
			cudaMemcpy(dN, N, matrix_size, cudaMemcpyHostToDevice);

			// Allocate P on device
			cudaMalloc(&dP, matrix_size);

			dim3 dimGrid(width/block_size, width/block_size);
			dim3 dimBlock(block_size, block_size);

			// TO DO
			// call the kernel function
			// end TO DO

			cudaMemcpy(P, dP, matrix_size, cudaMemcpyDeviceToHost);

			cudaFree(dP);
			cudaFree(dM);
			cudaFree(dN);
		}

:Comments on MPI Program:

	* Now we can look at the MPI program. This MPI program is a revised version of the Matrix Mutiplication MPI program that was written by Blaise Barney. First we need to initialize the MPI execution environment, define the size of MPI_COMM_WORLD, and give a unique rank to each process. Then we ask the master to initialize the input matrices, divide these matrices, and send their pieces to each worker. Your task is to complete the following code at **TO DO**. ::

		/**************************** master task ************************************/
		if (taskid == MASTER) {

			/* Initializing both matrices on master node */
			for (i = 0; i < ROW_A; i++)
				for (j = 0; j < COL_A; j++)
					// change here to use random integer
					a[i][j]= 1;
			for (i = 0; i < COL_A; i++)
				for (j = 0; j < COL_B; j++)
					// change here to use random integer
					b[i][j]= 1;

			/* Computing the average row and extra row for each process */
			averow = ROW_A/numworkers;
			extra = ROW_A%numworkers;
			offset = 0;
			mtype = FROM_MASTER;

			/* Distributing the task to each worker */
			for (dest = 1; dest <= numworkers; dest++) {
				rows = (dest <= extra) ? averow+1 : averow;
				printf("Sending %d rows to task %d offset = %d\n", rows, dest, offset);
				
				// TO DO
				// send some rows of first matrix and entire second matrix to each worker
				// end TO DO

				offset = offset + rows;
			}
		}

	* We need to ask all workers to receive the messages sent from the master. Then we want each worker to call the CUDA function to compute their matrix multiplication. After having computed their multiplications, each worker needs to send their result back to the master. Please complete the following code at **TO DO**. ::

		/**************************** worker task ************************************/
		if (taskid > MASTER) {

			/* Each receives task from master*/	
			mtype = FROM_MASTER;
			
			// TO DO
			// receive the matrices sent from master
			// end TO DO

			/* Calling function from CUDA. Each worker computes on their GPU */
			
			// TO DO
			// call CUDA function 
			// end TO DO

			/* Each worker sends result back to the master */
			mtype = FROM_WORKER;
			MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
			MPI_Send(&c, rows*COL_B, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
		}

	* Finally, we need to ask the master to receive the result matrix sent from each worker, and prints the result matrix. ::

		/* Receive results from worker tasks */
		mtype = FROM_WORKER;
		for (i = 1; i <= numworkers; i++) {
			source = i;
			MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&c[offset][0], rows*COL_B, MPI_FLOAT, source, mtype, MPI_COMM_WORLD, &status);
			printf("Received results from task %d\n",source);
		}

		/* Master prints results */
		printf("******************************************************\n");
		printf("Result Matrix:\n");
		for (i = 0; i < ROW_A; i++) {
			printf("\n");
			for (j = 0; j < COL_B; j++)
				printf("%6.2f   ", c[i][j]);
		}


If you get stuck, download the entire source code:
	:download:`download CUDA program <matrix_mul.cu>`

	:download:`download MPI program <matrix_mul.c>`

