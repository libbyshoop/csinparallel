Activities
==========

.. highlight:: c

In this chapter, we are going to look at two problems, one is vector-matrix multiplication, and the other is matrix-matrix multiplication. We chose these problems because they are relatively easy to decompose, and not complicated. 

Activity 2: Vector Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this activity, we are going to compute vector-matrix multiplication in hybrid environment MPI and CUDA. You might already see this problem before in the MPI module. If so, it will be not much different. The basic idea is we would like to split the rows of the matrix, and ask the master to send some rows of the matrix and entire input vector to each worker. We then ask each worker to receive messages from the master, and each worker will call the CUDA function to do computation on their own GPU. Basically, on the GPU each thread will take care of a multiplication between an element of the matrix, and an element of the vector. This obviously would speed up our computation.

.. literalinclude:: vec_matrix_mul.cu
	:linenos:

.. literalinclude:: vec_matrix_mul.c
	:linenos:

Activity 3: Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this activity, we are going to compute matrix-matrix multiplication in hybrid environment MPI and CUDA. The basic idea is we want to split the rows of the first matrix, and ask the master to send some rows of the first matrix and the entire second matrix to each worker. We then ask each worker to receive messages sent from the master, and each worker will call the CUDA function to do computation on their own GPU. On the GPU each thread will take care of a multiplication between an element of the first matrix, and an element of the second matrix. 

.. note:: This is not the most efficient method of doing matrix-matrix multiplication by using CUDA and MPI because when the second matrix gets too large, a progammer may not be able to send it to each worker. Furthermore, a programmer can improve the kernel function to be more effective by using the shared memory archeticture in the GPU. 

:Comments on CUDA Program:

	* First let's look at the kernel function in the CUDA program. ::

		/* kernel function */
		__global__ void MatrixKernel(float *dM, float *dN, float *dP, int width) {

			/* calculate the row index of the dP element and M */
			int row = blockIdx.y * blockDim.y + threadIdx.y;
			/* calculate the column index of dP element and N */
			int col = blockIdx.x * blockDim.x + threadIdx.x;

			float pvalue = 0.0f;
			for (int k = 0; k < width; k++) {
				float M_elem = dM[row * width + k];
				float N_elem = dN[k * width + col];
				pvalue += M_elem * N_elem;
			}
			dP[row * width + col] = pvalue;
		}

	* We use 2-dimensional blocks and threads for matrices. Each thread calculate an element of the result matrix. 

	* Then we need to have a function that calls the kernel function on the host. This function is to allocate memory for the matrices on the device, copy matrices from host to device, compute the matrix multiplication, and copy the result matrix from device to the host. This function will be called in the MPI program. ::

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

	* Now we can look at the MPI program. This MPI program is a revised version of the Matrix Mutiplication MPI program that was written by Blaise Barney. First we need to initialize the MPI execution environment, define the size of MPI_COMM_WORLD, and give a unique rank to each process. Then we ask the master to initialize the input matrices, divide these matrices, and send their pieces to each worker. ::

		/**************************** master task ************************************/
	    if (taskid == MASTER) {

			/* Initializing both matrices on master node */
			for (i = 0; i < ROW_A; i++)
				for (j = 0; j < COL_A; j++)
					a[i][j]= 1;
			for (i = 0; i < COL_A; i++)
				for (j = 0; j < COL_B; j++)
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

	* We need to ask all workers to receive the messages sent from the master. Then we want each worker to call the CUDA function to compute their matrix multiplication. After having computed their multiplications, each worker needs to send their result back to the master. Please complete the following code at **TO DO**.::

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

Download the source code to do your activity: 
	:download:`download CUDA program <matrix_mul_todo.cu>`

	:download:`download MPI program <matrix_mul_todo.c>`

Download the entire source code:
	:download:`download CUDA program <matrix_mul.cu>`

	:download:`download MPI program <matrix_mul.c>`

