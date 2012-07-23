#include <stdio.h>
#include <cuda.h>

/* kernel function for computation on the GPU */
__global__ void kernel(int *A, int *x, int *y, int width, int block_size) {
    int i;
    int tid = blockIdx.y * blockDim.y + threadIdx.y;
    int entry = 0;
    for (i = 0; i < width; i++) {
        entry += A[tid * width + i] * x[i];
    }
    y[tid] = entry;
}

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
    kernel<<<dimGrid, dimBlock>>>(dev_A, dev_x, dev_y, width, block_size);

    /* Copy the output vector from GPU to CPU */
    cudaMemcpy(y, dev_y, vector_size, cudaMemcpyDeviceToHost);

    /* Free memory on GPU */
    cudaFree(dev_A);
    cudaFree(dev_x);
    cudaFree(dev_y);
}
