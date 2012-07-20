#include <cuda.h>
#include <stdio.h>

/* kernel function */
__global__ void MatrixKernel(float *dM, float *dN, float *dP, int width) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float pvalue = 0.0f;
    for (int k = 0; k < width; k++) {
        float M_elem = dM[row * width + k];
        float N_elem = dN[k * width + col];
        pvalue += M_elem * N_elem;
    }
    dP[row * width + col] = pvalue;
}

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
