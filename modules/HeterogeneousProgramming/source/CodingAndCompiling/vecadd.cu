#include <stdio.h>
#include <cuda.h>

/* kernel function */
__global__ void kernel(int *a, int *b, int *c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        c[index] = a[index] + b[index];
}

/* function to be called in the MPI program, and size is the number of elements in array */
extern "C" void run_kernel(int *a, int *b, int *c, int size, int nblocks, int nthreads) {

    /* pointers to the arrays on the GPU */
    int *dev_a, *dev_b, *dev_c; 

    /* Allocate memory on the GPU */
    cudaMalloc((void**)&dev_a, sizeof(int)*size);
    cudaMalloc((void**)&dev_b, sizeof(int)*size);
    cudaMalloc((void**)&dev_c, sizeof(int)*size);

    /* Copy array a and b from host to GPU */
    cudaMemcpy(dev_a, a, sizeof(int)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int)*size, cudaMemcpyHostToDevice);

    /* Calling the kernel function to do calculation */
    kernel<<<nblocks, nthreads>>>(dev_a, dev_b, dev_c, size);

    /* Copy the result array from device to host*/
    cudaMemcpy(c, dev_c, sizeof(int)*size, cudaMemcpyDeviceToHost);

    /* Free memory on the device */
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
