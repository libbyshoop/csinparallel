#include <stdio.h>
#include <cuda.h>

/* kernel function for GPU */
__global__ void kernel(void) {
}

extern "C" void hello(void) {
    kernel<<<1, 1>>>();
    printf("Hello World !\n");
}