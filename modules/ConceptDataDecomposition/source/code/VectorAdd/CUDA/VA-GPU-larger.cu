/*
 * A simplified example of vector addition in CUDA to illustrate the
 * data decomposition pattern using blocks of threads.
 *
 * To compile:
 *   nvcc -o va-GPU-larger VA-GPU-larger.cu
 */

#include <stdio.h>


// In this example we use a very small number of blocks
// and threads in those blocks for illustration 
// on a very small array
//#define N 8
//#define numThread 2 // 2 threads in a block
//#define numBlock 4  // 4 blocks

// This is still a small array of size 512
#define N 4 * 128
#define numThread 32 // 32 threads in a block
#define numBlock 4  // 4 blocks

// Try larger combinations of N, numThread, and numBlock  below and
// recomplie the code.  On most GPUs, the maximum value for numThread is 1024.
// Do some investigation of how many blocks you can declare for your
// particular GPU.
//#define N 
//#define numThread 
//#define numBlock 

/*
 *  The 'kernel' function that will be executed on the GPU device hardware.
 */
__global__ void add( int *a, int *b, int *c ) {

    // the initial index that this thread will work on
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // In this above example code, we assume a linear set of blocks of threads in the 'x' dimension,
    // which is declared in main below when we run this function.

    // The actual computation is being done by individual threads
    // in each of the blocks.
    // e.g. suppose we use 4 blocks and 32 threads per block (128 threads will run in parallel)
    //      and our total array size N is 4 * 128, or 512
    //      the thread whose threadIdx.x is 30 within block 0 will compute c[30],
    //          because tid = (32 * 0)  + 30
    //      the thread whose threadIdx.x is 30 within block 1 will compute c[62],
    //          because tid = (32 * 1) + 30
    //      the thread whose threadIdx.x is 30 within block 3 will compute c[126],
    //          because tid = (32 * 3) + 30
    //
    //     in the first round of parallel execution within the following while loop,
    //          c[0] through c[127] will be computed concurrently in this example scenario
    //
    while (tid < N) {
        c[tid] = a[tid] + b[tid];       // The actual computation done by the thread
        tid += blockDim.x;  // increment this thread's index by the number of threads per block
    }
    
    // Continuning the above example for N = 512 and 128 threads running concurrently in 4 blocks of 32:
    //   In the second round of parallel execution of 128 threads (4 blocks of 32 threads) within the loop,
    //   c[128] through c[255] would be computed concurrently.
    //   The thread whose tid was 0 will increment its tid to 128, thread 1 to 129, and so on to thread 127,
    //   whose index will be 255.
    //   Two more rounds of 128 concurrent threads would complete the while loop.
}


/*
 * The main program that directs the execution of vector add on the GPU
 */
int main( void ) {
    int *a, *b, *c;               // The arrays on the host CPU machine
    int *dev_a, *dev_b, *dev_c;   // The arrays for the GPU device

    // allocate the memory on the CPU
    a = (int*)malloc( N * sizeof(int) );
    b = (int*)malloc( N * sizeof(int) );
    c = (int*)malloc( N * sizeof(int) );

    // fill the arrays 'a' and 'b' on the CPU with dummy values
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // allocate the memory on the GPU
     cudaMalloc( (void**)&dev_a, N * sizeof(int) );
     cudaMalloc( (void**)&dev_b, N * sizeof(int) );
     cudaMalloc( (void**)&dev_c, N * sizeof(int) );

    // copy the arrays 'a' and 'b' to the GPU
     cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice );
     cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice );

    // Execute the vector addition 'kernel function' on th GPU device,
    // declaring how many blocks and how many threads per block to use.
    add<<<numBlock,numThread>>>( dev_a, dev_b, dev_c );

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost );


    // verify that the GPU did the work we requested
    bool success = true;
    int total=0;
    printf("Checking %d values in the array.\n", N);
    for (int i=0; i<N; i++) {
        if ((a[i] + b[i]) != c[i]) {
            printf( "Error:  %d + %d != %d\n", a[i], b[i], c[i] );
            success = false;
        }
        total += 1;
    }
    if (success)  printf( "We did it, %d values correct!\n", total );

    // free the memory we allocated on the CPU
    free( a );
    free( b );
    free( c );

    // free the memory we allocated on the GPU
     cudaFree( dev_a );
     cudaFree( dev_b );
     cudaFree( dev_c );

    return 0;
}

