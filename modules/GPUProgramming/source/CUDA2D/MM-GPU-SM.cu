#include "../common/book.h"
#include "cuda.h"

#define blockD 32
   
main(void){

    void MatrixMultiplication(float *, float *, float *, int);

    const int Width = 1024;

    int size = Width * Width * sizeof(float);
    float *M, *N, *P;   
    
    // allocate memory on the CPU
    M = (float*)malloc(size);
    N = (float*)malloc(size);
    P = (float*)malloc(size);

    // initialize the matrices
    for (int y=0; y<Width; y++) {
	    for (int x=0; x<Width; x++){
	   		M[y*Width + x] = x + y*Width;
       		N[y*Width + x] = x + y*Width; 
	   }
    }

    MatrixMultiplication(M, N, P, Width);

    // free the memory allocated on the CPU
    free( M );
    free( N );
    free( P );

    return 0;
}

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
  // declare cache in the shared memory
  __shared__ float Mds[blockD][blockD];
  __shared__ float Nds[blockD][blockD];
 
  // keep track of column index of the Pd element using thread index
  int x = threadIdx.x + blockIdx.x * blockDim.x; // x is column
  // keep track of row index of the Pd element using thread index
  int y = threadIdx.y + blockIdx.y * blockDim.y; // y is row

  float Pvalue = 0;
  // Loop over the Md and Nd block dimension required to compute the Pd element
  for (int m = 0; m < Width/blockD; m++){
	
    // collaboratively loading of Md and Nd blocks into shared memory	 
    Mds[threadIdx.y][threadIdx.x] = Md[y * Width + (m * blockD + threadIdx.x)];
    Nds[threadIdx.y][threadIdx.x] = Md[(m * blockD + threadIdx.y) * Width + x];
    __syncthreads();
    
    // keep track of the running sum    
    for (int k = 0; k < blockD; k++)
      Pvalue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
    __syncthreads();
  }
  
  // write back to the global memory
  Pd[y * Width + x] = Pvalue;
}

void MatrixMultiplication(float *M, float *N, float *P, int Width) {

    int size = Width * Width * sizeof(float);
    float *Md, *Nd, *Pd; 

    // capture start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // allocate memory on the GPU
    HANDLE_ERROR( cudaMalloc((void**)&Md, size) );
    HANDLE_ERROR( cudaMalloc((void**)&Nd, size) );
    HANDLE_ERROR( cudaMalloc((void**)&Pd, size) );

    // transfer M and N to device memory
    HANDLE_ERROR( cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice) );

    // kernel invocation code
    dim3 dimBlock(blockD, blockD);
    dim3 dimGrid(Width/blockD, Width/blockD);
    MatrixMulKernel<<<dimGrid, dimBlock>>>( Md, Nd, Pd, Width);

    // transfer P from device    
    HANDLE_ERROR( cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost) );

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree(Md) );
    HANDLE_ERROR( cudaFree(Nd) );
    HANDLE_ERROR( cudaFree(Pd) );

    // destroy events to free memory
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
}
