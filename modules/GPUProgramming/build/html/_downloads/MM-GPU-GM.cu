#include "../common/book.h"
#include "cuda.h"

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

__global__ void Kernel(float *Md, float *Nd, float *Pd, int Width) {

  // Calculate the column index of the Pd element, denote by x
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  // Calculate the row index of the Pd element, denote by y
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  float Pvalue = 0;
  // each thread computes one element of the output matrix Pd.      
  for (int k = 0; k < Width; ++k) {
    Pvalue += Md[y*Width + k] * Nd[k*Width + x];
  }

  // write back to the global memory
  Pd[y*Width + x] = Pvalue;
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
    dim3 dimBlock(32, 32);
    dim3 dimGrid(Width/32, Width/32);
    Kernel<<<dimGrid, dimBlock>>>( Md, Nd, Pd, Width);

    // transfer P from device     
    HANDLE_ERROR( cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost) );

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
