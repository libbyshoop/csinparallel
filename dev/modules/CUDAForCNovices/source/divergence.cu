/*
 * Code to demonstrate thread divergence
 *
 * compile with:
 *   nvcc -o divergence divergence.cu
 */

#include <stdio.h>

__global__ void kernel_1(int *a) {
    int tid = threadIdx.x;
    int cell = tid % 32;
    a[cell]++;
}

__global__ void kernel_2(int *a) {
    int tid = threadIdx.x;
    int cell = tid % 32;
    switch(cell) {
    case 0:
      a[0]++;
      break;
    case 1:
      a[1]++;
      break;
    case 2:
      a[2]++;
      break;
    case 3:
      a[3]++;
      break;
    case 4:
      a[4]++;
      break;
    case 5:
      a[5]++;
      break;
    case 6:
      a[6]++;
      break;
    case 7:
      a[7]++;
      break;
    default:
      a[cell]++;
    }
}

void check(cudaError_t retVal) {
  //takes return value of a CUDA function and checks if it was an error
  if(retVal != cudaSuccess) {
    fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(retVal));
    exit(1);
  }
}

int main() {
  int a[32];   //array used for bucketing (on host)
  int *a_dev;  //GPU version
  
  //allocate the memory on the GPU
  check(cudaMalloc((void**)&a_dev, 32*sizeof(int)));
  
  //initialize a[i] to 0
  for (int i=0; i<32; i++)
    a[i] = 0;

  //allocate timers
  cudaEvent_t start;
  check(cudaEventCreate(&start));
  cudaEvent_t stop;
  check(cudaEventCreate(&stop));

  //start timer
  check(cudaEventRecord(start,0));

  //transfer array to the GPU
  check(cudaMemcpy(a_dev, a, 32*sizeof(int), cudaMemcpyHostToDevice));
  
  //call the kernel
  int threads = 512;
  int blocks = 512;
  kernel_2<<<blocks,threads>>>(a_dev);
  
  //transfer array back to the host
  check(cudaMemcpy(a, a_dev, 32*sizeof(int), cudaMemcpyDeviceToHost));

  //stop timer and print time
  check(cudaEventRecord(stop,0));
  check(cudaEventSynchronize(stop));
  float diff;
  check(cudaEventElapsedTime(&diff, start, stop));
  printf("time: %f ms\n", diff);
  
  /*
    //print bucket contents
    for(int i=0; i<32; i++)
    printf("%d ", a[i]);
    printf("\n");
  */
  
  //free memory and deallocate timers
  check(cudaFree(a_dev));
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(stop));
}
