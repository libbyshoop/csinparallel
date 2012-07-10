#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define Width 1024

int main(void){

  int size = Width * Width * sizeof(float);
  float *M, *N, *P;

  // Allocate the matrices M,N,P
  M = (float*)malloc(size);
  N = (float*)malloc(size);
  P = (float*)malloc(size);

  // fill in the matrices with data
  for (int y=0; y<Width; y++) {
    for (int x=0; x<Width; x++){
      M[y*Width + x] = x + y*Width;
      N[y*Width + x] = x + y*Width; 
    }
  }

  // capture the start time
  struct timeval timer_start, timer_end;
  gettimeofday(&timer_start, NULL);

  // Matrix Multiplication Computation
  for (int i = 0; i < Width; i++){
    for (int j = 0; j < Width; j++){
      float sum = 0;
      for (int k = 0; k < Width; k++){
        sum += M[i * Width + k] * N[k * Width + j];
      }
      P[i * Width + j] = sum;
    }
  }
    
  // Capture the stop time and display the time to generate
  gettimeofday(&timer_end, NULL);
  double timer_spent = timer_end.tv_sec - timer_start.tv_sec +
         (timer_end.tv_usec - timer_start.tv_usec) / 1000000.0;
  printf("Time spent: %.6f\n", timer_spent);

  // free memory allocated to matrices M,N,P
  free( M );
  free( N );
  free( P );

  return 0;
}
