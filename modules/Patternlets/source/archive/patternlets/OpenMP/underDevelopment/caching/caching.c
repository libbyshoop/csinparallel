/* caching.c
 * ... shows how caching can make a big performance difference
 *
 * Joel Adams (Calvin College), November 2009.
 * Usage: ./caching
 * Exercise:
 *  - Does not work (yet).
 */

#include<stdio.h>
#include<string.h>   // memset()
#include<omp.h>

#define N 1024       // 512 works, 1024 does not

// initialize all elements of a matrix to a particular value
void init(int m[N][N], int value) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            m[i][j] = value;
        }
    }
}
 
// traditional multiply for NxN matrices
void multiply(int m1[N][N], int m2[N][N], int m3[N][N]) {
    int i, j, k, sum;
    double startTime = 0.0, 
           stopTime = 0.0,
           totalTime = 0.0;


    startTime = omp_get_wtime();

//    #pragma omp parallel for private(j, k)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = 0;
            for (k = 0; k < N; k++) {
                sum += m1[i][k] * m2[k][j];
            }
            m3[i][j] = sum;
        }
    }
    stopTime = omp_get_wtime();
    totalTime = stopTime - startTime;

    printf("Multiplying two %dx%d matrices took %0.12f\n\n",
            N, N, totalTime);
}

// display NxN matrix
void print(int m[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("\t%d", m[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main() {
    int a[N][N], 
        b[N][N], 
        c[N][N];
  
    init(a, 2);
    init(b, 2);

    multiply(a, b, c);

//    print(c);

    return 0;
}

