/****************************************************************************
* Program: Sequential Pi Calculation
* This PI program was taken from Argonne National Laboratory.
* Sequential version made by Hannah Sonsalla, Macalester College
****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void getInput(int argc, char* argv[], int* n);

void getInput(int argc, char* argv[], int* n){
	if(argc != 2){
		fprintf(stderr, "usage: %s <number of bins> \n", argv[0]);
        fflush(stderr);
        *n = -1;
	} else {
		*n = atoi(argv[1]);
	}

    if (*n <= 0) {
        exit(-1);
    }
}

int main(int argc, char *argv[]) {

   int  i,
        n;                                              /* the number of bins */

    double PI25DT = 3.141592653589793238462643;         /* 25-digit-PI*/
  
    double pi,                                          /* value of PI in total*/
           step,                                        /* the step */
           sum,                                         /* sum of area under the curve */
           x;

    clock_t start_time,          /* starting time */
           end_time;            /* ending time */

    getInput(argc, argv, &n);
    printf("number of bins is %d\n", n);
    start_time = clock();

    step = 1.0 / (double) n;
    sum = 0.0;
    for (i = 1; i <= n; i += 1) {
        x = step * ((double)i - 0.5);
        sum += (4.0/(1.0 + x*x));
    }

    pi = step * sum;

    printf("Pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
    end_time = clock();
    printf("Time of calculating PI is: %f seconds \n", (double)(end_time-start_time)/CLOCKS_PER_SEC);
   
    return 0;
}
