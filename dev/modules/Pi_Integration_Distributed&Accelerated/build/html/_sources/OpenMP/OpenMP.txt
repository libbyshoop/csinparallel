========================================
Pi Using Numerical Integration: Open MP
========================================


This is the simplest parallelization strategy we will consider for this problem.  The example inserts one OpenMP pragma to parallelize the loop for computing areas of rectangles and summing those areas, as shown below. In this case, the loop iterations must be independent in order to execute in parallel.Notice how the computation of the midpoint for each rectangle, x, is done within the loop rather than relying on adding the width of each segment from the initial midpoint value (in the downloadable serial code). ::

	 sum = 0;
	 h = 2.0 / n_rect;
	 /* NOTE: i is automatically private, and n_rect, and h are shared */
	 #pragma omp parallel for private( x ) reduction(+: sum)
	 for( i = 0; i < n_rect; i++ ) {
	     x = -1 + ( i + 0.5 ) * h;
	     sum += sqrt( 1 - x * x ) * h;
	 }
	 pi = sum*2.0;

*Comments on the code*:
  * None of the OpenMP threads in the parallelized  for  loop will modify any of the variables  n_rect   or  h   so it is safe for those variables to be shared among the threads.  
  * The   omp parallel for  pragma parallelizes a  for  loop by giving different threads their own subrange of values of the loop-control variable ( i  in this case).  Hence, that variable  i  is automatically private or local to each thread. The “work” variable, x, must be private to each thread or a race condition will result. This can be done through the private() clause or declaring the x variable within the parallel region (within the for-loop).
  * The variable  sum  (which holds a partial sum of rectangle areas) must be computed locally for each thread, in order to avoid a race condition.  As each thread finishes its work, its resulting value of  sum  (representing that thread’s subtotal of areas of rectangles) must be added to a grand total in order to produce the correct (global) value of  sum  after the  for  loop is finished.  This subtotalling/grand totalling procedure is accomplished using an OpenMP  reduction()  clause. The calues also ensures that threads use a local version of sum.
  * The number of OpenMP threads used does *not* need to match the number of cores on your system.  The number of threads can be controlled by setting the environment variable OMP_NUM_THREADS at runtime.  The function omp_set_num_threads() may be used to set the number of threads from within the code.

Further Exploration
---------------------
Download the serial_ and OpenMP_ codes. Build and run them. Compare the timing results you collected for the sequential program to the time performance of this parallel program using various numbers of threads using OMP_NUM_THREADS.  Does the parallel program perform better?  Is the speed up as much as you would expect?  If not, can you hypothesize why?  

.. _serial: 
.. _OpenMP: https://code.google.com/p/eapf-tech-pack-practicum/source/browse/trunk/pi_integration/pi_area_omp.c

Complete Code
----------------
::

	/* Estimate pi as twice the area under a semicircle
 	 Command-line arguments (optional, default values below).
 	  1. first command line arg is integer number of rectangles to use
 	  2. if two command-line args, second arg is number of OpenMP threads
 	 WARNING:  minimal error checking is performed on these command-line args */
 
	#include <stdio.h>
	#include <math.h>
	#include <stdlib.h>
	
	/* parameters that may be overridden on the command-line */
	long n_rect = 10485760;  	/* default number of rectangles */
	int threadct = 8;        	/* default number of OpenMP threads to use */
 
	int main(int argc, char** argv) {
	double a = -1.0, b = 1.0;  /* lower and upper interval endpoints */
	double h;              	/* width of a rectangle subinterval */
	double f(double x);    	/* declare function that defines curve */
	double sum;            	/* accumulates areas all rectangles so far */
	long i;  /* loop control */
	 /* parse command-line arguments, if any */
	if (argc > 1)
    	n_rect = strtol(argv[1], NULL, 10);
	if (argc > 2)
	    threadct = strtol(argv[2], NULL, 10);
	if (n_rect <= 0 || threadct <= 0) {
   	    printf("Error in command-line argument(s)\n");
   		return 1;  /* indicates error exit */
	}
 
	h = (b - a)/n_rect; 
 
	/* compute the estimate */
 	 sum = 0;
	#pragma omp parallel for num_threads(threadct) \
	shared (a, n_rect, h) private(i) reduction(+: sum)
	for (i = 0; i < n_rect; i++) {
   	     sum += f(a + (i+0.5)*h) * h;
	}
 	printf("With n = %d rectangles and %d threads, ", n_rect, threadct);
	printf("the estimate of pi is %.20g\n", sum);
	return 0;
	}
  
	double f(double x) {
	return 2*sqrt(1-x*x);
	}


