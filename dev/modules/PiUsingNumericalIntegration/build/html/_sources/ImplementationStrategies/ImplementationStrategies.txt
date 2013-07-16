**************************
Implementation Strategies
**************************

We provide several sample implementations below. The choice you make among these computations depends not only on which parallel platforms you have access to but also on various computational factors such as the size of your computation.  

A Sequential Implementation
----------------------------

We will develop and discuss parallel solutions to our pi-as-area problem in other modules. For now, we will present sequential code for a solution, as a reference point for the parallel solutions.

As suggested above, we can use a simple programming loop to add the areas of all the rectangles, as in the following C language code. The following loop carries out the bulk of the computation by adding the areas of all the rectangles. Here,  numRect  represents the number of rectangles.::

	sum = 0;
	width = 2.0 / numRect;
	for ( i = 0; i < numRect; i++ ) {
		x = -1 + ( i + 0.5 ) * width;
		sum += sqrt( 1 - x * x ) * width;
	}
	pi = sum*2.0;


In this code segment, sum , pi , and x are floating-point variables, where x represents an x value that determines the height of a rectangle, and we use the more self-explanatory names width and numRect in place of the symbols h and N in our mathematical explanation.

The program pi_area_serial.c_ implements the algorithmic strategy as outlined above in C++ language sequentially, i.e., with no (programmer-level) parallelism

.. _pi_area_serial.c: http://code.google.com/p/eapf-tech-pack-practicum/source/browse/trunk/pi_integration/pi_area_serial.c

*Further Exploration*:
  * The default number of rectangles is 10 million.  Time a run of the program with that default number of rectangles, then use the program’s optional command-line argument to compare with the timing for other rectangle counts, such as 100 million, 1 billion, 100,000, etc.  Do these timings vary as you might expect?
  * Are there any anomalies if you start from a small number of rectangle counts and keep doubling it? If so how do you explain them?
  * The x values in pi_area_serial.c are calculated by repeatedly adding the width of each rectangle. How would it change the results to instead calculate x as is done in the code snippet above?

  
Here is an alternative approach to computing the value of x for each rectangle::

	sum = 0;
	 width = 2.0 / numRect;
	x = -1 + width/2;
	for ( i = 0; i < numRect; i++ ) {
	     x += width;
	     sum += sqrt( 1 - x * x ) * width;
	}
	pi = sum*2.0;

Could there be a difference in the resulting value of π if we use this approach rather than the original approach for computing x ? Explain.

Complete listing of pi_area_serial.c_::

	/* Estimate pi as twice the area under a semicircle
	 Command-line arguments (optional, default values below).
	 1. first command line arg is integer number of rectangles to use
	 WARNING: minimal error checking is performed on this command-line arg */
	
	#include <stdio.h>
	#include <math.h>
	#include <stdlib.h>
	
	#define NUM_RECTANGLES 10000000;        /* default number of rectangles */

	int main(int argc, char** argv) {
	    double width;       /* width of a rectangle subinterval */ 
	    double x;           /* an x_i value for determining height of rectangle */
	    double sum;         /* accumulates areas all rectangles so far */
	    long numRect = NUM_RECTANGLES;       /* number of rectangles */
	    long i; 		/* loop control */
	
	/* parse command-line arguments, if any */
	if (argc > 1)
	   numRect = strtol(argv[1], NULL, 10);
	if (numRect <= 0) {
	   printf("Error in command-line argument\n");
	   return 1; /* indicates error exit */
	}
	
	/* compute the estimate */
	
	sum = 0;
	width = 2.0 / numRect;
	for ( i = 0; i < numRect; i++ ) {
	     x = -1 + ( i + 0.5 ) * width;
	     sum += sqrt( 1 - x * x ) * width;
	  }
	  pi = sum*2.0;
	  printf("With n = %d rectangles, ", numRect);
	  printf("the estimate of pi is %.20g\n", sum);
	  return 0;
	}

	 
