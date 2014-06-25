.. Pi Using Numerical Integration: MPI documentation master file, created by
   sphinx-quickstart on Wed Jun 05 11:27:05 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pi Using Numerical Integration: MPI
===============================================================


Distributed memory bears lots of similarity on the surface to shared memory in that the work can be evenly divided between computational elements. The key detail to be aware of is that the shared memory thread requires few resources to put it into play; an MPI process requires significant care and feeding. The best example of this is if the task was to get a drink from the refrigerator, with that latency being the cost of a single cycle, i.e reaching the register file, then shared memory is like going out the the garage refrigerator. Going to a remote memory in another MPI process is like traveling 40 miles for the drink. The impact of this will not be seen in this code, but it does provide a framework a person use for adding a more representative CPU and bandwidth load.

The MPI version is also similar to the OpenMP, in that both the OpenMP thread and the MPI process learn of their portion of the work from their rank among all the other threads/processes. An MPI_Reduce is done at the end to perform a log(n) summation of the partial areas from each process.
 
Further Exploration
---------------------
  * What happens when you have more than one process running on a particular processors. How can  you know how many processors you have, or if you have more than one process running on a processor?

Complete Code
--------------


The full code can be downloaded here_.

.. _here: https://code.google.com/p/eapf-tech-pack-practicum/source/browse/trunk/pi_integration/pi_area_mpi.c

::

	/*  calculating pi via area under the curve
	 *  This code uses an algorithm fairly easily ported to all parallel methods.
	 *  Since it calculates pi, it is easy to verify that results are correct.
	 *  It can also be used to explore accuracy of results and techniques for managing error.
	 */
	
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>
	#include <mpi.h>
	
	#define NUMRECT 10000000
	
	/*  students learn in grammar school that the area of a circle is pi*radius*radius.
	 *  They learn in high school that the formula of a circle is x^2 + y^2 = radius^2.
	 *
	 *  These facts allows students calculating pi by estimating area of mid-point rectangles
	 *
	 *  Area of unit circle is pi, y = sqrt(1-x^2) is formula for semicircle from -1 to 1
	 */

	int main(int argc, char **argv) {
	
    	    int        numRect;                                    // number of rectangles
    	    int        i;                                          // loop index
    	    int        rank, size;      	 // MPI info used to know segment a process owns
    	    double        width;                            // width of each rectangle
    	    double        startingX = -1.0, overallWidth = 2.0;
    	    double        x;                                       // x value of midpoint
    	    double        pi, halfPi, partPi = 0.0;  // sum of area of rectangles gives pi/2
	
	
    	    numRect = argc == 2 ? atoi(argv[1]) : NUMRECT;        // get number of rectangles
	
	
    	    MPI_Init(&argc, &argv);
    	    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	
    	    numRect        /= size;                     // adjust so each process
    	    overallWidth        /= size;                 //    does its part of the curve
    	    startingX        += rank * overallWidth;    // Can then carry on serial cacl for that process
	
	
    	    width        = overallWidth / numRect;     // calculate width of each rectangle
    	    x = startingX - width/2;                        // setup for x to be at midpoint
	
	
    	    for (i=0; i<numRect; ++i) {                // calculate area of each rectangle
    	            x         += width;                
    	            partPi += width * sqrt(1.0 - x*x);
    	    }
	
	
    	    MPI_Reduce(&partPi, &halfPi, 1, MPI_DOUBLE,  MPI_SUM, 0, MPI_COMM_WORLD);        
	
	
    	    if (rank == 0) {
    	            pi = 2.0 * halfPi;
    	            printf ("\n==\n==\t%20s = %15.10f\n",    "pi",        pi);
    	            printf ("==\t%20s = %15d\n",       "total rectangles",    size*numRect);
    	            printf ("==\t%20s = %15d\n==\n\n",    "processes",          size);
    	    }
    	    MPI_Finalize();        
    	    return 0;
	}
  

