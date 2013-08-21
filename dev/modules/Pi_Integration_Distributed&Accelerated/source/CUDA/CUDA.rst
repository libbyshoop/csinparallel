================================================================
Pi Using Numerical Integration: CUDA
================================================================

Several implementations have a reduction operator that can combine partial results in log(n) time. CUDA has a more complicated memory model, namely a group of blocks,  each containing a group of synchronizable threads. Each thread gets its work as with shared and distributed memory, with a slight difference being the division of labor needing to be first  transferred from CPU memory to shared CUDA memory. The CUDA thread also has parallel reduce of thread results within a block. The reduction becomes starved of actual work, with only one thread performing the final add, but is overall done in log(n) time, so it is still a win. The block results are then transferred from CUDA memory to CPU memory, where a final linear reduction is performed.  There is a `sample implementation`_ available.

.. _`sample implementation`: https://code.google.com/p/eapf-tech-pack-practicum/source/browse/trunk/pi_integration/pi_area_cuda.cu


Further Exploration
---------------------

  * The code uses 32 blocks per grid and 256 threads per block. Must these numbers be used? What are advantages/disadvantages of changing them? Is the ratio between theses numbers significant?
  * This code uses floats. What differences do you see with the other area under the curve codes? How can you affect any differences while still using floats? Can you use doubles with CUDA? If so, how do you test when you can?
  * The x values in pi_area_serial.c are calculated by repeatedly adding the width of each rectangle. How would it change the results to instead calculate x as is done in the code snippet above?
  * There is a coalescing of thread values within a block by repeated passes that starve down to an effective length of 1. Was this a wise choice? Why wasn’t the same technique used for all threads at once, and not just block by block for the threads within that block?

Complete Code
--------------

::
	
	 
	/*  calculating pi via area under the curve
	 *  This code uses an algorithm fairly easily ported to all parallel methods.
	 *  Since it calculates pi, it is easy to verify that results are correct.
	 *  It can also be used to explore accuracy of results and techniques for managing error.
	 */
	
	#include <stdio.h>
	#include <stdlib.h>
	#include <cuda.h>
	
	
	#define NUMRECT 10000000
	
	
	/*  students learn in grammar school that the area of a circle is pi*radius*radius.
