.. Pi Using Numerical Integration: ArBB documentation master file, created by
   sphinx-quickstart on Wed Jun 05 11:32:16 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pi Using Numerical Integration: ArBB
================================================================

The Intel® Array Building Blocks (ArBB) code is written to execute in a data-parallel fashion. No explicit loop is written since computations are applied to each element of data containers. This does mean that extra space must be allocated to hold values that would otherwise be generated dynamically by a for-loop.  The details are below, or can be `downloaded directly`_.

.. _`downloaded directly`: http://code.google.com/p/eapf-tech-pack-practicum/source/browse/trunk/pi_integration/pi_area_arbb.cpp

In the main() routine, the first line allocates a (dense) container of double-precision floating point numbers (f64), indices, and initializes it to hold the floats from 0.0 to num_rect-1 by strides of 1.0. This is the “simulation” of the for-loop seen in previous examples. The next line allocates a container to hold the computed area of each individual rectangle (as indexed by corresponding elements from the indices container). The arbb::call() function launches an execution of the calc_pi routine, sending as parameters the two previously allocated containers and the width of each rectangle to be used. The sole task of calc_pi() is to map the computation found in the calc_area() routine onto each element of the iVals container, then place the results into the corresponding location of the area container. The calc_area() function is written to handle a single rectangle. However, the runtime of ArBB will perform this computation on all elements of the iVals container and update the area container elements. Finally, the arbb::sums() function is used to perform an addition reduction of all elements within the area container, that sum is multiplied by 2.0, and the product is stored in the ArBB scalar pi. ::

	# include <iostream>
	# include <cstdlib>
	# include <arbb.hpp>
	
	const long num_rect=10485760;
	void calc_area(arbb::f64 &y, arbb::f64 h, arbb::f64 i)
	{
	       arbb::f64 x = -1.0f + (i+0.5f) * h;
	       y = arbb::sqrt(1.0 - x*x) * h ;
	}
	void calc_pi(arbb::dense<arbb::f64> &areas, arbb::f64 width, arbb::dense<arbb::f64> iVals)
	{
	       arbb::map(calc_area)(areas, width, iVals);
	}
	
	int main(int argc, char *argv[])
	{
	   arbb::dense<arbb::f64> iterations = arbb::indices(arbb::f64(0.0), num_rect, arbb::f64(1.0));
	   arbb::dense<arbb::f64> areas(num_rect);
	   arbb::f64 h = 1.0f / num_rect;
	   arbb::call(calc_pi)(areas, h, iterations);
	   arbb::f64 pi = arbb::sum(areas) * 2.0f;
	   std::cout << "Pi =" << arbb::value(pi) << std::endl;
	   return 0;
	}


