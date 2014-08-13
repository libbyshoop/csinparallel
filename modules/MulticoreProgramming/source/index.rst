.. Programming with Multiple Cores documentation master file, created by
   sphinx-quickstart on Tue Jun 26 16:07:18 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Multicore Programming with OpenMP
==================================

This module depicts how to use OpenMP to solve a fairly simple mathematical calculation: estimating the area under a curve as a summation of trapezoids.  Some issues that arise when using multiple threads to complete this task are introduced.  Once you have working code, we discuss how you can estimate the *speedup* you obtain using varying numbers of threads and have you think about how scaleable this problem is as you increase the number of trapezoids (the problem size) and use more threads.

.. toctree::
   :maxdepth: 1

   introOpenMP/GettingstartedwithOpenMP
   correctOpenMP/FixingOurOpenMPcode
   timingAndScalability/TimingOnMTLandscalability


.. commented out for now
	Indices and tables
	==================

	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`
..
