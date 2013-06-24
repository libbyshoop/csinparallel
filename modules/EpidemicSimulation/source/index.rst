.. Epidemic Simulation documentation master file, created by
   sphinx-quickstart on Thu Jun 13 13:47:49 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Epidemic Simulation
===================

.. toctree::
   :maxdepth: 1

   intro/intro
   cppspecs/cppspecs
   parallelizing/parallelizingcpp
   parallelizing/parallelizingseqgocode
   parallelizing/parallelizinggowalkthrough
   parallelizing/parallelizingadvanced
   testing/testing

Note on this module: there are three different versions of instructions for parallelizing in Go. 

   - The first gives sequential Go code, thoroughly commented for both Go style and the program's overall logic, and simply asks students to parallelize it. This could be used in conjunction with the C++ version or not, as desired.

   - The second gives pretty detailed specs and a walkthrough for writing them up, directed at students who have either Python or C++ experience and haven't necessarily written a C++ version of this code already (and then a more thorough set of parallelizing directions). 

   - The third version is for students with some programming experience who are porting their existing C++ code into Go; it merely highlights some key structural and syntactic differences between the two languages and then lists resources for figuring out the rest.

.. comment
	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`
