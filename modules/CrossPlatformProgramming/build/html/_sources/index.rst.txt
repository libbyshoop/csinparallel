.. You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

****************************************
Cross Platform Parallel Programming
****************************************

This module illustrates how problems can be written across different platforms.
The document is split into various examples, each of which has a sequential version,
OpenMP version, MPI verison and MPI+OpenMP hybrid version. Code examples include
estimating pi using a Monte Carlo method and estimating pi using integration. Before
diving into the examples, there will be some background on shared memory,
distributed memory and hybrid systems. The examples are illustrated with
either C or C++ programming languages, using standard popular available libraries.

Source Code
************

Please download all examples from this tarball:
:download:`cross_platform_examples.tgz <cross_platform_examples/cross_platform_examples.tgz>`

A C code file for each example below can be found in subdirectories of the
cross_platform_examples directory, along with a makefile and an example of how
to execute the program with the exception of the Monte Carlo Pi example. This
example contains a C++ file and makefile. This was necessary as we needed a
reentrant function for random number generation.

**Hardware and Software Needed**

*OpenMP*: To compile and run the OpenMP and hybrid versions, you will need a C
compiler with OpenMP. The GNU C compiler is OpenMP compliant. We assume you are
building and executing these on a Unix command line.

*MPI*: For MPI and hybrid versions, you will either need access to a cluster of
computers or a multiprocessor machine with an MPI library installed in order
to be able to try the examples. If using a cluster of computers, you will need
a machine file specific to your cluster.

Examples
************
.. toctree::
  :caption: Table of Contents
  :titlesonly:

  CrossPlatform
  MonteCarloPi/Pi
  UnderCurvePi/Pi2

.. comment
	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`
