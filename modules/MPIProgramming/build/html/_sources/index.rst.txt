.. You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

****************************************
Message Passing with MPI
****************************************

This module illustrates how to use MPI to implement message passing programs for
distributed-memory systems. The document is split into chapters of examples.
Code examples include Monte Carlo version of calculating pi, integration using
the trapezoidal rule, sorting using odd even transposition and merge sort.
Before diving into the examples, first there will be some background on
distributed-memory systems and MPI. The examples are illustrated with the C
programming language, using standard popular available libraries.

Source Code
************

Please download all examples from this tarball:
:download:`MPI_examples.tgz <MPI_examples/MPI_examples.tgz>`

A C code file for each example below can be found in subdirectories of the MPI_examples
directory, along with a makefile and an example of how to execute the program with the
exception of the Monte Carlo Pi example. This example contains a C++ file and makefile.
This was necessary as we needed a reentrant function for random number generation.

**Hardware and Software Needed**

You will either need access to a cluster of computers or a multiprocessor machine
with an MPI library installed in order to be able to try the examples.
If using a cluster of computers, you will need a machine file specific to your cluster.

Examples
************
.. toctree::
  :caption: Table of Contents
  :titlesonly:

  GettingStartedWithMPI
  calculatePi/Pi
  trapezoidIntegration/trapezoid
  oddEvenSort/oddEven
  mergeSort/mergeSort

.. comment
	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`
