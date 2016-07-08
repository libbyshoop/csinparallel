.. Drug Design in Parallel documentation master file, created by
   sphinx-quickstart on Thu Jun 13 13:14:56 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================
Drug Design in Parallel
=======================

This document contains several parallel programming solutions to the drug design exemplar using alternative parallel and distributed computing (PDC) technologies. We begin by describing a general solution with a simplification for educational purposes and provide a serial, or sequential version using this algorithm. Then we describe each of several parallel implementations that follow this general algorithm. The last chapter provides a discussion of the performance implications of the solutions and the parallel design patterns used in them.

An example on multiple parallel and distributed systems
*******************************************************

If you work through all of the versions of the code, you will be using different software libraries on different types of hardware:

- Single shared-memory multicore machines using the OpenMP library with C++11
- Single shared-memory multicore machines using the C++11 threads library
- Single shared-memory multicore machines using the Go programming language
- Distributed system clusters with several machines using the Message Passing Interface (MPI) library and C++11
- Distributed system clusters with Hadoop installed for map-reduce computing using a Java code example

You will need access to these hardware/software combinations in order to run each version.

Additional System Requirements
------------------------------

The following examples require that you have Threaded Building Blocks (TBB) installed on your system. This library from Intel is typically simple to install on linux systems (or you may already have it).

- The OpenMP version
- The C++11 threads version

To explore the Go language implementation, you will need to have Go installed on your system and know how to compile and run Go programs.

Code
*********

You can download a full set of code as :download:`dd.tar.gz <code/dd.tar.gz>`.
If you are going to work only with some individual versions, a link to the code and the Makefile (when appropriate) are given for each implementation. The sequential, OpenMP, MPI, and threads versions are written in C++11 and have a Makefile.

.. toctree::
   :maxdepth: 1

   intro/intro
   sequentialimplementation/sequentialimplementation
   openmp/openmp
   c++11threads/c++11threads
   mpi/mpi
   go/go
   hadoop/hadoop
   evaluation/evaluation
   lookingahead/lookingahead

.. comment
	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`
