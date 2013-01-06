.. Parallel Patternlets documentation master file, created by
   sphinx-quickstart on Sat Jan  5 09:59:20 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Parallel Patternlets
=====================

This document contains simple examples of basic elements that are combined to form
patterns often used in
programs employing parallelism.  The examples are separated between
two major *coordination patterns*: 
	
	1. message passing used on clusters of distributed computers, and 
	2. mutual exclusion between threads executing concurrently on a single shared memory system.  

Both sets of examples are illustrated
with the C programming language, using standard popular available libraries.
The message passing example uses
a C library called MPI (Message Passing Interface).  The mutual Exclusion/shared memory
examples use the OpenMP library.

Source Code
************

Please download all examples from this tarball: 
:download:`patternlets.tgz <patternlets.tgz>`

Contents:
**********

.. toctree::
	:maxdepth: 1

	MessagePassing/MPI_Patternlets
	SharedMemory/OpenMP_Patternlets



.. comment
	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`

