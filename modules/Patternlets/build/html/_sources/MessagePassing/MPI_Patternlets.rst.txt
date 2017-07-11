*************************************
Message Passing Parallel Patternlets
*************************************

Parallel programs contain *patterns*:  code that recurs over and over again
in solutions to many problems.  The following examples show very simple
examples of small portions of
these patterns that can be combined to solve a problem.  These C code examples use the
Message Passing Interface (MPI) library, which is suitable for use on either a
single multiprocessor machine or a cluster
of machines.

Source Code
************

Please download all examples from this tarball:
:download:`MPI.tar.gz <../patternlets/MPI.tar.gz>`

A C code file for each example below can be found in subdirectories of the MPI directory,
along with a makefile and an example of how to execute the program.


Patternlets Grouped By Type
***************************

:doc:`MasterWorker_MessagePassing`

:doc:`DataDecomp_Broadcast`

:doc:`Reduction_Scatter_Gather`

:doc:`Barrier_Tags`

.. toctree::
	:hidden:

	MasterWorker_MessagePassing
	DataDecomp_Broadcast
	Reduction_Scatter_Gather
	Barrier_Tags
