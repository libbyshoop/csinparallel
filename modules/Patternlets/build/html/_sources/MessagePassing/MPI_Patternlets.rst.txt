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
If you are working on these for the first time, you may want to visit them in
order. If you are returning to review a particular patternlet or the pattern
categorization diagram, you can refer to them individually.

:doc:`RunningMPI`

:doc:`ProgramStructure`

:doc:`Communication`

:doc:`Broadcast`

:doc:`Reduction_Scatter_Gather`

:doc:`Barrier_Tags`

:doc:`AdvancedTopics`

.. toctree::
	:hidden:

	RunningMPI
	ProgramStructure
	Communication
	Broadcast
	Reduction_Scatter_Gather
	Barrier_Tags
	AdvancedTopics
