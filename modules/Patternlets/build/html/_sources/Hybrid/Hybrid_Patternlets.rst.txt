********************************************
Hybrid Patternlets in MPI and OpenMP
********************************************

Hybrid programming techniques are becoming more popular as larger clusters become
available. Such techniques are particularly useful when running very large problem
sizes. We will program in a hybrid environment, using MPI and OpenMP. This environment
will rely on both distributed memory from MPI and shared memory from OpenMP.

Pure MPI essentially has one MPI process on each core. A programming model
that combines MPI and OpenMP uses MPI to distribute work among cores, each
of which uses OpenMP to work on its task. In other words, MPI describes
parallelism between processes and the thread parallelism in OpenMP provides
a shared-memory model within a process. Conceptually, a program with 3 processes
with 2 threads each looks like this:

.. image:: Hybrid.png
	:width: 800

Hybrid Program Model
=========================
The MPI + OpenMP model generally includes these steps:

1. MPI Initialization
2. OMP parallel regions within MPI process
3. Finalize MPI

.. image:: HybridModel.png
	:width: 600

Source Code
************

Please download all examples from this tarball:
:download:`hybrid-MPI+OpenMP.tgz <../patternlets/hybrid-MPI+OpenMP.tgz>`

These C code examples use the Message Passing Interface (MPI) library and
OpenMP pragmas. MPI is suitable for use on either a single multicore
machine or a cluster of machines. See :doc:`../MessagePassing/RunningMPI` for more information
about running the code on a cluster of machines. You will need a C compiler
with OpenMP in order to compile and run the following examples.

A C code file for each example below can be
found in subdirectories of the hybrid-MPI+OpenMP directory, along with a makefile
and an example of how to execute the program.

Patternlets
************

:doc:`Spmd`

.. toctree::
	:hidden:

	Spmd
