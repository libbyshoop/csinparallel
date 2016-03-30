.. Pandemic with MPI documentation master file, created by
   sphinx-quickstart on Fri Jul 26 13:25:06 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pandemic with MPI
=================

In this module you will read about how we can model the spread of infectious diseases computationally.
We can make use of distributed computing with message passing to shorten the time needed to model very large populations,
which can be computationally intensive on a single computer.
Each section linked below explains both the computational model and the code, which uses the
Message Passing Interface Library, or MPI, to build a distributed processing solution.

.. toctree::
   :maxdepth: 1

   0-Introduction/introduction
   1-ProgramStructure/programstructure
   2-DataStructure/datastructure
   3-Initialize/initialize
   4-Infection/infection
   5-Display/display
   6-Core/core
   7-Finalize/finalize
   7.1-BuildAndRun/mpi_build
   8-OpenMP/openmp
   9-Cuda/cuda

Hitting the next links takes you from one chapter to another and previous takes you back one chapter.

.. comment
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`


