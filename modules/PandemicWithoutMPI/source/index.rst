.. Pandemic Without MPI documentation master file, created by
   sphinx-quickstart on Wed Jul 24 09:40:33 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pandemic Modeling Exemplar using OpenMP 
================================================

This example contains a fully functional simulation of the type of modeling done by epidemiologists to answer the question: what happens when an infectious disease hits a population?  Code for an original serial version is provided and described in detail.  Next, descriptions of a parallel version using OpenMP is provided, where the code is modified from the original serial version.

**Acknowledgment**: Many thanks to Aaron Weeden of the Shodor Foundation for the original version of this material and code.

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
   7.1-BuildAndRun/build.rst
   8-OpenMP/openmp
   8.1-ompBuildAndRun/omp_build.rst
..   9-Cuda/cuda
  
.. comment
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

