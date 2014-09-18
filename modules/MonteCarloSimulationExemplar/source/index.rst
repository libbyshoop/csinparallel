.. Monte Carlo Simulation Examplar documentation master file, created by
   sphinx-quickstart on Tue Jul  1 17:27:03 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Monte Carlo Simulation Exemplar
===========================================================

Text by Devin Bjelland and Libby Shoop, Macalester College

Accompanying instructor videos by Dave Valentine, Slippery Rock University

Monte Carlo simulations are a class of algorithms that are quite easy
to convert from their original sequential solutions to corresponding parallel
or distributed solutions that run much faster.  This module introduces these
type of algorithms, providing some examples with C++ code for both the original
sequential version and the parallelized OpenMP version.

**Hardware and Software Needed**

    - You will need access to a multicore computer with a C/C++ compiler that enables compilation with OpenMP.
    - If you want to try some of the other examples for Message Passing using MPI you will need access to a cluster of computers with an MPI library installed.
    - If you want to try some of the other examples for CUDA on GPUs, you will need acces to a computer with a CUDA-capable nVIDIA GPU and you will need the nVIDIA CUDA Toolkit installed.

This document contains several sections with example C++ code to explain Monte Carlo methods
and how to parallelize them using the OpenMP library.  The last three sections contain
exercises that you can try and explain a more advanced topic for ensuring greater accuracy.

.. toctree::
   :maxdepth: 1

   Introduction/Introduction
   Introduction/CoinFlip
   Threads/Threads_OMP
   Threads/OpenMP_CoinFlip
   RouletteSimulation/RouletteSimulation
   DrawFourSuitsExample/DrawFourSuitsExample
   Plinko/PlinkoGame.rst
   NextSteps/Exercises
   SeedingThreads/SeedEachThread

.. comment
  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`

