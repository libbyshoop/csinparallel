.. Concept: The Data Decomposition Pattern documentation master file, created by
   sphinx-quickstart on Wed Jul 16 13:33:54 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   Author:  Libby Shoop, Macalester College

========================================
Concept: The Data Decomposition Pattern
========================================

**Prologue**

This document contains reading material that introduces a classic and ubiquitous
pattern used in parallel programs: **data decomposition**.
Programs containing this pattern perform computations over elements of data
stored in linear data structures (and potentially other types of data structures; but we will stick to linear here).  In non-parallel, or 'serial', or 'sequential' implementations of programs that contain linear data structures, we often iterate over all the data elements using a for loop. In parallel implementations, we need to decide which elements will be computed by multiple processing units at the same time. We dub the choices we make in our parallel implementation for achieving this the **data decomposition pattern**, because we will chose a decomposition, or mapping of elements in the data structure to processing units available to us.  We introduce an
example analogous to "Hello World" for this data decomposition programming pattern:
addition of vectors, using the simple linear array data structure.  We have code examples for different types of hardware and software that enable parallel computation.

The first two chapters introduce the problem and describe ways to decompose it onto processing units.  The next three chapters show examples of just how this is done for three different hardware and software combinations. We wrap up with alternative mapping schemes and some questions for reflection.


**Nomenclature**


A **Processing Unit** is an element of software that can execute instructions on hardware.  On a multicore computer, this would be a *thread* running on a core of the multicore chip.  On a cluster of computers, this would be a *process* running on one of the computers. On a co-processor such as a Graphics Processing Unit (GPU), this would be a *thread* running on one of its many cores.

A program that uses only one procesing unit is referred to as a *serial* or *sequential* solution. You are likely most familiar with these types of programs.

**Prerequisites**

- Knowledge of C programming language is helpful.

- Basic understanding of three types of parallel and distributed computing (PDC) hardware:
    - Shared-memory multicore machines
    - Clusters of multiple computers connected via high-speed networking
    - Co-processor devices, such as graphical processing units (GPU)
    
- Though not strictly necessary, if you want to compile and run the code examples, you will need unix-based computers with:
    - MPI installed on a single computer or cluster (either MPICH2 or OpenMPI)
    - gcc with OpenMP (most versions of gcc have this enabled already) and a multicore computer
    - CUDA developer libraries installed on a machine with a CUDA-capable GPU
    
We have compiled and run these on linux machines or clusters with a few different GPU cards.

**Code Examples**

You can download :download:`VectorAdd.tgz <code/VectorAdd.tgz>` to obtain all of the code examples shown in the following sections of this reading.


.. toctree::
    :maxdepth: 1
    
    VectorAdd/VectorAddProblem
    Decomposition/VecAddDecomposition
    Decomposition/MPI_VecAdd
    Decomposition/OpenMP_VecAdd
    Decomposition/CUDA_VecAdd
    Decomposition/Variations


.. comment
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search` 

