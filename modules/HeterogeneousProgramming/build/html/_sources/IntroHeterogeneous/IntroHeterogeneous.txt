Introduction to Heterogeneous Computing
=======================================

We know that we can decompose the task and distribute each smaller task to each worker, and let each node work simultaneously in an MPI program. Thus it enables a programmer to work on a larger problem size, and reduce the computational time. Similarly, CUDA C is a GPU programming language, and is another parallel programming model. Our goal of this module is to program in the hybrid environment CUDA and MPI.

It turns out that we can produce a new parallel programming model that combines MPI and CUDA together. The idea is that we use MPI to distribute work among nodes, each of which uses CUDA to work on its task. In order for this to work, we need to keep in mind that we need to have a cluster or something similar, where each node contains GPUs capable of doing CUDA computation.



.. centered:: Figure 1: Heterogeneous Programming Model: CUDA and MPI

.. todo:: Please Do Your Reading: `On the Comparative Performance of Parallel Algorithms on Small GPU/CUDA Clusters <http://www.hipc.org/hipc2009/documents/HIPCSS09Papers/1569247597.pdf>`_