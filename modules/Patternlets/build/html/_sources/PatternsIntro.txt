*****************************
Parallel Programming Patterns
*****************************

Like all programs, parallel programs contain many **patterns**: useful ways of writing code that are used repeatedly by most developers because they work well in practice.  These patterns have been documented by developers over time so that useful ways of organizing and writing good parallel code can be learned by new programmers (and even seasoned veterans).



An organization of parallel patterns
*************************************

When writing parallel programs, developers use patterns that can be grouped into two main categories:

1. Strategies
2. Concurrent Execution Mechanisms

Strategies
==========

When you set out to write a program, whether it is parallel or not, you should be considering two primary strategic considerations:

1. What *algorithmic strategies* to use
2. Given the algorithmic strategies, what *implementation strategies* to use

In the examples in this document we introduce some well-used patterns for both algorithmic strategies and implementation strategies.  Parallel algorithmic strategies primarily have to do with making choices about what tasks can be done concurrently by multiple processing units executing concurrently.  Parallel programs often make use of several patterns of implementation strategies.  Some of these patterns contribute to the overall structure of the program, and others are concerned with how the data that is being computed by multiple processing units is structured.  As you will see, the patternlets introduce more algorithmic strategy patterns and program structure implementation strategy patterns than data structure implementation strategy patterns.

Concurrent Execution Mechanisms
================================

There are a number of parallel code patterns that are closely related to the system or hardware that a program is being written for and the software library used to enable parallelism, or concurrent execution.  These *concurrent execution* patterns fall into two major categories:

1. *Process/Thread control* patterns, which dictate how the processing units of parallel execution on the hardware (either a process or a thread, depending on the hardware and software used) are controlled at run time.  For the patternlets described in this document, the software libraries that provide system parallelism have these patterns built into them, so they will be hidden from the programmer.

2. *Coordination* patterns, which set up how multiple concurrently running tasks on processing units coordinate to complete the parallel computation desired.

In parallel processing, most software uses one of
two major *coordination patterns*: 
	
	1. **message passing** between concurrent processes on either single multiprocessor machines or clusters of distributed computers, and 
	2. **mutual exclusion** between threads executing concurrently on a single shared memory system.  

These two types of computation are often realized using two very popular C/C++ libraries:

	1. MPI, or Message Passing Interface, for message passing.
	2. OpenMP for threaded, shared memory applications.

OpenMP is built on a lower-level POSIX library called Pthreads, which can also be used by itself on shared memory systems.


A third emerging type of parallel implementation involves a *hybrid computation* that uses both of the above patterns together, using a cluster of computers, each of which executes multiple threads.  This type of hybrid program often uses MPI and OpenMP together in one program, which runs on multiple computers in a cluster.

This document is split into chapters of examples.  There are examples for message passing using MPI and shared memory using OpenMP.
(In the future we will include shared memory examples using Pthreads, and hybrid computations using a combination of MPI and OpenMP.)

Most of the examples are illustrated
with the C programming language, using standard popular available libraries. In a few cases, C++
is used to illustrate a particular difference in code execution between the two languages or to make use of a C++ BigInt class.

There are many small examples that serve to illustrate a common pattern.  They are designed for you to try compiling and running on your own to see how they work.  For each example, there are comments within the code to guide you as you try them out.  In many cases, there may be code snippets that you can comment and/or uncomment to see how the execution of the code changes after you do so and re-compile it.

Depending on you interest, you can now explore MPI Patternlets or OpenMP Patternlets.


:doc:`MessagePassing/MPI_Patternlets`

:doc:`SharedMemory/OpenMP_Patternlets`




