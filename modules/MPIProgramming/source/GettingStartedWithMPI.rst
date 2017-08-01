
***********************************************
Getting Started with Message Passing using MPI
***********************************************

Multiprocessor Machines and Distributed Memory
***********************************************
A single system that uses two or more central processing units (CPUs) is considered
multiprocessing. A multiprocessor computer system with distributed memory contains
core-memory pairs that are connected by a network. Each core has its own private
memory that is only accessible to that core. It is important to note that a process
is essentially one core-memory pair. Thus, two processes can communicate by sending and
receiving messages (functions). In the examples, we will learn how
to use a library called MPI to enable us to write programs that can use multicore
processors and distributed memory to write programs that can complete a task
faster by taking advantage of message-passing.

The key idea for programming distributed memory systems is to determine how to
allocate the tasks across processes. Unlike shared memory systems, race conditions
are not present in distributed memory systems. Throughout the examples, we will
see how we can use “task parallelism” to execute the same task on different
parts of a desired computation in processes and gather the results when each
process is finished.

Getting Started with MPI
============================
We will use a standard message-passing system for parallel programming
called MPI (Message-Passing Interface). The functions defined by the MPI
library are callable from C, C++ and Fortran. This interface is a useful
tool for helping to facilitate synchronization and communication between
processes. MPI communications fall under two categories: point-to-point
and collective. Point-to-point functions involve communication between
two specific processes whereas collective communication functions involve
communication among all processes. Typically, each CPU or core in a
multicore machine is assigned only one process in an MPI program.

Depending on your interest, you can now explore the below examples.

:doc:`calculatePi/Pi`

:doc:`trapezoidIntegration/trapezoid`

:doc:`oddEvenSort/oddEven`

:doc:`mergeSort/mergeSort`
