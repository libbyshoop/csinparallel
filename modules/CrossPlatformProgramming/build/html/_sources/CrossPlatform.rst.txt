
***********************************************
Cross Platform Programming
***********************************************

Shared Memory
***********************************************
Writing

Multiprocessor Machines and Distributed Memory
***********************************************
Rework writing ---

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


Hybrid Systems
***********************************************
Writing

Examples
***********************************************
Depending on your interest, you can now explore the below examples.

:doc:`MonteCarloPi/Pi`

:doc:`UnderCurvePi/Pi2`
