========================================
Vector Add with MPI
========================================

Message passing is one way of distributing work to multiple *processes* that run indepentdently and concurrently on either a single computer or a cluster of computers. The processes, which are designated to start up in the software and are run by the operating system of the computer, serve as the processing units. This type of parallel programming has been used for quite some time and the software libraries that make it available follow a standard called Message Passing Interface, or MPI.

One feature of MPI programming is that one single program designates what all the various processes will do-- a single program runs on all the processes. Each process has a number or *rank*, and the value of the rank is used in the code to determine what each process will do.  In the following code, the process numbered 0 does some additional work that the other processes do not do. This is very common in message passing solutions, and process 0 is often referred to as the master, and the other processes are the workers.  In the code below, look for three places where a block of code starts with this line:

.. code-block:: c

    if (rank == MASTER)  {

This is where the master is doing special work that only needs to be done once by one process. In this case, it is the initialization of the arrays at the beginning of the computation, the check to ensure accuracy after the main computation of vector addition is completed, and freeing the memory for the arrays at the end.

The MPI syntax in this code takes some getting used to, but you should see the pattern of how the data decomposition is occuring for a process running this code:

1. First initialize your set of processes (the number of processes in designated when you run the code).
2. Determine how many processes there are working on the problem.
3. Determine which process rank I have.
4. If I am the master, initialze the data arrays.
5. Create smaller arrays for my portion of the work.
6. Scatter the equal-sized chunks of the larger original arrays from the master out to each process to work on.
7. Compute the vector addition on my chunk of data.
8. Gather the completed chunks of my array C and those of each process back onto the larger array on the master.
9. Terminate all the processes.

The following code contains comments with these numbered steps where they occur.  This is the file
**VectorAdd/MPI/VA-MPI-simple.c** in the compressed tar file of examples that accompanies this reading.

.. literalinclude:: ../code/VectorAdd/MPI/VA-MPI-simple.c	
    :language: c
    :linenos:
    :lines: 20-153

