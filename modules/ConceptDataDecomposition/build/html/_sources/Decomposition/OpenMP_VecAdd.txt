========================================
Vector Add with OpenMP
========================================

Computers with multicore processors and a single shared memory space are the norm, including not only laptops and desktops, but also most phones and tablets. Using multiple cores concurrently on these machines, can be done in several programming languages; we will demonstrate the use of C with a set of compiler directives and library functions known as OpenMP.  The OpenMP standard is built into many C compilers, including gcc on unix machines.

OpenMP on shared memory multicore machines creates *threads* that execute concurrently.  The creation of these threads is implicit and built by the compiler when you insert special directives in the C code called *pragmas*.
The code that begins executing main() is considered thread 0. At certain points in the code, you can designate that more threads should be used in parallel and exucute concurrently. This is called *forking* threads.

In the code below, you will see this pragma, which is implicitly forking the threads to complete the computation on equal chunks of the orginal array:

.. code-block:: c

    #pragma omp parallel for shared(a, b, c) private(i) schedule(static, 2)

The ``shared`` keyword indicates that the arrays are shared in the same memory space for all threads, and the ``private`` keyword indicates that each thread will have its own copy of the index counter i that it will increment.

The ``schedule`` keyword is used in this pragma to indicate how many consecutive iterations of the loop, and thus computations on consecutive elements of the arrays, that each thread will execute.  In data decomposition, we like to call this the **chunk size** assigned to each thread (not necessarily a universal term, but  hopefully it conveys the idea). To mimic our simple 8-element example, this code (shown below) sets the number of threads to 4 and the chunk size to 2.

The syntax of this OpenMP code example below is very similar to the original sequential version. In fact, it was derived from the sequential version by adding this pragma, including the OpenMP library, called omp.h, setting how many threads to use and the chuck size just before the forking, and adding some print statements to illustrate the decomposition and verify the results.

This pragma around for loops is built into openMP because this 'repeat N times" pattern occurs so frequently in a great deal of code. This simplicity can be deceiving, however-- this particular example lends itself well to having the threads share data, but other types of problems are not this simple.  This type of data decomposition example is sometimes called *embarassingly parallel*, because each thread can read and update data that no other thread should ever touch.

This code is the file
**VectorAdd/OpenMP/VA-OMP-simple.c** in the compressed tar file of examples that accompanies this reading.

.. literalinclude:: ../code/VectorAdd/OpenMP/VA-OMP-simple.c	
    :language: c
    :linenos:


