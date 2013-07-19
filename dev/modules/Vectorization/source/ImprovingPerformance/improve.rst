Improving Performance
=====================

Pointer Aliasing
----------------

If the compiler sees two pointers to the same memory location, it may skip vectorizing the loop if it sees an opportunity for race conditions.

Removing pointer aliasing helps ensure that loop dependencies between iterations do not exist.

.. note:: TODO: Give example

Aligning Data
-------------

By aligning the data on 16 byte boundaries in memory, the compiler can use faster aligned load instructions.

One keyword that can be used in the ``__attribute__`` keyword::

    float array[30] __attribute___((aligned(base, [offset])));

One can also use the macro ``pragma vector aligned`` before the vectorizable loop to get maximum speedup from data alignment.

Function Inlining
-----------------

By enabling the "interprocedural optimization" option, the compiler may be able to further optimize through automatic function inling.

This is enabled with the *-ipo* compiler flag.

Non-Contiguous Memory Acccess
-----------------------------

Loops with non-unit stride or indirect addressing may be inefficient to vectorize::

    for (int i=0; i < SIZE; i+=2)
        b[i] += a[i] * x[index[i]]

With contiguous memory access, multiple consecutive ints, floats or doubles can be loaded simultaneously.

Data Dependencies
-----------------

Vectorization is possible when changing the order of operations in a loop does not change the result. For example, Read after Write, as in ``for (j=1, j<MAX; j++) A[j] = A[j - 1] + 1;`` cannot be vectorized. On the other hand, Write after Read, for example ``for (j=1; j < MAX; j++) A[j-1] = A[j] + 1;`` can be vectorized.

In this loop, "No iteration with a higher value of j can complete before an iteration with a lower value of j, so vectorization is safe" -- despite that this sort of loop is not usually parallelizable.

Loops, Pointers, and Data Structures: Do's and Don'ts
---------------------------------------------------------

One should avoid:

* Complex or variable termination conditions for loops
* ``switch``, ``goto``, or ``return`` statements in loops
* Function calls 

One should use:

* The loop index as the array index when possible
* Array notation instead of pointers or separately incremented indices
* A structure of arrays instead of an array of structures.

Pragmas
-------

::
  
    #pragma ivdep  

This tells the compiler to ignore *potential* data dependencies - it will not ignore proven ones.::

    # pragma vector always

This tells the compiler to always vectorize a loop if is is safe, even if it is not efficient. ::

    #pragma novector

This tells the compiler not to vectorize the following loop.
