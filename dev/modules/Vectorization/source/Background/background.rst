About Vectorization
===================

Introduction
------------

Especially in scientific computing, there are many instances where the same operation is being performed on an entire data set or subset. In these instances, it is often useful to apply the same operation to multiple pieces of data in parallel. This is often referred to as vectorization.

What is Vectorization
---------------------

Vectorization is the act of applying the same instruction to multiple pieces of data in parallel. This is often referred to a *Single instruction, multiple data* (SIMD). Consider the following ``for`` loop::

    for(i=0; i<= MAX; i++)
        c[i] = a[i] + b[i];

On a normal processor, this is implemented as follows.

.. image:: ScalarAddition.png
..
    :width: 100px
    :align: left
    :height: 100px
    :alt: alt-text
..
    :figclass:none
    Caption (if we use the figure directive)

Note all the wasted, unused memory locations.

Whene compiled on a vector processor, the same for loop above is implemented as follows:

.. image:: VectorAddition.png

.. note:: TODO: Do we want to put in a section that is a bit more hardware-focussed

Using Vectorization
-------------------

Vectorizing existing code is usually straightforward. At its simplest, vectorization does not require complicated data structures or an extensive library. Even without any changes to existing code, the Intel C++ may automatically add vectorized instructions where it can detect that doing so will not change the operation of the code (there are no data dependencies or race conditions). It can provide you feedback about what it has vectorized, what not, and why.

Often, however, either more code can be vectorized or the code can be vectorized more efficiently if you use various techniques to modify the computation slightly.

Also, Vectorization integrates easily with other, higher-level forms of parallelism, such as message-passing or shared memory.
