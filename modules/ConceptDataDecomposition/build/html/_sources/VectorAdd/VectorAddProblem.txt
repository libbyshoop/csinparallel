*******************
Vector Addition
*******************

The problem we will examine is this: we wish to add each element of vector A with its corresponding element in vector B and placing the sum of the two elements in its corresponding location in vector C.  This example problem has sometimes been called the "Hello, World" of parallel programming. The reasons for this are:

- the sequential implementation of the code is easy to understand 
- the pattern we employ to split the work is used many times in many situations in parallel programming

Once you understand the concept of splitting the work that we illustrate in this example, you should be able to use this pattern in other situations you encounter.

The problem is quite simple and can be illustrated as follows:

.. figure:: VectorAdditionProblem.png
    :width: 600px
    :align: center
    :alt: Vector Addition Illustration
    :figclass: align-center

We have two arrays, A and B, and array C will contain the addition of corresponding elements in A and B. In this simple example we are illustrating very small arrays containing 8 elements each.  Suppose those elements are integers and A and B had the following elements in it:

.. figure:: VecAddSolution.png
    :width: 600px
    :align: center
    :alt: A has integers 1,2,3,1,4,1,6,7. B has integers 1,2,3,1,5,2,6,1. C has 2,4,6,2,9,3,12,8.
    :figclass: align-center
    
The elements in array C above depict the result of adding a vector as stored in array A to a vector as stored in array  B.

We use very small vectors of size 8 for illustration purposes. A sequential solution to this problem, written in C code, is found in the file named **VectorAdd/Serial/VA-sequetial.c** in the compressed tar file of examples that accompanies this reading. It looks like this:

.. literalinclude:: ../code/VectorAdd/Serial/VA-sequential.c	
    :language: c
    :linenos:
    
Note the for loop that is doing the actual work we desire, beginning on line 35. This depicts what we sometimes refer to as the 'do N times' pattern in classical sequential programming.  In the next section we will describe how we consider using multiple processing units to do this work in parallel.