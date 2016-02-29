=============================================
Patterns used when threads share data values
=============================================


11. Shared Data Algorithm Strategy: Parallel-for-loop pattern needs non-shared, private variables
****************************************************************************************************

*file: openMP/11.private/private.c*

*Build inside 11.private directory:*
::
	make private

*Execute on the command line inside 11.private directory:*
::
	./private

In this example, you will try a parallel for loop where additional variables (i, j in the code) cannot be shared by all of the threads, but must instead be *private* to each thread, which means that each thread has its own copy of that variable.  In this case, the outer loop is being split into chunks and given to each thread, but the inner loop is being executed by each thread for each of the elements in its chunk.  The loop counting variables must be maintained separately by each thread.  Because they were initially declared outside the loops at the begininning of the program, by default these variables are shared by all the threads.

.. literalinclude::
	../patternlets/openMP/11.private/private.c
    :language: c
    :linenos:



12. Race Condition: missing the mutual exclusion coordination pattern
*********************************************************************

*file: openMP/12.mutualExclusion-atomic/atomic.c*

*Build inside 12.mutualExclusion-atomic directory:*
::
	make atomic

*Execute on the command line inside 12.mutualExclusion-atomic directory:*
::
	./atomic

When a variable must be shared by all the threads, as in this example below, an issue called a *race condition* can occur when the threads are updating that variable concurrently.  This happens because there are multiple underlying machine instructions needed to complete the update of the memory location and each thread must execute all of them atomically before another thread does so, thus ensuring **mutual exclusion** between the threads when updating a shared variable.  This is done using the OpenMP pragma shown in this code.

.. literalinclude::
	../patternlets/openMP/12.mutualExclusion-atomic/atomic.c
    :language: c
    :linenos:


13. The Mutual Exclusion Coordination Pattern: two ways to ensure
******************************************************************

*file: openMP/13.mutualExclusion-critical/critical.c*

*Build inside 13.mutualExclusion-critical directory:*
::
	make critical

*Execute on the command line inside 13.mutualExclusion-critical directory:*
::
	./critical

Here is another way to ensure **mutual exclusion** in OpenMP.

.. literalinclude::
	../patternlets/openMP/13.mutualExclusion-critical/critical.c
    :language: c
    :linenos:


14.  Mutual Exclusion Coordination Pattern: compare performance
****************************************************************

*file: openMP/14.mutualExclusion-critical2/critical2.c*

*Build inside 14.mutualExclusion-critical2 directory:*
::
	make critical2

*Execute on the command line inside 14.mutualExclusion-critical2 directory:*
::
	./critical2

Here is an example of how to compare the performance of using the atomic pragma directive and the critical pragma directive.  Note that there is a function in OpenMP that lets you obtain the current time, which enables us to determine how long it took to run a particular section of our program.

.. literalinclude::
	../patternlets/openMP/14.mutualExclusion-critical2/critical2.c
    :language: c
    :linenos:


15.  Mutual Exclusion Coordination Pattern: language difference
***************************************************************

*file: openMP/15.mutualExclusion-critical3/critical3.c*

*Build inside 15.mutualExclusion-critical3 directory:*
::
	make critical3

*Execute on the command line inside 15.mutualExclusion-critical3 directory:*
::
	./critical3

The following is a C++ code example to illustrate some language differences between C and C++.  Try the exercises described in the code below.

.. literalinclude::
	../patternlets/openMP/15.mutualExclusion-critical3/critical3.cpp
    :language: c++
    :linenos:


Some Explanation
=================

A C line like this:


     printf("Hello from thread #%d of %d\n", id, numThreads);

is a single function call that is pretty much performed atomically, so you get pretty good output like.  
::
   Hello from thread #0 of 4
   Hello from thread #2 of 4
   Hello from thread #3 of 4
   Hello from thread #1 of 4

By contrast, the C++ line:


     cout << "Hello from thread #" << id << " of " << numThreads << endl;

has 5 different function calls, so the outputs from these functions get interleaved within the shared stream cout as the threads 'race' to write to it.  You may have observed output similar to this:


  Hello from thread #Hello from thread#Hello from thread#0 of 4Hello from thread#

  2 of 43 of 4

  1 of 4

The other facet that this particular patternlet shows is that OpenMP's atomic directive will not fix this -- it is too complex for atomic, so the compiler flags that as an error.  To make this statement execute indivisibly, you need to use the critical directive, providing a pretty simple case where critical works and atomic does not.

