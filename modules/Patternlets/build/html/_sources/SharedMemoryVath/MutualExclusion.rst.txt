=============================================
Patterns used when threads share data values
=============================================


10. Shared Data Algorithm Strategy: Parallel-for-loop pattern needs non-shared, private variables
****************************************************************************************************

*file: Vath_pth/10.private/private.C*

*Build inside 10.private directory:*
::

	make private

*Execute on the command line inside 09.private directory:*
::

	./private

In this example, you will try a parallel for loop where variables (beg, end in the code)
cannot be shared by all of the threads, but must instead be *private* to each thread, which means
that each thread has its own copy of that variable. In this case, the outer loop is being
split into chunks and given to each thread, but the inner loop is being executed by each
thread for each of the elements in its chunk. The beginning and end chunk variables must be maintained
separately by each thread. Because they were initially declared outside the thread function at the
beginning of the program, by default these variables are shared by all the threads.

.. literalinclude::
	../patternlets/Vath_pth/10.private/private.C
  :language: c++
  :linenos:


11. Race Condition: missing the mutual exclusion coordination pattern
*********************************************************************

*file: Vath_pth/11.raceCondition/raceCondition.C*

*Build inside 11.raceCondition directory:*
::

  make raceCondition

*Execute on the command line inside 11.raceCondition directory:*
::

  ./raceCondition

When a variable must be shared by all the threads, as in this example below, an issue
called a *race condition* can occur when the threads are updating that variable concurrently.
This happens because there are multiple underlying machine instructions needed to
complete the update of the memory location and each thread must execute all of them
atomically before another thread does so, thus ensuring **mutual exclusion** between
the threads when updating a shared variable.

Atomic operations are lock-free algorithms that attempt to go ahead and run
the program with threads executing in parallel. If a race condition occurs,
it is necessary to start over. Note that atomic operations may perform redundant work.
In contrast, reduction ensures mutual exclusion and is considered pessimistic. Since
a race condition could possibly happen, reduction makes sure it never happens
by using mutex locks. In Pthreads, there are no atomic services so we will stick
with lock reduction.

.. literalinclude::
  ../patternlets/Vath_pth/11.raceCondition/raceCondition.C
  :language: c++
  :linenos:


12.  Mutual Exclusion Coordination Pattern: language difference
***************************************************************

*file: Vath_pth/12.languageDiff/languageDiff.C*

*Build inside 12.languageDiff:*
::

  make languageDiff

*Execute on the command line inside 12.languageDiff directory:*
::

  ./languageDiff

The following is a C++ code example to illustrate some language differences between C and C++.

C: printf is a single function and is performed atomically

C++: cout <<   << endl may have many different function calls so the outputs will be interleaved

A solution to the mixed output would be to implement a thread safe
cout class which uses critical sections and locks to give each thread
exclusive access to stdout. We will not look further into this.
Note: The Reduction utility class actually does this.
Try the exercises described in the code below.


.. literalinclude::
  ../patternlets/Vath_pth/12.languageDiff/languageDiff.C
  :language: c++
  :linenos:
