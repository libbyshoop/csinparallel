===========================================================================
Data Decomposition Algorithm Strategies and Related Coordination Strategies
===========================================================================


5. Shared Data Decomposition Algorithm Strategy:  chunks of data per thread using a parallel for loop implementation strategy
*******************************************************************************************************************************

*file: Vath_pth/05.parallelLoop-equalChunks/parallelLoopEqualChunks.C*

*Build inside 05.parallelLoop-equalChunks directory:*
::

	make parallelLoopEqualChunks

*Execute on the command line inside 05.parallelLoop-equalChunks directory:*
::

	./parallelLoopEqualChunks 4
	Replace 4 with other values for the number of threads, or leave off


An iterative for loop is a remarkably common pattern in all programming, primarily used to
perform a calculation N times, often over a set of data containing N elements, using each
element in turn inside the for loop.  If there are no dependencies between the calculations
(i.e. the order of them is not important), then the code inside the loop can be split
between forked threads.  When doing this, a decision the programmer needs to make is to
decide how to partition the work between the threads by answering this question:

* How many and which iterations of the loop will each thread complete on its own?

We refer to this as the **data decomposition** pattern because we are decomposing the
amount of work to be done (typically on a set of data) across multiple threads.
In the following code, this is done in Pthreads by using the vath library function
*ThreadRange()* inside of the thread function (line 39) in the following code.

.. literalinclude::
	../patternlets/Vath_pth/05.parallelLoop-equalChunks/parallelLoopEqualChunks.C
  :language: c++
  :linenos:

Once you run this code, verify that the default behavior for this function is this
sort of decomposition of iterations of the loop to threads, when you set the
number of threads to 4 on the command line:

.. image:: ParalleFor_Chunks-4_threads-1.png


What happens when the number of iterations (16 in this code) is not evenly divisible by the number of threads?
Try several cases to be certain how the compiler splits up the work.
This type of decomposition is commonly used when accessing data that is stored in
consecutive memory locations (such as an array) that might be cached by each thread.


6. Shared Data Decomposition Algorithm Strategy:  one iteration per thread in a parallel for loop implementation strategy
***************************************************************************************************************************

*file: Vath_pth/06.parallelLoop-ChunksOf1/parallelLoopChunksOf1.C*

*Build inside 06.parallelLoop-ChunksOf1 directory:*
::

	make parallelLoopChunksOf1

*Execute on the command line inside 06.parallelLoop-ChunksOf1 directory:*
::

	./parallelLoopChunksOf1 4
	Replace 4 with other values for the number of threads

You can imagine other ways of assigning threads to iterations of a loop besides that
shown above for four threads and 16 iterations.  A simple decomposition sometimes used
when your loop is not accessing consecutive memory locations would be to let each
thread do one iteration, up to N threads, then start again with thread 1 taking the next iteration.
Section A in the code is an explicit way of doing it in Pthreads.

.. literalinclude::
	../patternlets/Vath_pth/06.parallelLoop-ChunksOf1/parallelLoopChunksOf1.C
  :language: c++
  :linenos:

7.  Shared Data Decomposition Algorithm Strategy: Revisited
*************************************************************

*file: Vath_pth/07.parallelLoop-Revisited/parallelLoopRevisited.C*

*Build inside 07.parallelLoop-Revisited directory:*
::

	make parallelLoopRevisited

*Execute on the command line inside 07.parallelLoop-Revisited directory:*
::

	./parallelLoopRevisited 4
	Replace 4 with other values for the number of threads

The following example computes factorials for the numbers 2 through 1024, placing the result in an array.
This array of results is the data in this data decomposition pattern. Since each number will
take a different amount of time to compute, this is a case where not using consecutive iterations of the
work improves the performance. Try the tasks listed in the header of the code shown below to see this.

.. literalinclude::
	../patternlets/Vath_pth/07.parallelLoop-Revisited/parallelLoopRevisited.C
  :language: c++
  :linenos:


8. Coordination Using Collective Communication: Reduction
*********************************************************

*file: Vath_pth/08.reduction/reduction.C*

*Build inside 08.reduction directory:*
::

	make reduction

*Execute on the command line inside 08.reduction directory:*
::

	./reduction 4
	Replace 4 with other values for the number of threads, or leave off

Once threads have performed independent concurrent computations, possibly
on some portion of decomposed data, it is quite common to then *reduce*
those individual computations into one value. This type of operation is
called a **collective communication** pattern because the threads must somehow
work together to create the final desired single value.

In this example, an array of randomly assigned doubles represents a set of shared data (a more
realistic program would perform a computation that creates meaningful data values; this is just an example).
Note the common sequential code pattern found in the function called *sequentialSum* in the code
below (starting line 58): a for loop is used to sum up all the values in the array.

Next let's consider how this can be done in parallel with threads.
Somehow the threads must `communicate` to keep the overall sum updated
as each of them works on a portion of the array. The *Reduction <T>* utility
from the vath library is used. This class contains a mutex that guards operations
that are performed on the shared variable. A Reduction object is created on
line 32 that can be used to accumulate the values in the array. In the thread worker
function, line 52 shows how to sum partial results to a shared variable.
Each thread has the shared variable accumulate their partial result. Therefore,
at completion all values in the array are summed together.

.. literalinclude::
	../patternlets/Vath_pth/08.reduction/reduction.C
    :language: c++
    :linenos:

Something to think about
========================

Do you have an ideas about why the parallel version without reduction did not
produce the correct result?  Later examples will hopefully shed some light on this.

9. Coordination Using Collective Communication: Reduction revisited
********************************************************************
*file: Vath_pth/09.reduction2/reduction2.C*

*Build inside 09.reduction2 directory:*
::

	make reduction2

*Execute on the command line inside 09.reduction2 directory:*
::

	./reduction2 4 8192
	Replace 4 with other values for the number of threads
	Replace 8192 with other values for n (computing up to n factorial)

The next example uses many threads to generate computations of factorials of n.
Though there are likely other better ways to compute factorials, this
example uses a very simple approach to illustrate how reduction can be used with the
multiplication operation instead of addition in the previous example. This is shown
on line 56 in the code below, which also makes use of an additional C++ file, BigInt.h:

.. literalinclude::
	../patternlets/Vath_pth/09.reduction2/reduction2.C
  :language: c++
  :linenos:

With this code you can begin to explore the time it takes to execute the program when using increasing numbers of threads for various values of n. Follow the instructions at the top of the file.
