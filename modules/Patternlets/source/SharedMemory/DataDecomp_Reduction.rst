===========================================================================
Data Decomposition Algorithm Strategies and Related Coordination Strategies
===========================================================================


6. Shared Data Decomposition Algorithm Strategy:  chunks of data per thread using a parallel for loop implementation strategy
*******************************************************************************************************************************

*file: openMP/06.parallelLoop-equalChunks/parallelLoopEqualChunks.c*

*Build inside 06.parallelLoop-equalChunks directory:*
::

	make parallelLoopEqualChunks

*Execute on the command line inside 06.parallelLoop-equalChunks directory:*
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
In the following code, this is done in OpenMP using the *omp parallel for* pragma
just in front of the for statement (line 27) in the following code.

.. literalinclude::
	../patternlets/openMP/06.parallelLoop-equalChunks/parallelLoopEqualChunks.c
    :language: c
    :linenos:

Once you run this code, verify that the default behavior for this pragma is this
sort of decomposition of iterations of the loop to threads, when you set the
number of threads to 4 on the command line:

.. image:: ParalleFor_Chunks-4_threads-1.png


What happens when the number of iterations (16 in this code) is not evenly divisible by the number of threads?  Try several cases to be certain how the compiler splits up the work.
This type of decomposition is commonly used when accessing data that is stored in
consecutive memory locations (such as an array) that might be cached by each thread.

7. Shared Data Decomposition Algorithm Strategy:  one iteration per thread in a parallel for loop implementation strategy
*******************************************************************************************************************************

*file: openMP/07.parallelLoop-chunksOf1/parallelLoopChunksOf1.c*

*Build inside 07.parallelLoop-chunksOf1 directory:*
::

	make parallelLoopChunksOf1

*Execute on the command line inside 07.parallelLoop-chunksOf1 directory:*
::

	./parallelLoopChunksOf1 4
	Replace 4 with other values for the number of threads, or leave off

You can imagine other ways of assigning threads to iterations of a loop besides that
shown above for four threads and 16 iterations.  A simple decomposition sometimes used
when your loop is not accessing consecutive memory locations would be to let each
thread do one iteration, up to N threads, then
start again with thread 0 taking the next iteration.  This is declared in OpenMP
using the pragma on line 31 of the following code.  Also note that the commented code
below it is an alternative explicit way of doing it.  The schedule clause is the preferred style
when using OpenMP and is more versatile, because you can easily change the `chunk size`
that each thread will work on.

.. literalinclude::
	../patternlets/openMP/07.parallelLoop-chunksOf1/parallelLoopChunksOf1.c
    :language: c
    :linenos:

This can be made even more
efficient if the next available thread simply takes the next iteration.
In OpenMP, this is done by using *dynamic* scheduling instead of the static scheduling shown
in the above code.  Also note that the number of iterations, or chunk size, could
be greater than 1 inside the schedule clause.


8. Coordination Using Collective Communication: Reduction
*********************************************************

*file: openMP/08.reduction/reduction.c*

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

In this example, an array of randomly assigned integers represents a set of shared data (a more realistic program would perform a computation
that creates meaningful data values; this is just an example).
Note the common sequential code pattern found in the function called *sequentialSum* in the code
below (starting line 51): a for loop is used to sum up all the values in the array.

Next let's consider how this can be done in parallel with threads.
Somehow the threads must implicitly `communicate` to keep the overall sum updated
as each of them works on a portion of the array.
In the *parallelSum* function, line 64 shows a special clause that
can be used with the parallel for pragma in OpenMP for this. All values
in the array are summed together by using the OpenMP
parallel for pragma with the `reduction(+:sum)` clause on the variable **sum**,
which is computed in line 66.

.. literalinclude::
	../patternlets/openMP/08.reduction/reduction.c
    :language: c
    :linenos:

Something to think about
========================

Do you have an ideas about why the parallel for pragma without the reduction clause did not
produce the correct result?  Later examples will hopefully shed some light on this.

9. Coordination Using Collective Communication: Reduction revisited
********************************************************************

*Build inside 09.reduction-userDefined directory:*
::

	make reduction2

*Execute on the command line inside 09.reduction-userDefined directory:*
::

	./reduction  4 4096
	Replace 4 with other values for the number of threads
	Replace 4096 with other values for n (computing up to n factorial)

The next example uses many threads to generate computations of factorials of n. Though there are likely other better ways to compute factorials, this
example uses a very simple approach to illustrate how reduction can be used with the
multiplication operation instead of addition in the previous example. The pragma for
this is on line 34 in the code below, which also makes use of an additional C++ file, BigInt.h:

.. literalinclude::
	../patternlets/openMP/09.reduction-userDefined/reduction2.cpp
    :language: c++
    :linenos:

With this code you can begin to explore the time it takes to execute the program when using increasing numbers of threads for various values of n. Follow the instructions at the top of the file.

10. Dynamic Data Decomposition
********************************************************************

*Build inside 10.parallelLoop-dynamicSchedule directory:*
::

	make dynamicScheduling

*Execute on the command line inside 10.parallelLoop-dynamicSchedule directory:*
::

	./dynamicScheduling 4
	Replace 4 with other values for the number of threads

The following example computes factorials for the numbers 2 through 512, placing the result in an array. This array of results is the data in this data decomposition pattern. Since each number will take a different amount of time to compute, this is
a case where using dynamic scheduling of the work improves the performance. Try the tasks lsited in the header of the code shown below to see this.

.. literalinclude::
	../patternlets/openMP/10.parallelLoop-dynamicSchedule/dynamicScheduling.cpp
    :language: c++
    :linenos:
