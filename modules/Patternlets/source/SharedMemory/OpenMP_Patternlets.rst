********************************************
Shared Memory Parallel Patternlets in OpenMP
********************************************

When writing programs for shared-memory hardware with multiple cores, 
a programmer could use a
low-level thread package, such as pthreads. An alternative is to use
a compiler that processes OpenMP *pragmas*, which are compiler directives that
enable the compiler to generate threaded code.  Whereas pthreads uses an **explicit**
multithreading mosel in which the programmer must explicitly create and manage threads,
OpenMP uses an **implicit** multithreading model in which the library handles
thread creation and management, thus making the programmer's task much simpler and
less error-prone.

The following are examples of C code with OpenMP pragmas
The firrst three are basic illustrations to get used to the OpenMP pragmas.
The rest illustrate how to implement particular patterns and what can
go wrong when mutual exclusion is not properly ensured.

Note: by default OpenMP uses the **Thread Pool** pattern of concurrent execution.
OpenMP programs initialze a group of threads to be used by a given program 
(often called a pool of threads).  These threads will execute concurrently
during portions of the code specified by the programmer.

Source Code
************

Please download all examples from this tarball: 
:download:`patternlets.tgz <../patternlets.tgz>`

A C code file for each example below can be found in subdirectories of the OpenMP directory,
along with a makefile.

0. The basic fork-join pattern 
******************************

The `omp parallel` pragma on line 22, when uncommented, tells the compiler to
fork a set of threads to execute that particular line of code.

.. literalinclude::
	../patternlets/OpenMP/00.forkJoin/forkJoin.c
	:language: c
	:linenos:

*file: patternlets/OpenMP/00.forkJoin/forkJoin.c*

1. Fork-join: setting the number of threads
***********************************************************

Note how there is an OpenMP function for setting the number of threads to use in 
the next 'fork'.

.. literalinclude::
	../patternlets/OpenMP/01.forkJoin2/forkJoin2.c
	:language: c
	:linenos:


*file patternlets/OpenMP/01.forkJoin2/forkJoin2.c*

2. Single Program, multiple data
***********************************************************

Note how there are OpenMP functions to
obtain a thread number and the total number of threads.
We have one program, but multiple threads executing,
each with a copy of the id and num_threads variables.

When the pragma is uncommented, note what the default number of threads
is.  Here the threads are forked and execute the block of code insode the
curly braces on lines 22 through 26.

.. literalinclude:: 
	../patternlets/OpenMP/02.spmd/spmd.c
    :language: c
    :linenos:

*file: patternlets/OpenMP/02.spmd/spmd.c*

3. Single Program, multiple data
*********************************
 
Here we enter the number of threads to use on the command line.

.. literalinclude::
	../patternlets/OpenMP/03.spmd2/spmd2.c
    :language: c

*file: patternlets/OpenMP/03.spmd2/spmd2.c*

4. Barrier
***********

Note what happens with and without the commented pragma on line 35.

.. literalinclude::
	../patternlets/OpenMP/04.barrier/barrier.c
    :language: c

*file: patternlets/OpenMP/04.barrier/barrier.c*

5. Master-Worker Implementation Strategy
*****************************************

.. literalinclude::
	../patternlets/OpenMP/05.masterWorker/masterWorker.c
    :language: c

*file: patternlets/OpenMP/05.masterWorker/masterWorker.c*

6. Shared Data Decomposition Pattern:  blocking of threads in a parallel for loop
*********************************************************************************

.. literalinclude::
	../patternlets/OpenMP/06.parallelForLoop-blocks/parallelForBlocks.c
    :language: c

*file: patternlets/OpenMP/06.parallelForLoop-blocks/parallelForBlocks.c*

7. Shared Data Decomposition Pattern:  striping of threads in a parallel for loop
*********************************************************************************

.. literalinclude::
	../patternlets/OpenMP/07.parallelForLoop-stripes/parallelForStripes.c
    :language: c

*file: atternlets/OpenMP/07.parallelForLoop-stripes/parallelForStripes.c*

8. Collective Communication: Reduction
***************************************

Once threads have performed independent concurrent computations, possibly
on some portion of decomposed data, it is quite commen to then *reduce*
those individual computations into one value. In this example, an array of randomly assigned integers represents a set of shared data. All values
in the array are summed together by using the OpenMP
parallel for pragma with the `reduction(+:sum)` clause on the variable **sum**,
which is computed in line 61.

.. literalinclude::
	../patternlets/OpenMP/08.parallelForLoop-reduction/reduction.c
    :language: c
    :linenos:

*file: patternlets/OpenMP/08.parallelForLoop-reduction/reduction.c*

9. Shared Data Pattern: Parallel-for-loop needs non-shared, private variables
******************************************************************************

.. literalinclude::
	../patternlets/OpenMP/09.parallelForLoop-private/private.c
    :language: c

*file: patternlets/OpenMP/09.parallelForLoop-private/private.c*

10. Race Condition: missing the mutual exclusion patterm
********************************************************

.. literalinclude::
	../patternlets/OpenMP/10.mutualExclusion-atomic/atomic.c
    :language: c

*file: patternlets/OpenMP/10.mutualExclusion-atomic/atomic.c*

11. Mutual Exclusion: two ways to ensure
****************************************

.. literalinclude::
	../patternlets/OpenMP/11.mutualExclusion-critical/critical.c
    :language: c

*file: patternlets/OpenMP/11.mutualExclusion-critical/critical.c*

12.  Mutual Exclusion Pattern: compare performance
**************************************************

.. literalinclude::
	../patternlets/OpenMP/12.mutualExclusion-critical2/critical2.c
    :language: c

*file: patternlets/OpenMP/12.mutualExclusion-critical2/critical2.c*

13. Task Decomposition Pattern using OpenMP section directive
**************************************************************

.. literalinclude::
	../patternlets/OpenMP/13.sections/sections.c
    :language: c

*file: patternlets/OpenMP/13.sections/sections.c*
