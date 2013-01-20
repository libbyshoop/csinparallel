********************************************
Shared Memory Parallel Patternlets in OpenMP
********************************************

When writing programs for shared-memory hardware with multiple cores, 
a programmer could use a
low-level thread package, such as pthreads. An alternative is to use
a compiler that processes OpenMP *pragmas*, which are compiler directives that
enable the compiler to generate threaded code.

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

A C code file for each example below can be found in subdirectories of the OpneMP directory,
along with a makefile.

0. The OMP parallel pragma
***************************

The `omp parallel` pragma on line 18, when uncommented, tells the compiler to
fork a set of threads to execute that particular line of code.

.. literalinclude::
	../patternlets/OpenMP/00.simpleParallel/simpleParallel.c
	:language: c
	:linenos:

1. Hello, World: default number of OpenMP threads
***********************************************************

Note how there are OpenMP functions to
obtain a thread number and the total number of threads.
When the pragma is uncommented, note what the default number of threads
is.  Here the threads are forked and execute the block of code insode the
curly braces on lines 18 through 20.

.. literalinclude:: 
	../patternlets/OpenMP/01.parallelHello/parallelHello.c
    :language: c
    :linenos:

2. Hello, World
***************
 
Here we enter the number of threads to use on the command line.

.. literalinclude::
	../patternlets/OpenMP/02.parallelHello2/parallelHello2.c
    :language: c

3. Master-Worker Implementation Strategy
*****************************************

.. literalinclude::
	../patternlets/OpenMP/03.masterWorker/masterWorker.c
    :language: c

4. Shared Data Decomposition Pattern:  blocking of threads in a parallel for loop
*********************************************************************************

.. literalinclude::
	../patternlets/OpenMP/04.parallelForBlocks/parallelForBlocks.c
    :language: c

5. Shared Data Decomposition Pattern:  striping of threads in a parallel for loop
*********************************************************************************

.. literalinclude::
	../patternlets/OpenMP/05.parallelForStripes/parallelForStripes.c
    :language: c

6. Collective Communication: Reduction
***************************************

Once processes have performed independent concurrent computations, possibly
on some portion of decomposed data, it is quite commen to then *reduce*
those individual computations into one value. In this example, an array of randomly assigned integers represents a set of shred data. All values
in the array are summed together by using the OpenMP
parallel for pragma with the `reduction(+:sum)` clause on the variable **sum**,
which is computed in line 61.

.. literalinclude::
	../patternlets/OpenMP/06.reduction/reduction.c
    :language: c
    :linenos:

7. Shared Data Pattern: Parallel-for-loop needs non-shared, private variables
******************************************************************************

.. literalinclude::
	../patternlets/OpenMP/07.private/private.c
    :language: c

8. Race Condition: missing the mutual exclusion patterm
********************************************************

.. literalinclude::
	../patternlets/OpenMP/08.atomic/atomic.c
    :language: c

9. Mutual Exclusion: two ways to ensure
****************************************

.. literalinclude::
	../patternlets/OpenMP/09.critical/critical.c
    :language: c

10.  Mutual Exclusion Pattern: compare performance
**************************************************

.. literalinclude::
	../patternlets/OpenMP/10.critical2/critical.c
    :language: c

11. Task Decomposition Pattern using OpenMP section directive
**************************************************************

.. literalinclude::
	../patternlets/OpenMP/11.sections/sections.c
    :language: c

