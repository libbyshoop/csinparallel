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

.. literalinclude::
	../patternlets/OpenMP/00.simpleParallel/simpleParallel.c
	:language: c

1. Hello, World: default number of OpenMP threads
***********************************************************

Note how there are OpenMP functions to
obtain a thread number and the total number of threads.
When the pragma is uncommented, note what the default number of threads
is.

.. literalinclude:: 
	../patternlets/OpenMP/01.parallelHello/parallelHello.c
    :language: c

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

4. Shared Data Pattern:  blocking of threads in a parallel for loop
*******************************************************************

.. literalinclude::
	../patternlets/OpenMP/04.parallelForBlocks/parallelForBlocks.c
    :language: c

5. Shared Data Pattern:  striping of threads in a parallel for loop
**********************************************************************

.. literalinclude::
	../patternlets/OpenMP/05.parallelForStripes/parallelForStripes.c
    :language: c

6. Shared Data Pattern: Parallel-for-loop needs non-shared, private variables
******************************************************************************

.. literalinclude::
	../patternlets/OpenMP/06.private/private.c
    :language: c

7. Race Condition: missing the mutual exclusion patterm
********************************************************

.. literalinclude::
	../patternlets/OpenMP/07.atomic/atomic.c
    :language: c

8. Mutual Exclusion: two ways to ensure
****************************************

.. literalinclude::
	../patternlets/OpenMP/08.critical/critical.c
    :language: c

9.  Mutual Exclusion Pattern: compare performance
**************************************************

.. literalinclude::
	../patternlets/OpenMP/09.critical2/critical.c
    :language: c

10. Task Decomposition Pattern using OpenMP section directive
**************************************************************

.. literalinclude::
	../patternlets/OpenMP/10.sections/sections.c
    :language: c

