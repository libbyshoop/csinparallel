*********************************************************
Shared Memory Program Structure and Coordination Patterns
*********************************************************

0. Program Structure Implementation Strategy: The basic fork-join pattern
*************************************************************************


*file: Vath_pth/00.forkJoin/forkJoin.C*

*Build inside 00.forkJoin directory:*
::

  make forkJoin

*Execute on the command line inside 00.forkJoin directory:*
::

	./forkJoin

The *SPool TH()* constructor on line 20, tells the compiler to create a team of two worker threads.
The *Dispatch()* function activates the set of threads to execute the thread function passed as an argument.
The *WaitForIdle()* function joins the threads after all worker threads have completed their task. Notice that
unlike OpenMP, the join is explicit. You can conceptualize how this works using the following diagram,
where time is moving from left to right:

.. image:: ForkJoin.png
	:width: 800

Observe what happens on the machine where you are running this code.


.. literalinclude::
	../patternlets/Vath_pth/00.forkJoin/forkJoin.C
	:language: c++
	:linenos:

1. Program Structure Implementation Strategy: Fork-join with setting the number of threads
******************************************************************************************

*file Vath_pth/01.forkJoin2/forkJoin2.C*

*Build inside 01.forkJoin2 directory:*
::

  make forkJoin2

*Execute on the command line inside 01.forkJoin2 directory:*
::

  ./forkJoin2

This code illustrates that one program can fork and join more than once.
Programmers can set the number of threads to use when creating the team of worker threads.

Note on line 22 there is a vath library SPool utility function called *Spool TH()* that takes
the number of threads as an argument. Follow the instructions in the header of the
code file to understand how constructing SPool objects, and forking and joining threads repeatedly works.

.. literalinclude::
  ../patternlets/Vath_pth/01.forkJoin2/forkJoin2.C
  :language: c++
  :linenos:



2. Program Structure Implementation Strategy: Single Program, multiple data
***************************************************************************

*file: Vath_pth/02.spmd/spmd.C*

*Build inside 02.spmd directory:*
::

	make spmd

*Execute on the command line inside 02.spmd directory:*
::

	./spmd

Note how there is a SPool utility function *GetRank()* to
obtain a thread number. We have one program, but multiple threads executing
the thread function, each with a copy of the rank variable.
Programmers write one program, but write it in such a way that
each thread has its own data values for particular variables.
This is why this is called the *single program, multiple data* (SPMD) pattern.

Most parallel programs use this SPMD pattern as writing one program
is ultimately the most efficient method for programmers. It does require you
as a programmer to understand how this works, however. Each thread executing in
parallel has its own set of variables. Conceptually, it looks like this,
where each thread has its own memory for the variable rank:

.. image:: SPMD.png
	:width: 800

When you execute the code, what do you observe about the order of the printed lines?
Run the program multiple times--does the ordering change? This illustrates an
important point about threaded programs: *the ordering of execution of statements
between threads is not guaranteed.* This is also illustrated in the diagram above.

.. literalinclude::
  ../patternlets/Vath_pth/02.spmd/spmd.C
  :language: c++
  :linenos:


3. Program Structure Implementation Strategy: Single Program, multiple data with user-defined number of threads
***************************************************************************************************************

*file: Vath_pth/03.spmd2/spmd2.C*

*Build inside 03.spmd2 directory:*
::

  make spmd2

*Execute on the command line inside 03.spmd2 directory:*
::

  ./spmd2 4
  Replace 4 with other values for the number of threads

Here we enter the number of threads to use on the command line. This is a useful way to
make your code versatile so that you can use as many threads as you would like. In this case,
a global pointer to a SPool object is declared and it is later initialized by main().

.. literalinclude::
  ../patternlets/Vath_pth/03.spmd2/spmd2.C
  :language: c++
  :linenos:


4. Coordination: Synchronization with a Barrier
***********************************************

*file: Vath_pth/04.barrier/barrier.C*

*Build inside 04.barrier directory:*
::

  make barrier

*Execute on the command line inside 04.barrier directory:*
::

  ./barrier 4
  Replace 4 with other values for the number of threads

The barrier pattern is used in parallel programs to ensure that all threads complete
a parallel section of code before execution continues. This can be necessary when
threads are generating computed data (in an array, for example) that needs to be
completed for use in another computation.

Conceptually, the running code is executing like this:

.. image:: Barrier.png
	:width: 800

Note what happens with and without the commented barrier function on line 42.

.. literalinclude::
  ../patternlets/Vath_pth/04.barrier/barrier.C
  :language: c++
  :linenos:
