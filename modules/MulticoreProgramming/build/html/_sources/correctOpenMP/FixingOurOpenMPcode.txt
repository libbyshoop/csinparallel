Creating a correct threaded version
====================================

Race Conditions
---------------

A program has a race condition if the correct behavior of that program
depends on the timing of its execution. With 2 or more threads, the
program trap-omp.C has a race condition concerning the shared variable
``integral``, which is the accumulator for the summation performed by that
program's for loop.

When threadct == 1, the single thread of execution updates the shared variable 
``integral``\  on every iteration, by reading the prior value of the memory location for  
``integral``, computing and adding the value f(a+i\*h), then storing the result into that memory location. (Recall that a variable is a named location in main memory.)

But when threadct > 1, there are at least two independent threads, executed on separate physical cores, that are reading then writing the memory location for 
``integral``. The incorrect answer results when the reads and writes of that memory location get out of order. Here is one example of how unfortunate ordering can happen with two threads:

========================= =========================== ===========================
                           Thread 1                    Thread 2
========================= =========================== ===========================
source code:              | integral += f(a+i\*h);    | integral += f(a+i\*h); 
execution of binary code: |                           |
                          | 1. read value of integral |
                          | 2. add  f(a+i\*h)         | 1. read value of integral 
                                                      | 2. add  f(a+i\*h) 
                          | 3. write sum to  integral |
                                                      | 3. write sum to  integral
========================= =========================== ===========================


In this example, during one poorly timed iteration for each thread,
Thread 2 reads the value of the memory location integral before Thread 1
can write its sum back to integral. The consequence is that Thread 2
replaces (overwrites) Thread 1's value of integral, so the amount added
by Thread 1 is omitted from the final value of the accumulator integral.

.. topic:: To Do 

   Can you think of other situations where unfortunate ordering of thread
   operations leads to an incorrect value of integral? Write down at least
   one other bad timing scenario.  Other scenarios will work (you may have seen some of these during testing of your code).  Write down one of these.

.. note:: Thousands of occurrences of bad timing lead to the computed answer for integral being off by often 25% or more.

Avoiding Race Conditions
------------------------

One approach to avoiding this program's race condition is to use a separate local variable integral for each thread instead of a global variable that is shared by all the threads. But declaring integral to be private instead of shared in the pragma will only generate threadct partial sums in those local variables named integral -- the partial sums in those temporary local variables will not be added to the program's variable integral. In fact, the value in those temporary local variables will be discarded when each thread finishes its work for the parallel for if we simply make integral private instead of shared.

.. topic:: To Do

   Can you re-explain this situation in your own words?

Fortunately, OpenMP provides a convenient and effective solution to this problem. The OpenMP clause
::

   reduction(+: integral)   

will

#. cause the variable integral to be private (local) during the
   execution of each thread, and
#. add the results of all those private variables, then finally
#. store that sum of private variables in the global variable named
   integral.


Reduction
----------

This code example contains a very common *pattern* found in
parallel programs: **reducing several values down to one**.  This is so common that the designers of OpenMP chose to make it simple to declare that a variable was involved on a reduction.  The code here represents one way that reduction takes place: during a step-wise calculation of smaller pieces of a larger problem.  The other common type of reduction is to accumulate all of the values stored in an array.

.. seealso:: You can use other arithmetic operators besdides plus in the reduction clause-- see `the OpenMP tutorial section on reduction <https://computing.llnl.gov/tutorials/openMP/#REDUCTION>`_

According to the OpenMP Tutorial, here is how the reduction is being done inside the compiled OpenMP code (similar to how we described it above, but generalized to any reduction operator):

         "A private copy for each listed variable is created for each thread.  At the end of the reduction, the reduction operation is applied to all private copies of the shared variable, and the final result is written to the global shared variable."

Thus, a variable such as ``integral`` that is decalred in a reduction clause is both private to each thread and ultimately a shared variable.


.. topic:: To Do

   Add the above reduction clause to your OpenMP pragma and remove integral from the list of shared variables, so that the pragma in your code appears as shown below.  Then recompile and test your program many times with different numbers of threads. You should now see the correct answer of 2.0 every time you execute it with multiple threads -- a correct multi-core program!

.. literalinclude:: trap-omp-working.C
   :language: c++
   :lines: 33-37

.. topic:: Tricky Business

   Note that with the incorrect version, you sometimes got lucky (perhaps even many times) as you tried running it over and over. This is one of the most perplexing and difficult aspects of parallel programming: **it can be difficult to find your bugs because many times your program will apear to be correct**.  Remember this: you need to be diligent about thinking through your solutions and questioning whether you have coded it correctly.

Thread-safe Code
----------------

A code segment is said to be *thread-safe* if it remains correct when
executed by multiple independent threads. The body of this loop is
not thread-safe; we were able to make it so by indicating that a *reduction*
was taking place on the variable ``integral``.  


Some C/C++ libraries are identified as thread-safe, meaning that each function
in that library is thread-safe. Of course, calling a thread-safe
function doesn't insure that the code with that function call is
thread-safe. For example, the function f() in our example, is
thread-safe, but the body of that loop is not thread-safe.


