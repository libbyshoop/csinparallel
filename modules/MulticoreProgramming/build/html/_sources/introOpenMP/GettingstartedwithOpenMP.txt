Getting Started with Multicore Programming using OpenMP
=======================================================

Notes about this document
--------------------------
This is designed to be a lab activity  that you will perform on linux machines and/or on the Intel Manycore Testing Lab (MTL).

:Comments:

   Sections labeled like this are important explanations to pay attention to.

.. topic:: Dig Deeper:

   Comments in this format indicate possible avenues of exploration for people seeking more challenge or depth of knowledge.


Multicore machines and shared memory
-------------------------------------

Multicore CPUs have more than one ‘core’ processor that can execute
instructions at the same time.   The cores share main memory.  In the
next few activities, we will learn how to use a library called OpenMP to
enable us to write programs that can use multicore processors and shared
memory to write programs that can complete a task faster by taking
advantage of using many cores.  These programs are said to work “in
parallel”.  We will start with our own single machines, and then
eventually use a machine with a large number of cores provided by Intel
Corporation, called the Manycore Testing Lab (MTL).

Parallel programs use multiple ‘threads’ executing instructions
simultaneously to accomplish a task in a shorter amount of time than a
single-threaded version.  A process is an execution of a program. A
thread is an independent execution of (some of) a process's code that
shares (some) memory and/or other resources with that process. When
designing a parallel program, you need to determine what portions could
benefit from having many threads executing instructions at once.  In
this lab, we will see how we can use “task parallelism” to execute the
same task on different parts of a desired computation in threads and
gather the results when each task is finished.

Getting started with OpenMP
----------------------------

We will use a standard system for parallel programming
called\ `  <http://openmp.org/wp/>`_\ `OpenMP <http://openmp.org/wp/>`_,
which enables a C or C++ programmer to take advantage of multi-core
parallelism primarily through preprocessor pragmas. These are directives
that enable the compiler to add and change code (in this case to add
code for executing sections of it in parallel).

.. seealso:: More resources about OpenMP can be found here: `http://openmp.org/wp/resources/ <http://openmp.org/wp/resources/>`_.

        



We will begin with a short C++ program, parallelize it using OpenMP, and
improve the parallelized version. This initial development work can be
carried out on a linux machine.  Working this time with C++ will not be
too difficult, as we will not be using the object-oriented features of
the language, but will be taking advantage of easier printing of output.

The following program computes a Calculus value, the "trapezoidal
approximation of  

.. math::

   \int_0^x \sin(x) {d}{x}


using :math:`2^{20}` equal subdivisions.”   The exact answer from this computation
should be 2.0.

.. literalinclude:: trap-omp-unfinished.C
   :language: c++
   :linenos:



:Comments:

   * If a command line argument is given, the code segment below converts that argument to an integer and assigns that value to the variable threadct, overriding the default value of 1. This uses the two arguments of the function main(), namely argc and argv. This demo program makes no attempt to check whether a first command line argument argv[1] is actually an integer, so make sure it is (or omit it).

   .. literalinclude:: trap-omp-unfinished.C
      :language: c++
      :lines: 19-20

   * The variable *threadct* will be used later to control the number of threads to be used. Recall that a process is an execution of a program. A **thread** is an independent execution of (some of) a process's code that shares (some) memory and/or other resources with that process. We will modify this program to use multiple threads, which can be executed in parallel on a multi-core computer.

   * The preprocessor macro \_OPENMP is defined for C++ compilations that include support for OpenMP. Thus, the code segment below provides a way to check whether OpenMP is in use.

   .. literalinclude:: trap-omp-unfinished.C
      :language: c++
      :lines: 24-28


   * The above code also shows the convenient way to print to stdout in C++. 


   * The following lines contain the actual computation of the trapezoidal approximation:

   .. literalinclude:: trap-omp-unfinished.C
      :language: c++
      :lines: 30-37
           

   Since n == :math:`2^{20}`, the for loop adds over 1 million values. Later in this
   lab, we will parallelize that loop, using multiple cores which
   will each perform part of this summation, and look for speedup in the
   program's performance.

To Do:
------

On a linux machine, create a file named trap-omp.C containing the program above or grab it and save it from the following link: 

:download:`download trap-omp.C <trap-omp.C>`  (on most browsers, right-click, save link as)

To compile your file, you can enter the command:
::

   %  g++ -o trap-omp trap-omp.C -lm -fopenmp


.. note:: Here, % represents your shell prompt, which is usually a machine name followed by either % or $.


First, try running the program without a command-line argument, like this:
::

   %  ./trap-omp

This should print a line "\_OPENMP defined, threadct = 1", followed by a
line indicating the computation with an answer of 2.0. Next, try
::

   %  ./trap-omp 2

This should indicate a different thread count, but otherwise produce the
same output. Finally, try recompiling your program omitting
the -fopenmp flag. This should report \_OPENMP not defined, but give the
same answer 2.0.

Note that the program above is actually using only a single core, whether
or not a command-line argument is given. It is an ordinary C++ program
in every respect, and OpenMP does not magically change ordinary C++
programs; in particular, the variable *threadct* is just an ordinary local
variable with no special computational meaning.

To request a parallel computation, add the following pragma preprocessor
directive, just before the for loop.

.. literalinclude:: trap-omp-notworking.C
   :language: c++
   :lines: 33-34


The resulting code will have this format:

.. literalinclude:: trap-omp-notworking.C
   :language: c++
   :lines: 31-37


:Comments: 

   * Make sure no characters follow the backslash character before the end of the first line. This causes the two lines to be treated as a single pragma (useful to avoid long lines). 

   * The phrase **omp parallel for** indicates that this is an OpenMP pragma for parallelizing the for loop that follows immediately. The OpenMP system will divide the :math:`2^{20}` iterations of that loop up into *threadct* segments, each of which can be executed in parallel on multiple cores.

   * The OpenMP clause **num\_threads(threadct)** specifies the number of threads to use in the parallelization.
   
   * The clauses in the second line indicate whether the variables that appear in the for loop should be shared with the other threads, or should be local private variables used only by a single thread. Here, four of those variables are globally shared by all the threads, and only the loop control variable i is local to each particular thread.


Enter the above code change (add the pragma preprocessor directive), then compile and test the resulting executable with one thread, then  more than one thread, like this:
::

   %  g++ -o trap-omp trap-omp.C -lm -fopenmp
   %  ./trap-omp
   %  ./trap-omp 2
   %  ./trap-omp 3
   etc.

.. topic:: Dig Deeper:

   * The `OpenMP tutorial <https://computing.llnl.gov/tutorials/openMP/>`_ contains more information about advanced uses of OpenMP. Note that OpenMP is a combination of libraries and compiler directives that have been defined for both Fortran and C/C++.

   * OpenMP provides other ways to set the number of threads to use, namely the omp\_set\_num\_threads() library function (see `tutorial section on library routines <https://computing.llnl.gov/tutorials/openMP/#RunTimeLibrary>`_), and the OMP\_NUM\_THREADS linux/unix environment variable (see `tutorial section on environment variables <https://computing.llnl.gov/tutorials/openMP/#EnvironmentVariables>`_).


   * OpenMP provides several other clauses for managing variable locality, initialization, etc. Examples: default, firstprivate, lastprivate, copyprivate.  You could investigate this further in the `tutorial section pertaining to clauses <https://computing.llnl.gov/tutorials/openMP/#Clauses>`_.

What's happening?
------------------

After inserting the parallel for pragma, observe that for threadct == 1 (e.g., no
command-line argument), the program runs
and produces the correct result of 2.0, but that
::

   % ./trap-omp 2

which sets threadct == 2, sometimes produces an incorrect answer (perhaps about
1.5). What happens with repeated runs with that and other (positive)
thread counts? Can you explain why?

**Note:** Try reasoning out why the computed answer is correct for one
thread but incorrect for two or more threads. Hint: All of the values
being added in this particular loop are positive values, and the
erroneous answer is too low.

If you figure out the cause, think about how to fix the problem. You may
use the\ `  <http://openmp.org/wp/>`_\ `OpenMP
website <http://openmp.org/wp/>`_ or other resources to research your
solution, if desired.


   


