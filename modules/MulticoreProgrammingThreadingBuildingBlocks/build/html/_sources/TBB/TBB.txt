***************************************
Intel's Threading Building Blocks (TBB)
***************************************

Introduction
#############

OpenMP works well for adding parallelism to loops in working sequential code, and it's available for C, C++, and Fortran languages on many platforms (including Linux, Windows, and Macintosh OS X). Older versions of OpenMP did not readily support non-loop parallelism or programming with concurrent data structures, but OpenMP version 3.0 (released May 2008) provides a task feature for programming such computations. 

Intel's `Threading Building Blocks (TBB)`_ provides an object-oriented approach to implementing parallel algorithms, for the C++ language (and any of the three platforms). Adding parallelism to existing code in TBB is somewhat more involved than in OpenMP, but is considerably less complicated than programming in a native threads package for a particular operating system. The forthcoming new standard for the C++ language is likely to include parallelism similar to TBB.

.. _`Threading Building Blocks (TBB)`: http://threadingbuildingblocks.org/

For You To Do
##############

Create the code
----------------

Enter the following TBB program into a file ``trap-tbb.cpp``. Or you can download the file :download:`trap-tbb.cpp <trap-tbb.cpp>`  

.. literalinclude:: trap-tbb.cpp
    :language: cpp
    :lines: 1-51


Study this code carefully
--------------------------

Observe the following:

This program does *not* use a command-line argument (and the ``cstdlib`` library is not needed). Unlike OpenMP, TBB does not provide a simple way to request a particular number of threads. Instead, the TBB system chooses a number of threads to use automatically. (OpenMP will also make such a selection for you if you do not specify the number of threads to use.)

The following lines prepare for using TBB.

.. literalinclude:: trap-tbb.cpp
    :language: c++
    :lines: 5-6

Recall that in the OpenMP code, we parallelized the loop below by adding a pragma just before that for loop.

::

         for(i = 1; i < n; i++) {
            integral += f(a+i*h);
          }

In order to program a comparable computation in TBB, we create a class ``SumHeights`` whose method ``operator()`` contains the following loop: 

.. literalinclude:: trap-tbb.cpp
       :language: c++
       :lines: 21-23

Observe that the forms of the two loops indicate the same iterative computation, if one matches 1 to ``r.begin()``, ``n`` to ``r.end()`` and ``integral``, ``a``, and h to ``SumHeights`` state variables ``my_int``, ``my_a``, and ``my_h``.  One way to describe this relationship is to say that the class ``SumHeights`` is a "wrapper" 	around its loop. 

The class ``SumHeights`` defines ``operator()``, which means that an object of type ``SumHeights`` can be called using function-call syntax. Since ``operator()`` is defined here with one argument, this means we can cause the for loop to execute using a call ``sh(range)``, where ``sh`` is an object of type ``SumHeights`` and ``range`` is an appropriate argument.

#. Note that ``operator()`` is a ``const`` method (indicated by the const after ``)`` and before ``{`` ), which means that it is permitted to call ``sh(range)`` with a ``const`` object sh.

#. The argument ``r`` of ``operator()`` indicates the *range* of the loop, i.e., the starting and ending values for the loop control variable.

#. The loop control variable ``i`` has type ``int`` in the OpenMP implementation, but type ``size_t`` in the TBB implementation. ``size_t`` is an integer type, which may be equivalent to ``int``, ``long``, or another integer type depending on implementation.

#. The range ``r`` has the type ``blocked_range<size_t>``. This is a *templated type* built over the ``size_t`` type. There could be ``blocked_range`` types built over other types, as well, e.g., ``int`` or ``long``.

The constructor ``SumHeights()`` makes local copies ``my_a``, etc., of variables ``a``, etc., in ``main()``, enabling values in ``main()`` to be used within the class ``SumHeights``. 

       
The call to ``parallel_for`` in ``main()`` automatically subdivides (or chunks) the range ``r`` for multi-threaded parallel computation. ``parallel_for`` expects a range in its first argument, and an object with a method ``operator()`` having one range argument in its second argument. 

The variable ``integral`` is passed by reference in the constructor ``SumHeights()`` in an effort to use that memory location ``integral`` as an accumulator during the parallelized computation.

The constructor initializes the state variables ``my_a``, ``my_h``, and ``my_int`` using *colon initializers*. In the constructor definition

.. literalinclude:: trap-tbb.cpp
        :language: c++
        :lines: 26-28
 
The expression ``my_a(a)`` located after the colon : and before the curly bracket ``{`` has the same effect as an assignment   

::

  my_a = a;

would if it occurred *between* the curly brackets. Colon initialization is optional for the state variables ``my_a`` and ``my_h``, but it is required for the state variable ``my_int``, because that state variable was defined using a reference type. 

.. note:: Can you detect any problems in this code?


Execute this code
------------------

For this lab, we will run this TBB program on the MTL.

First, if necessary, copy the program file you created to a local machine for connecting to MTL, e.g., your laptop. You will need to use a ‘terminal’ on Macs or ‘Putty’ on PCs.  If you are off campus, you will need to ssh into a machine on your campus before then logging into the MTL machine at Intel’s headquarters in Oregon.

You can login to the MTL computer, as follows

::

  laptop% ssh accountname@192.55.51.81

Use one of the student account usernames provided to you, together with the password distributed to the class. 

Next, copy your program from your laptop or local linux machine to the MTL machine. One way to do this is to use another window (to keep for copying your code), then enter the following command from the directory where your code is located:

:: 

  scp trap-tbb.cpp accountname@192.55.51.81:
    

On the remote MTL system, execute the following command, which sets up environment variables for compiling with TBB: 

::

  source/opt/intel/Compiler/11.1/056/tbb/bin/tbbvars.sh intel64

The ``intel64`` command-line argument prepares for 64-bit compilation.

After making this copy, login into the MTL machine 192.55.51.81 in another window.

To compile your program that was copied in a prior step, issue this command: 

::

  192.55.51.81% g++ -o trap-tbb trap-tbb.cpp -ltbb_debug

.. note:: You can use ``-ltbb`` instead of ``-ltbb_debug`` for a production version of the library instead of one with debugging hooks.

Now run your program with the following command: 

::

  192.55.51.81% ./trap-tbb

The result is significantly less than 2! Can you think of an explanation for the answer being so far off?

Also run several time tests of your program

::

  192.55.51.81% time ./trap-tbb

What do you observe in these time tests? How do the times compare to timed runs of ``trap-omp`` for various thread sizes? 



