************************
Counting Primes Exercise
************************

In this module, you will gain some exposure to the Java Thread class and also see some of the issues in parallel programming. Begin with the code :download:`SerialPrimes.java <SerialPrimes.java>`. This program counts the number of prime numbers (those only divisible by 1 and themselves) between 1 and 2,000,000. Look through this program, compile it, and then time a run of it with the command line

.. code-block:: bash

	time java SerialPrimes

After a short wait (depending on your computer’s processing power) you should be told that there are 148,933 primes in the range 1–2,000,000.

In addition to the output from the program itself, the time program will print out some information on your program’s running time. The first two numbers are the amount of CPU time the program took while in user mode (i.e. not during system calls) and the amount of system time (i.e. during system calls). The third number is the total amount of time between when the program started and when it ended; this is called the wall clock time.

Now that you are familiar with the serial (i.e. non-parallel) implementation, it is time to take a look at a naive multi-threaded implementation. Download a copy of :download:`ThreadedPrimes.java <ThreadedPrimes.java>` from the module files and look through its code. Instead of a prime-finding loop in main, the work is done by ``PrimeFinder`` objects, each of which is responsible for finding primes within a range specified during object construction. The main method creates two of these objects with complementary ranges and assigns one of them to each of two threads. To allow this, ``PrimeFinder`` implements the ``Runnable`` interface, which simply requires the ``run`` method. Once the work is assigned to each thread, the threads are started by calling their ``start`` methods.

Compile and run this program. The results are unimpressive; the program dramatically undercounts the number of primes. The problem is that main is printing the number of primes before the ``PrimeFinder`` objects have completed their counts. To fix this, add the lines

.. code-block:: java

	t1.join();
	t2.join();

after the threads are started. The ``join`` method does not return until the thread has died. Thus, these calls delay the printing of results until both of the counting threads have completed.

After adding these lines, recompile the program and run it again. The counted number of primes still does not agree with the value given by the serial program. Can you identify the cause of this difference? See if you can figure it out before moving on.

