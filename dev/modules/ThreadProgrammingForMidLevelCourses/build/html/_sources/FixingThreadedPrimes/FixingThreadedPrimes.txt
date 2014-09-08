**************************
Fixing ThreadedPrimes.java
**************************

The problem is that our program suffers from a race condition when it updates ``pCount``. Although ``pCount++`` is a single Java statement, the thread can be interrupted in the middle of it, leading to some primes not being included in the count. To fix this, we will add a lock around the increment line. Create a static object called ``lock`` in the ``ThreadedPrimes`` class. (The type does not matter; you could simply make it an ``Object``.) Then, embed the line incrementing ``pCount`` inside a block

.. code-block:: java

	synchronized(lock) {
	   ...
	}

With this, the increment can only occur after the lock associated with the lock object has been acquired. In Java, every object has a lock associated with it; we just made a static object so it could be shared by the two occurrences of ``PrimeFinder``. The ``synchronized`` statement automatically acquires the lock before entering the protected block of code and automatically releases it at the block’s end.

After this change, the number of primes should be properly calculated. One potential criticism of this solution, however, is that the two threads constantly need the lock. This does not cause the program much overhead since the critical section is so short, but a better solution is possible. Our prime algorithm is what is called embarrassingly parallel because it is so easy to break into subproblems that can run independently; testing the primality of each number is completely independent of all the other numbers. With a problem like this, it seems wasteful to use the lock so heavily. Instead, modify the solution so that each ``PrimeFinder`` object keeps its own count of the number of primes it has found. Then, after it has counted all the primes within its range, it should update ``pCount`` (using a lock of course). With this modification, each thread only claims the lock once.

So how fast is the resulting program? Run it using time and compare the result to the serial version. The first two numbers should be nearly identical between the two implementations because they do almost exactly the same things and therefore need the same amount of CPU time. The wall clock times should differ, though. The speedup of our parallel implementation is the serial wall clock time divided by the parallel wall clock time. Since your machine should have at least two processors and we are creating two threads to run on them, an ideal speedup would be 2. In practice, parallel programs cannot normally achieve linear speedup, as this would be called, since some parts of the program run serially and communication (the lock in our case) incurs some overhead. However, we should be able to get a speedup extremely close to 2 for this program because it splits into parallel subproblems so well.

Unfortunately, I get a speedup of only about 1.6. Can you explain this? (Hint: What happens when you set the two parts to find primes within the ranges 1-1,100,000 and 1,100,001–2,000,000 instead splitting the test region evenly?) Once you understand the problem, see if you can fix it. (There are at least two fairly straightforward ways to do so.)

If you have extra time, continue to experiment with threaded programs to find prime numbers. What happens if you create more than two threads? Another way to speed up the computation is to only divide a candidate number by the primes smaller than its square root rather than all values smaller than its square root; can you use this to further improve your performance? (You will probably want to use the Vector class, which is a synchronized version of ArrayList.) You may also want to look into the Sieve of Eratosthenes, an ancient Greek algorithm for finding prime numbers up to a given limit.
