Advanced Topic:  Seeds For Different Threads
===============================================


Adding OpenMP pragmas on the 'workhorse' for loops where most of the computation is being done is often a helpful way to make your code run faster.  In the case of Monte Carlo simulations, there is one issue that should be addressed to ensure the best random distribution of numbers from the random number generator functions.  We must start each thread with a different seed.

Recall that random number generators start from a 'seed' large integer and create a sequence of integers by permuting the seed and each successive integer in a manner that ensures they are distributed across the range of all integers.  The key point is this: *the sequence of numbers from a random generator is always the same when it starts with the same seed*.  In code where we fork threads to do the work of generating random numbers, we lose the desired random distribution if each thread begins generating random numbers from the same seed.

The solution to this issue for threaded code, which you can :download:`download as coinFlip_omp_seeds.cpp <../code/montecarlo_openmp_cpp/coinFlip/coinFlip_omp_seeds.cpp>`, is to ensure that each thread has its own seed from which it begins generating its sequence of integers.  Let's revisit the coin flipping example.  Instead of generating one seed in main using time(), we can save a seed for each thread in an array and devise a function to create all of the seeds, based on the number of threads to run.  We can add this code at the beginning of our original file:

.. literalinclude:: ../code/montecarlo_openmp_cpp/coinFlip/coinFlip_omp_seeds.cpp
   :language: c++
   :lines: 22-44

Not how we change the seed value for each thread by using the thread's id to manipulate the original integer obtained from time().

Then later in the main function, we add a call to this function:

.. literalinclude:: ../code/montecarlo_openmp_cpp/coinFlip/coinFlip_omp_seeds.cpp
   :language: c++
   :lines: 72-75
   
For each trial, we still parallelize the workhorse for loop, while also ensuring that each thread running concurrently has its own seed as the starting point for later numbers.

.. literalinclude:: ../code/montecarlo_openmp_cpp/coinFlip/coinFlip_omp_seeds.cpp
   :language: c++
   :lines: 112-132
   

Study the above code carefully and compare it to our first version below.  The `pragma omp` directive above is forking the new set of threads, which do a bit of work to set up their own seeds.  Then the `pragma omp for` directive is indicating that those same threads should now split up the work of the for loop, just as in our previous example using the OpenMP pragma.  The first OpenMP version we showed you looked like this:

.. literalinclude:: ../code/montecarlo_openmp_cpp/coinFlip/coinFlip_omp.cpp
   :language: c++
   :lines: 82-95
   
.. note:: A common 'gotcha' that can cause trouble is if you accidentally use the original
    `pragma omp parallel for` directive near the for loop in the new version.  This causes     incorrect unintended behavior. Remember to remove the **parallel** keyword in the inner block when nesting bloaks as shown in the new version where we set up seeds first before splitting the loop work.

.. NOET to self:  get rid of schedule(static) clause?

Note that as before, in linux we need to use the rand_r() function for thread-safe generation of the numbers.  However, in Windows, the rand() function is thread-safe.

Try this yourself
-------------------

Try creating versions of the Roulette wheel simulation or drawing four suits that ensure that each thread is generating numbers from its own seed.


