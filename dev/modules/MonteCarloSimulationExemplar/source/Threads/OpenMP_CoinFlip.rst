Coin-flipping in Parallel
==========================

Now that we know a bit about how OpenMP works to provide threads that run code in parallel, let's look at how we can update our coin-flipping example.    
The places in this code where you see this comment:

.. literalinclude:: ../code/montecarlo_openmp_cpp/coinFlip/coinFlip_omp.cpp	
    :language: c
    :lines:  35-35

indicate where OpenMP was used to enable running the origianl coin-flipping code example on multiple threads, or where the code needed changes to enable running on multiple threads.  Examine these places in the following code:
    
.. literalinclude:: ../code/montecarlo_openmp_cpp/coinFlip/coinFlip_omp.cpp	
    :language: c
    :linenos:

Some notes about this code
--------------------------


#. On line 15 we include the OpenMP library.

#. On lines 75 and 98 we use the OpenMP function to return a wall clock time in seconds.  The difference between these provides the total amount of time to run the section of code enclosed by these lines.  Note that this OpenMP function called `omp_get_wtime` specifically provides the overall time for the threaded code to run.  We need to use this function because the original method using the clock() function does not work properly with threaded code.


#. Lines 82 - 85 indicate the setup for running the for loop of coin flips in equal numbers of iterations per thread. There are several directives needed to be added to the parallel for pragma:
    
    
    - `num_threads(nThreads)` designates how many threads to fork for this loop.
    - `default(none)` designates that all variables in the loop will be defined as either private within each thread or shared between the threads by the next three directives.
    - the \\ designates that the pragma declaration is continuing onto another line
    - `private(numFlips, seed)` designates that each thread will keep its own private copy of the variables numFlips and seed and update them independently.
    - `shared(trialFlips)` designates that the variable trialFlips is shared by all of the threads (this is safe because no thread will ever update it.)
    - `reduction(+:numHeads, numTails)` is a special indicator for the the two values numHeads and numTails, which need to get updated by all the threads simultaneously.  Since this will cause errors when the threads are executing, typically the OpenMP threaded code will have each thread keep a private copy of these variables while they execute their portion of the loop.  Then when they join back after they have finished , each thread's private numHeads and numTails sum is added to an overall sum and stored in thread 0's copy of numHeads and numTails.
    
#. You can download the file :download:`coinFlip_omp.cpp <../code/montecarlo_openmp_cpp/coinFlip/coinFlip_omp.cpp>` and try this code  yourself.  If you have 4 cores available on your computer, you should see the longer trials with many coin flips run almost four times faster than our earlier sequential version that did not use threads.

