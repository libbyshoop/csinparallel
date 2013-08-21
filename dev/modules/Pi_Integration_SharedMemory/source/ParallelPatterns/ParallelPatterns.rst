=================================
Parallel Patterns
=================================

If you have not yet done so, read the introdutory material found `here`_.

.. _`here`: "Link/back/to/Intro"

Parallel algorithm strategy patterns.  
+++++++++++++++++++++++++++++++++++++++

These example programs for computing pi as an area of a semicircle exhibit a number of parallel design patterns.  Here, we will list many of the patterns present in some of our solutions of the problem of computing pi.

  * Most of our parallel implementations exhibit the **Geometric decomposition** parallel-algorithm strategy pattern.  For example, in the OpenMP solution , the OpenMP system parallelizes a  for  loop by dividing its loop-control range into subranges.  We can view each subrange as a rectangle-based geometric approximation of some portion of the semicircle.  In fact, we use the pattern name Geometric Decomposition for any decomposition into “chunks” of data, such as contiguous subranges of a loop-control range, even if the resulting computation does not correspond to a geometric figure.  

Implementation strategy patterns.  
++++++++++++++++++++++++++++++++++

There are two types of patterns at this level: *program structure* patterns focus on organizing a program;  and *data structure* patterns describe common data structures specific to parallel programming.  Several such patterns appear among the code examples above.


  * The OpenMP solution exemplifies the **Loop parallel** program-structure pattern, because the OpenMP pragma requests that the summation loop be carried out in parallel, using OpenMP’s automatic mechanism for carrying out subranges of the summation in parallel. The TBB implementation pi_area_tbb.c provides another example, through its use of  tbb::parallel_reduce  which performs a multithreaded sum over subranges.
  * The Pthreads, Windows threads, and Go code examples illustrate the **Master-worker** algorithm strategy pattern, in which one thread assigns work tasks to worker threads or processes, seeking a balanced load.  When those tasks have varying execution times, a program may seek load balancing by having workers process one task at a time and return for a next task.  However, each of these cases achieves load balancing by assigning tasks known to require about the same time to each worker (i.e., adding up areas of nearly equal numbers of rectangles per worker).
  * The Pthreads and Windows threads implementations also show a **Fork-join** algorithm strategy pattern, in the way they create, launch, and await completion of their threads.  The Go implementation also demonstrates that pattern, accomplishing the “join” operation on each goroutine (process or thread) by collecting that thread’s results over the shared communication channel.  
  

Concurrent execution patterns. 
+++++++++++++++++++++++++++++++

This level consists of *advancing program counters* patterns, which concern the timing relationships for executing instructions among different threads or processes, and *coordination* patterns, which provide mechanisms for processes or threads to correctly access the data they need (cf. interprocess communication).  Our examples illustrate several concurrent-execution patterns.

  * The **Thread pool** advancing-program-counter pattern appears in the Pthreads and Windows threads code examples, in that those implementations each create an array of threads for parallel computation (here, in order to serve as worker threads in a master-worker strategy).  Programs with multiple parallel segments may reuse such threads for subsequent parallel operations;  however, this simple example has only one portion of parallel code, so these programs use their threads only once each.  
  * The reduce operations in the OpenMP and TBB  implementations present the **Collective communication** coordination pattern.  
  * The **Mutual exclusion** coordination pattern appears in the  pthread_mutex_t  variable  gLock  in the Pthreads example, and in the  CRITICAL_SECTION  variable  gCS  in the Windows threads example.
  * Finally, the Go code illustrates the **Message passing** coordination pattern, when goroutines computing subtotals of areas of rectangles communicate those results to the original process using the channel  parts .
