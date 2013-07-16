=================================
Looking Ahead: Parallel Patterns
=================================

Parallel Design Patterns  
-------------------------

As described in the module Introduction to Parallel Design Patterns, the OPL system of parallel patterns is organized into four levels. 
  * The top level, called the **Application Architecture Level**, consists of structural and computational patterns for designing large pieces of software (whether or not that software will be implemented with parallelism). 
  * Implementing this problem will not require a large software application – it’s more on the scale of a particular iterative computation *within* an application. Therefore, we will not need to consider a combination of structural and computational patterns to solve this problem. We need only find a way to add up many areas of rectangles, then multiply by 2. 
  * The next level of OPL patterns is the **Parallel Algorithms Strategy Level**. Two patterns at this level relate to our discussion above.
    - The **Data parallel** pattern involves applying the same computational operations to multiple data values, assuming that those operations can be performed independently for different data values. We can consider the values xias our multiple data values, and the rectangle-area computation as our operations.
    - The **Geometric decomposition pattern** involves breaking a large computation up into smaller “chunks” that we can compute in parallel. If we use a large number of rectangles N in our area approximation, we can apply this pattern by having multiple parallel computations that each add the areas of a subset of the rectangles.

**Note**: These patterns (as we have just described them) indicate strategies for designing parallel solutions to our problem which will prove useful when we write parallel code to compute π. However, our problem may not quite fit the exact OPL definitions of these terms. In particular, OPL refers to a data structure for both of these pattern names, whereas we might simply add up areas of rectangles in a loop without an explicit data 
structure of values. Also, the OPL **Geometric decomposition pattern** refers to concurrently updatable “chunks,” whereas our “chunks” of data values will never change. Nevertheless, these two algorithmic strategy patterns will help guide us to good parallelizations for given computational platforms.

Parallel algorithm strategy patterns.  
+++++++++++++++++++++++++++++++++++++++

These example programs for computing pi as an area of a semicircle exhibit a number of parallel design patterns.  Here, we will list many of the patterns present in some of our solutions of the problem of computing pi. Our example codes all illustrate the **Geometric decomposition**, but few fit OPL’s **Data parallel** parallel-algorithm strategy pattern.

  * Most of our parallel implementations exhibit the **Geometric decomposition** parallel-algorithm strategy pattern.  For example, in the OpenMP solution , the OpenMP system parallelizes a  for  loop by dividing its loop-control range into subranges.  We can view each subrange as a rectangle-based geometric approximation of some portion of the semicircle.  In fact, we use the pattern name Geometric Decomposition for any decomposition into “chunks” of data, such as contiguous subranges of a loop-control range, even if the resulting computation does not correspond to a geometric figure.  
  * Many programmers might say that these programs also shows “data parallelism,” since they process different data in different threads or processes.  In fact, some authors consider nearly any parallel computation to exhibit “data parallelism” (and other authors argue that nearly any parallel computation exemplifies “task parallelism”).  We will follow the OPL group and narrow the **Data parallel** parallel-algorithm strategy pattern to situations that apply a stream of instructions to elements of a data structure.  By this definition, the Array Building Blocks implementation and arguably the CUDA implementation qualify as using the **Data parallel** pattern.


Implementation strategy patterns.  
++++++++++++++++++++++++++++++++++

There are two types of patterns at this level: *program structure* patterns focus on organizing a program;  and *data structure* patterns describe common data structures specific to parallel programming.  Several such patterns appear among the code examples above.


  * The OpenMP solution exemplifies the **Loop parallel** program-structure pattern, because the OpenMP pragma requests that the summation loop be carried out in parallel, using OpenMP’s automatic mechanism for carrying out subranges of the summation in parallel. The TBB implementation pi_area_tbb.c provides another example, through its use of  tbb::parallel_reduce  which performs a multithreaded sum over subranges.
  * The **Strict data parallel** program-structure pattern provides a particular approach to implementing the higher-level **Data parallel**	 algorithm strategy pattern, which we may think of as an alternative to a **Loop parallel** implementation:  instead of visiting each value in turn, we think of processing all the values simultaneously, applying a single instruction stream to each of those values in parallel.  The ArBB code provides a higher-level example of this pattern, and the CUDA implementation shows that pattern at a lower level (i.e., closer to the hardware architecture).  
  * The Pthreads, Windows threads, and Go code examples illustrate the **Master-worker** algorithm strategy pattern, in which one thread assigns work tasks to worker threads or processes, seeking a balanced load.  When those tasks have varying execution times, a program may seek load balancing by having workers process one task at a time and return for a next task.  However, each of these cases achieves load balancing by assigning tasks known to require about the same time to each worker (i.e., adding up areas of nearly equal numbers of rectangles per worker).
  * The Pthreads and Windows threads implementations also show a **Fork-join** algorithm strategy pattern, in the way they create, launch, and await completion of their threads.  The Go implementation also demonstrates that pattern, accomplishing the “join” operation on each goroutine (process or thread) by collecting that thread’s results over the shared communication channel.  
  * Finally, we note that the ArBB code illustrates the **Distributed array** data structure pattern.  


Concurrent execution patterns. 
+++++++++++++++++++++++++++++++

This level consists of *advancing program counters* patterns, which concern the timing relationships for executing instructions among different threads or processes, and *coordination* patterns, which provide mechanisms for processes or threads to correctly access the data they need (cf. interprocess communication).  Our examples illustrate several concurrent-execution patterns.


  * The **SIMD (Single Instruction Multiple Data)** advancing-program-counter pattern appears in both the ArBB code and the CUDA code.  This pattern refines the **Strict data parallel** pattern, by specifying that concurrent execution should happen without programming how it will happen.  
  * The **Thread pool** advancing-program-counter pattern appears in the Pthreads and Windows threads code examples, in that those implementations each create an array of threads for parallel computation (here, in order to serve as worker threads in a master-worker strategy).  Programs with multiple parallel segments may reuse such threads for subsequent parallel operations;  however, this simple example has only one portion of parallel code, so these programs use their threads only once each.  
  * The reduce operations in the OpenMP and TBB  implementations present the **Collective communication** coordination pattern.  
  * The **Mutual exclusion** coordination pattern appears in the  pthread_mutex_t  variable  gLock  in the Pthreads example, and in the  CRITICAL_SECTION  variable  gCS  in the Windows threads example.
  * Finally, the Go code illustrates the **Message passing** coordination pattern, when goroutines computing subtotals of areas of rectangles communicate those results to the original process using the channel  parts .
