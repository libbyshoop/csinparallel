.. role:: uline

*************
Looking Ahead
*************

Parallel patterns
#################

#. Structural and computational patterns (Application architecture level). Map-reduce is the structural pattern.

#. Parallel algorithm strategy patterns. We use the Data parallelism pattern in parallel implementations of this exemplar, in which we apply our ``Map()`` algorithm to each element of the task queue (or vector) for independent computation.

#. Implementation strategy patterns.  

	- Our map-reduce algorithms represented in ``MR::run()`` methods for parallel implementations use a :uline:`Master-worker` program-structure pattern, in which one thread launches multiple worker threads and collects their results. In the case of OpenMP and Hadoop, the master-worker computation is provided by the underlying runtime or framework. In addition, the Boost threads code exhibits an explicit :uline:`Fork-join` program-structure pattern, and the OpenMP code’s ``omp parallel for`` pragma implements the :uline:`Loop parallel` program-structure pattern, as does the Boost threads code, with its ``while`` loop, and the Go implementation with its ``for`` loop in its “map” stage. In addition, Hadoop proceeds using an internal :uline:`Bulk synchronous parallel (BSP)` program-structure pattern, in which each stage completes its computation, communicates results and waits for all threads to complete before the next stage begins. The ``MR::run()`` methods of our C++ parallel implementations for multicore computers also wait for each stage to complete before proceeding to the next, which is similar to the classical BSP model for distributed computing. The Go implementation exhibits BSP at both ends of its sort stage, when it constructs an array of all pairs and completes its sorting algorithm. Most of our implementations use a :uline:`Task queue` program-structure pattern, in which the task queue helps with load balancing of variable-length tasks.   

	- Besides these program-structure patterns, our examples also illustrate some *data-structure* patterns, namely :uline:`Shared array` (which we’ve implemented using TBB’s thread-safe ``concurrent_vector``\ ) and :uline:`Shared queue` (TBB’s ``concurrent_bounded_queue``\ ). Arguably, the use of channels ``ligands`` and ``pairs`` in the Go implementation constitutes a :uline:`Shared queue` as well.

#. We named our array of threads ``pool`` in the Boost threads implementation in view of the :uline:`Thread pool advancing-program-counter` pattern. Note that OpenMP also manages a thread pool, and that most runtime implementations of OpenMP create all the threads they’ll need at the outset of a program and reuse them as needed for parallel operations throughout that program. Go also manages its own pool of goroutines (threads). The Go example demonstrates the :uline:`Message passing coordination` pattern. We used no other explicit coordination patterns in our examples, although the TBB shared data structures internally employ (scalable) :uline:`Mutual exclusion` in order to avoid race conditions.
