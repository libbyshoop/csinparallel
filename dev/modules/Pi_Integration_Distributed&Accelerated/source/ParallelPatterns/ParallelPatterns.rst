=================================
Parallel Patterns
=================================

If you have not yet done so, read the introdutory material found `here`_.

.. _`here`: "Link/back/to/Intro"

Parallel algorithm strategy patterns.  
+++++++++++++++++++++++++++++++++++++++

These example programs for computing pi as an area of a semicircle exhibit a number of parallel design patterns.  Here, we will list many of the patterns present in some of our solutions of the problem of computing pi. Our example codes all illustrate the **Geometric decomposition**, but few fit OPL’s **Data parallel** parallel-algorithm strategy pattern.

  * Many programmers might say that these programs also shows “data parallelism,” since they process different data in different threads or processes.  In fact, some authors consider nearly any parallel computation to exhibit “data parallelism” (and other authors argue that nearly any parallel computation exemplifies “task parallelism”).  We will follow the OPL group and narrow the **Data parallel** parallel-algorithm strategy pattern to situations that apply a stream of instructions to elements of a data structure.  By this definition, the Array Building Blocks implementation and arguably the CUDA implementation qualify as using the **Data parallel** pattern.


Implementation strategy patterns.  
++++++++++++++++++++++++++++++++++

There are two types of patterns at this level: *program structure* patterns focus on organizing a program;  and *data structure* patterns describe common data structures specific to parallel programming.  Several such patterns appear among the code examples above.

  * The **Strict data parallel** program-structure pattern provides a particular approach to implementing the higher-level **Data parallel**	 algorithm strategy pattern, which we may think of as an alternative to a **Loop parallel** implementation:  instead of visiting each value in turn, we think of processing all the values simultaneously, applying a single instruction stream to each of those values in parallel.  The ArBB code provides a higher-level example of this pattern, and the CUDA implementation shows that pattern at a lower level (i.e., closer to the hardware architecture).  
  * Finally, we note that the ArBB code illustrates the **Distributed array** data structure pattern.  


Concurrent execution patterns. 
+++++++++++++++++++++++++++++++

This level consists of *advancing program counters* patterns, which concern the timing relationships for executing instructions among different threads or processes, and *coordination* patterns, which provide mechanisms for processes or threads to correctly access the data they need (cf. interprocess communication).  Our examples illustrate several concurrent-execution patterns.

  * The **SIMD (Single Instruction Multiple Data)** advancing-program-counter pattern appears in both the ArBB code and the CUDA code.  This pattern refines the **Strict data parallel** pattern, by specifying that concurrent execution should happen without programming how it will happen.  
  