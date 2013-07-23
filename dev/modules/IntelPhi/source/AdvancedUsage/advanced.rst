.. comment
    Much of this was heavily plagiarized from various publications by Intel, all of which are under its copyright. All my work herein is given to the CSinParallel Project.

Advanced Usage
=========================

The Intel Phi coprocessor provides means of monitoring its processes and its health, as well as for monitoring its activity on a per-core basis.

Using Cilk to Parallelize the Code
----------------------------------

First download :download:`fib_cilk_mic.cpp <code/fib_cilk_mic.cpp>` and :download:`fib_cilk_mic.h <code/fib_cilk_mic.h>` to the Xeon.  Now, edit *fib_cilk_mic.cpp* to make use of the keywords ``_Cilk_shared`` and ``Cilk_offload``: these are equivalent to ``__attribute__((target(mic)))`` and ``#pragma offload target(mic)``.

To offload the complete calculation onto the Phi, we need to modify the original call in::

    std::cout<<"Fibonacci #"<< N << ":" << fib(N) << std::endl; 
    
to:: 

    std::cout<<"Fibonacci #"<<N<<":"<<_Cilk_offload fib(N)<<std::endl;

As both ``fib()`` and ``fib_serial()`` are executed on the Phi, we need to inform the compiler to create shared versions of them by extending the definitions of both functions to::

    _Cilk_shared int fib_serial (long long int N)
    _Cilk_shared int fib(long long int N)

Now, compile this with::

    icpc -o fib_cilk_mic fib_cilk_mic.cpp
    
Run ``fib_cilk_mic`` for different numbers in the range of 42 to 48. Also, time a run of ``fib_cilk_mic`` to calculate Fib #42 with ``time ./fib_cilk_mic 42``. Note the real, user, and system time.

Seeing the Hardware Specs of the Phi
----------------------------------------

First, log into the Xeon processor, making sure to use the -X flag at all ssh commands to be able X-forwarding. Then, run the command::

    /opt/intel/mic/bin/micinfo

This will tell you various pieces of information about the coprocessor, including what version of Linux it runs, its total number of active cores, and its die temperature.

Then, run some code on the coprocessor, either by running it directly on the Phi or by offloading it, as instructed in :ref:`sect-Using_Intel_Phi`, with one difference: make sure  that you run the command with *&* so that you can run other commands. Then, run ``ssh 172.31.1.1 top -n 1`` or ``ssh mic0 -n 1`` from the Xeon host to run ``top`` on the coprocessor.

Finally, start the system management and configuration tool and get familiar with the information that it reports::

   /opt/intel/mic/bin/micsmc &

This is what you needed X forwarding for, incidentally. The window may take a few seconds to load.

Now, again run some code on the coprocessor, and watch how ``micsmc`` responds.

Data Persistence
----------------

When you offload multiple stages of the computation to the Phi, you often want to reuse memory. Take a look at :download:`omp_3stageoffload_nopersist.cpp <code/omp_3stageoffload_nopersist.cpp>`, which is an artificial separation of :download:`omp_offload_ours.cpp <code/omp_offload_ours.cpp>`, into three stages. With the following commands, build it and run it, observing what happens::

    icc -O3 omp_3stageoffload_nopersist.cpp -o mmul_nopersist
    ./mmul_nopersist 2048
    
You will see the following error message::

    offload error: process on the device 0 was terminated by signal 11
    
Now compare omp_3stageoffload_nopersist to :download:`omp_3stageoffload_persist.cpp <code/omp_3stageoffload_persist.cpp>`. Notice the definitions in line 32-35:

.. literalinclude:: code/omp_3stageoffload_persist.cpp
   :lines: 32-35

These definitions use the ``alloc_if`` and ``free_if`` clauses, which both take one, essentially boolean, argument. The boolean argument of the ``alloc_if`` clause determines whether pointer variables in an ``in`` clause are allocated a fresh block of memory on the target when the following statement is executed on the target. If the expression evaluates to TRUE, a fresh memory allocation is performed for each variable listed in the clause. If the boolean condition evaluates to FALSE, no new memory is allocated and the existing pointer values on the target are reused. If neither is specified, fresh memory is allocated for each pointer variable by default (``alloc_if(1)`` is default).

``free_if`` is similar. If the argument evaluates to TRUE, then the memory pointed to by each variable listed in the clause is freed. If the condition evaluates to FALSE, no action is taken on the memory pointed to by variables in the list. If neither is specified, the memory is de-allocated by default (``free_if(1)`` is the default).

This explains the names that the above excerpt gives: ``ALLOC`` and ``FREE`` mean that the memory should be allocated and freed, respectively, and ``REUSE`` and ``RETAIN`` mean that the memory should be reused from a previous execution and that it should be retained for a future execution, respectively.

Now, note the use of these clauses in lines 123-129:

.. literalinclude:: code/omp_3stageoffload_persist.cpp
   :lines: 123-129

Then compile and run omp_3stageoffload_persist as follows::

    icc -O3 omp_3stageoffload_persist.cpp -o mmul_persist
    ./mmul_persist 2048
    
You should now get the expected result.

Asynchronous Data Transfer
--------------------------

Codes often operate on blocks of data which require the data block to be moved to the coprocessor at the start of the computation and back to the host at the end. Such codes benefit by the use of asynchronous data transfers where the coprocessor computes one block of data while another block is being transferred from the host. Asynchronous transfers can also improve performance for codes requiring multiple data transfers between the host and the coprocessor.

Take a look at the ``do_offload`` function in :download:`async_start.cpp <code/async_start.cpp>` and notice how the two arrays are processed one after the other using offload statements.

Then, compare this to :download:`async_ours <code/async_ours.cpp>`. Note the ``signal`` clauses in lines 57 and 78 in the ``offload_transfer`` pragma and the ``wait`` clause in line 87. The ``signal`` clause takes an argument which is an address expression associated with the dataset you wish to transfer. This initiates the transfer of the dataset and the CPU can continue past the pragma statement.

The wait clause creates a barrier, causing the activity specified in the pragma to begin only after all the data associated with its tag argument has been received.

Collecting and analyzing
