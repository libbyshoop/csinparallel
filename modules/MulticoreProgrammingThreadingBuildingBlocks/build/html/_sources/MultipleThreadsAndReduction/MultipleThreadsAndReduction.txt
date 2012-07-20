************************************
TBB, multiple threads, and reduction
************************************

The code above for a TBB trapezoidal computation produces an incorrect answer if there are multiple threads, because each thread attempts to update the shared variable ``integral`` without any mechanism to avoid one thread from overwriting the results produced by another thread.  We will solve this issue using a *reduction*, in which results will be computed in *local* variables for each thread, then those local results added together at the end.

1. To do the reduction in TBB, we will use the ``parallel_reduce`` call instead of the ``parallel_for`` call, and will use a modified class ``SumHeights2``. 

.. literalinclude:: trap-tbb2.cpp
    :language: cpp
    :lines: 1-65

:Comments:

  * The class ``SumHeights2`` handles the variable ``my_int`` differently than the class ``SumHeights`` does. Instead of ``SumHeights``'s misguided attempt to share main()'s memory location ``integral`` through reference types, the new class ``SumHeights2`` allocates a new separate (state) variable location my_int for each object of type ``SumHeights2`` (by avoiding reference types).

  * Also, ``my_int`` is a ``public`` state variable in ``SumHeights2``, instead of the default ``private`` visibility in the prior class ``SumHeights``. This makes it possible for a method of a ``SumHeights2`` object to compute a value and store that value in ``my_int``, then for another part of the code to access that computed value through that public state variable ``my_int``. (Alternatively, we could have made ``my_int`` private like the other state variables, and added a "getter" method ``get_my_int()`` to retrieve that computed value.)

  * The ``operator()`` definitions in the two classes differ in several ways.

    1. The code for the new class's operator ``SumHeights2::operator()`` begins my making local copies ``a2``, ``h2``, and ``int2`` of the variables ``my_a``, ``my_h``, and ``my_int``, and also storing the (unchanging) value of ``r.end()`` in another local variable. These local variable assignments are not necessary for the logical correctness of the code. Instead, they make it possible for the compiler to produce a more efficient computations. With this help, the compiler can realize that it's safe to use *registers* to implement those variables *instead of memory locations*, which would lead to faster access to those values.

    2. The loop is rewritten to use these local variables, but otherwise represents the same computation as in the previous ``SumHeights::operator()``.

    3. *After* the loop, the local variable ``int2`` is assigned to the state variable ``my_int``, in order to deliver the sum for this thread's subdivision (chunk) of the summation range.

    4. ``SumHeights2::operator()`` is *not* a ``const`` method. This means it's not safe for ``const`` objects to call this method -- they will be changed. In this case, the change is that ``my_int`` is changed when ``operator()`` is called.

  * The three-argument constructor for ``SumHeights`` is the same as the three-argument constructor for ``SumHeights2``, except for the handling of the third argument ``integral`` (discussed above).

  * However, the class ``SumHeights2`` has an additional constructor and an additional method ``join()``.

    1. The second constructor is called a *split constructor*. This constructor will be used to construct new instances of ``SumHeights2`` for additional threads brought into the summation computation.

    2. The method ``join()`` is used to add the partial sum from one thread's computation to a running sum -- i.e., to perform the reduction operation. 

  * Here is an overview description of the parallel computation for this program.

    1. An object ``sh2`` is allocated, using the three-argument constructor for ``SumHeights2``.

    2. The call to ``parallel_reduce()`` in ``main()`` performs ``sh2``'s ``operator()`` over the range 1 to n by subdividing (i.e., chunking) that range and assigning a thread to perform the trapezoidal sum for each chunk. 

    3. Each of those threads creates its own ``SumHeights2`` object using the splitting constructor.

      * The thread first calls that splitting object's ``operator()`` with that thread's range chunk to compute a partial sum.

      * Then, the thread calls ``sh2.join()`` with that splitting object as the argument, in order to add its partial sum to ``sh2``'s accumulator ``sh2.my_int``. 

    4. After all range chunks have been processed, ``parallel_reduce()`` finishes, leaving the final answer in the ``public`` state variable ``sh2.my_int``.

      * The splitting constructor for ``SumHeights2`` has a dummy argument of type ``split`` (defined by the TBB library), because without that extra argument, there would be no way for a compiler to tell the difference between a call to that splitting constructor and a call to ``SumHeight2``'s copy constructor.

2. Enter the program above, using the filename ``trap-tbb2.cpp``. Or you can download the file :download:`trap-tbb2.cpp <trap-tbb2.cpp>` 

   You can enter it on a lab machine, but then you'd have to disconnect Cisco VPN on your local machine (e.g., your laptop), ``scp`` the new file to your local machine, reconnect Cisco VPN, and ``scp`` to the MTL machine in order to transfer it to the MTL system.

3. Compile and test your ``trap-tbb2`` program on the MTL. Does it now produce the correct answer of 2 for the trapezoidal approximation?

4. Also time the performance of runs of this revised program, and compare to the time performance of runs of the prior program ``trap-tbb``.






































  
