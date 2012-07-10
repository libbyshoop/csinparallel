**************
Thread Advance
**************

Vector Dot Product
##################

In this example we will see how to perform a dot product using GPU computation. We know that the result of vector addition is a vector, but the result of vector dot product is a number. However, we can divide the vector dot product process into two steps. We first use CUDA to the multiplication process. After this step, the device will return a vector with all its elements as multiplication results to the host code. Then the CPU can do all the adding up process.

Vector Dot Product source file:
:download:`Dot-GM.cu <Dot-GM.cu>`

The Device Code
***************

.. literalinclude:: Dot-GM.cu	
    :language: c
    :lines: 28-37

The device code is pretty straight forward. Each thread multiplies a pair of corresponding elements in two vectors. After each thread done their job for the first time, if there are still elements left unprocessed, they runtime will instruct the threads to do another round of computation until all the elements are processed.

The Host Code
*************

.. literalinclude:: Dot-GM.cu	
    :language: c
    :lines: 39-102

The host code the much like the vector addition example. We first allocate the memory on host memory and device memory. Then we initialize the matrices and fill them with data. After that we copy the data from host to device and execute the kernel code. Finally we transfer the data back from device memory to host memory. Do not forget to use Event API to measure the performance.

However, we need to point out two differences. 

.. literalinclude:: Dot-GM.cu	
    :language: c
    :lines: 41-42

First is that we need to declare one more pointer for the host code. In the vector addition example, two sets of array pointers are enough. However, in this example, we are returning a number. Therefore, a pointer which pointing to that number is essential.

.. literalinclude:: Dot-GM.cu	
    :language: c
    :lines: 83-87

Another difference is that we need to add up all the elements in the returned vector. This can simply be done by adding a for loop in the host code. Notice that we put this loop outside of the Event API so that it will not interfere our timing result.

You can verify the result by adding the following code into the host code.

.. literalinclude:: Dot-GM.cu	
    :language: c
    :lines: 89-91

For people who are familiar with discrete math, the code above should be simple to apprehend. This function will give the result through a more *clever* way.

==================================================================================================

Vector Dot Product with Reduction
#################################

In the previous example, you might have the question: why do we need to return the whole array back to the CPU? Is is possible for use to first reduce them a little bit and then return it to the CPU?

Well, we can do reduction in CUDA. In this chapter, we will see how to use reduction in CUDA. But before we proceed, we first need to know something about shared memory in CUDA. In each of the blocks we create, CUDA runtime will assign a region memory to this block. This type of memory is called shared memory. When you are declaring your variables, you can add the CUDA C keyword **__shared__** to make this variable reside in shared memory. Why do we need shared memory?

When we are learning Block and Thread, we had the question why would we need two hierarchy to organize threads question in mind. Well, part of the reason is that we can benefit from shared memory by organize threads in blocks. When we declare a variable and make it reside in shared memory, CUDA runtime creates a copy of this variable in each block you launched in host code. Every threads in one block shares this memory, which means they can see and modify their shared memory. However, they cannot see or modify shared memory assigned in other blocks. What does this mean to us? Well, if you can have one region of memory that is private to threads in one block, you can explore ways to facilitate communication and collaboration between threads within this block. 

Another reason we like to use shared memory is that it is faster than global memory we used to use. As the latency of shared memory tends to be far lower than global memory, it is the ideal choice to serve as cache in each block. 

In this example, we will see how we use shared memory to serve as a cache-per-block and how we can perform reduction on it.

Vector Dot Product with Reduction source file:
:download:`Dot-SM.cu <Dot-SM.cu>`

The Device Code
***************

.. literalinclude:: Dot-SM.cu	
    :language: c
    :lines: 30-64

.. literalinclude:: Dot-SM.cu	
    :language: c
    :lines: 32-33

In the two lines of code above, we declared a cache in shared memory for this block. We will then use this cache to store each thread's running sum. You can also see that we set the size of the cache same as numThread so that each thread in the block can have its own place to store its running sum. Notice we only create one copy of cache instead of creating numBlock copies. The compiler will automatically create a copy of cache in each block's shared memory, meaning we only have to declare one copy.

.. literalinclude:: Dot-SM.cu	
    :language: c
    :lines: 40-43

The actual vector dot product computation is the similar to what we seen in the global memory version. However, there is little difference. You can see from the code above, instead of using

.. literalinclude:: Dot-GM.cu	
    :language: c
    :lines: 34

we use

.. literalinclude:: Dot-SM.cu	
    :language: c
    :lines: 42

The reason causes this difference is that we are finding a running sum in this example. In the previous example, as we are returning a vector exactly the same size as the input, we have enough space to store each multiplication results. Even we have less threads than vector dimension so that each thread will compute more than one value, they can store each value in different places. However, in this example, we have exactly the same amount of place for storage as number of threads in each block. If any thread compute more than once, they still have only one place to store values. This bring us to why we need to use running sum of each thread.

 
You may notice we add a line of code you have never seen before.

.. literalinclude:: Dot-SM.cu	
    :language: c
    :lines: 49-50

We have seen code similar to this before. When we are learning how to use CUDA Event API to measure performance, we used command **cudaEventSynchronize()** to synchronize the stop event. The purpose of **__syncthreads()** is somehow similar to the command **cudaEventSynchronize()**. 
When we are using shared memory to facilitate communication and collaboration between threads, we need to create a way to synchronize them as well. For example, if one thread is writing a number to the shared memory and another thread need to use that number for further computation, we want the first thread to finish its work before the second thread executes its command. What the command __syncthread() will do, is essentially create a barrier for all the threads and block them from executing further command. After all threads have finished executing commands before __syncthread(), then they can all proceed to the next command. 

After we make sure all the elements in cache is filled, we can proceed to the reduction process.

.. literalinclude:: Dot-SM.cu	
    :language: c
    :lines: 52-59

Suppose we have a cache with 256 entries. What the code above would do is that it first take 128 thread in the block and each thread will add two of the values in cache[a] and cache[a+128] and store the value back to cache[a]. Then in the second iteration it will take 64 thread in the block and each thread will add values in cache[a] and cache[a+64] and store the value back to cache[a]. After log2(numThread) times of operation, we would have the sum of all 256 values stored in the first element of the cache. Be really careful that we need to synchronize threads every time after we perform one reduction.

Finally, we choose the thread with index 0 to write the result back to the global memory.

.. literalinclude:: Dot-SM.cu	
    :language: c
    :lines: 61-64

The Host Code
*************

In general, the host code of the shared memory version is very similar to that of the global memory version. However, there are several difference we need to point out.

.. literalinclude:: Dot-SM.cu	
    :language: c
    :lines: 74, 93-94

In the above two lines of code, in the previous example, we declared two sets of pointers with each pointer pointing to array having the same size. This time, however, we need the output array to be smaller than the vector size. Since every block's will write only one number back to the global memory, the size of output array should be numBlock. 

Another point worth mentioning is that we define the numBlock in the following way instead of assign a number to it.

.. literalinclude:: Dot-SM.cu	
    :language: c
    :lines: 24,28

When we are choosing the number of blocks to launch in this problem, we faces to requirements. First is that we should not create too many blocks. In the final step where all the results returned by all the blocks are summed up, we are using CPU to compute. This means if we create too many blocks, we would leave CPU too much workload. Another requirements is that we cannot assign too less blocks either. As we can only fit 256 threads in each block, if we assign not enough blocks, we would end up having each thread doing many times of computation. Facing this two requirements, we came up with the solution above. We use the smaller number between 32 and (N+numThread-1) / numThread. The function (N+numThread-1) / numThread gives the smallest multiple of numThread that is equal or larger than the vector size. Calculating this number will ensure we have just enough blocks so that each element in a small vector has its own thread. If we are facing a small vector, we can us the later to assign not too many blocks. If we are facing a gigantic vector, 32 blocks is somehow enough to keep the GPU busy.

Be aware the number 32 was given by a CUDA programming book that is several years old. We decide to use it because we think its safe and yet sufficient for our problem size. If you are dealing with much larger problem size and have much more powerful GPU cards in hand, feel free to stretch this number to thousands even hundreds of thousand.
