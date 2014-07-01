*********
Excercise
*********

CUDA Files to Download
**********************

You should download :download:`addVectors.cu <addVectors.cu>` and :download:`divergence.cu <divergence.cu>` for use in this excercies.

Understanding CUDA
******************

Now you will examine some of the factors that affect the performance of programs that use the graphics processing unit (GPU). In particular, you’ll see the cost of transferring data back and forth to the graphics card and how the different threads are joined together. 

Let’s begin by compiling this program:

.. code-block:: c

		nvcc -o addVectors addVectors.cu

nvcc is the name of the compiler for CUDA, the -o addVectors part is telling the compiler that you’d like to create an executable called “addVectors”, and the last part is the name of the file to compile. If you get any error messages, let me know; this probably means that there is a problem with your .cshrc or with copying the file. Once you’ve successfully compiled, you can run the program with the following:

.. code-block:: c

		./addVectors

You should get a printout with a time and a list of even numbers from 0 to 18.

Now let’s examine the code itself. Open the file ``addVectors.cu``.

Right near the top of the file is the definition of the function kernel:

.. code-block:: c

		__global__ void kernel(int* res, int* a, int* b) {
		   //function that runs on GPU to do the addition
		   //sets res[i] = a[i] + b[i]; each thread is responsible for one value of i

		   int thread_id = threadIdx.x + blockIdx.x*blockDim.x;
		   if(thread_id < N) {
		      res[thread_id] = a[thread_id] + b[thread_id];
		   }
		}

This code mostly looks like something you would see in C or Java except for some details of the top line. If you have previously worked with Java but not with C, ``int*`` is (for our purposes) equivalent to ``int[]`` in Java and the global part denotes this function as a kernel that can run on the GPU. The first line of code in the function body sets ``thread id`` as a unique identifier for each thread. Its value is calculated as the number of the thread within its block ``threadIdx.x`` plus the product of its block number ``blockIdx.x`` and the size of each block ``blockDim.x``; recall that threads are organized into blocks to simplify the bookkeeping for the tremendous number of threads in CUDA programs. The id value is then used as the index into the arrays so that each thread performs exactly one of the additions in the array sum.

Modifying CUDA
**************

Next, let’s modify the CUDA code. Begin by changing the value next to the ``N`` in the setting of the array length (right above the declaration of kernel). In order for the program to take a measurable amount of time, set this to 1 million (1000000). Since we don’t want to actually see the vector sum (one million numbers would make quite a mess for output), go down to the "verify results" part of the code (2nd to last “paragraph” of code in the file). Change this for loop to only print if it finds an index ``i`` such that ``res[i] != a[i]+b[i]``, i.e. the program has failed to correctly add the vectors.

Once you’ve made these changes, it’s time to exit the editor and recompile the program.  Then type the same compilation command as before (the one with nvcc). Now when you run the program, there shouldn’t be any output other than the time, which will be significantly larger (about 4 milliseconds).

Let’s see how this time breaks down between the data transfer between the main system (call the host) and the graphics card. Open the file again and we’ll comment out the line that calls the kernel. Use your text editor to search for this line, if you search for “kernel” you will find the as the third occurrence.

Now comment out this line and the “verify” paragraph (down a couple of paragraphs). Then recompile, and run the program again. The program is now transferring the data back and forth, but not actually performing the addition. You’ll see that the running time hasn’t changed much. This program spends most of its time transferring data because the computation does very little to each piece of data and can do that part in parallel.

To see this another way, open the file again and uncomment the kernel call and the verify paragraph. Then comment out the lines that transfer the data to the GPU; these are in the the paragraph commented as “transfer a and b to the GPU” (use the search function to find it). Then modify the kernel to set ``res[thread_id]`` to an arbitrary integer value instead of ``a[thread id] + b[thread id]``. (The program initializes ``a[i]`` and ``b[i]`` to both be ``i``; see the “set up contents of a and b” paragraph.) The resulting program should be equivalent to the original one except that instead of having the CPU initialize the vectors and then copy them to the graphics card, the graphics card is using its knowledge of their value to compute the sum, thus avoiding the first data transfer. Recompile and rerun this program; now the time is considerably less than the 4 milliseconds we started with. (We’re no longer copying the two vectors, which are each a million entries long...)

If you have additional time, copy the file divergence.cu from the course directory, adapting the command we used to copy addVectors.cu. This file contains two kernels, creatively named kernel 1 and kernel 2. Examine them and verify that they should produce the same result. The running time is quite different however; find the line in main that calls kernel 2 and change it to call kernel 1 so that you can look at the difference in running time. This difference is caused by the fact that CUDA threads operate in lockstep; each thread in a warp spends time for each instruction that any thread in that warp wants to execute.

Questions
*********

- Describe the basic interaction between the CPU and GPU in a CUDA program.

- The first activity in the CUDA lab involved commenting out various data transfer operations in the program. What did this part of the lab demonstrate?
- Next, we compared the running time of two different procedures to run on the GPU.

.. code-block:: c

		__global__ void kernel_1(int *a)
		int tid = threadIdx.x;
		int cell = tid % 32;
		a[cell]++;
		}


		__global__ void kernel_2(int *a) {
		int tid = threadIdx.x;
		int cell = tid % 32;
		switch(cell) {
		    case 0:
		      a[0]++;
		      break;
		    case 1:
		      ...
		}


- What did this part of the lab demonstrate?
