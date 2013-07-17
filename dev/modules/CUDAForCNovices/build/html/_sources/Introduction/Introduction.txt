The Lab
============================

In this lab, you’ll be examining some of the factors that affect the
performance of programs that use the graphics processing unit (GPU). In
particular, you’ll see the cost of transferring data back and forth to
the graphics card and how the different threads are joined together.
We’ll see the good side of GPU programming (that it can make programs
faster) tomorrow in lecture.

I encourage you to work with someone else for this lab;
having fewer programs running will make the timing experiments more
accurate and I think having a partner will help with all the new
material (at least that way, you can both be lost together...).

CUDA
-----

To begin, you need to get onto a system with a graphics card. Once you have a terminal open, you need to add two lines to a configuration file so that you can use the compiler for CUDA. For this you need a text editor. Begin by opening your .cshrc file. If you are using emacs, begin by typing :: 

	emacs .cshrc

This file is a configuration file used by tcsh, which is your *shell*,
the program that you’re typing commands into. This file is read every
time the shell starts. Add the following lines:

::

      set path=(${path} /usr/local/cuda/bin)
      setenv LD_LIBRARY_PATH /usr/local/cuda/lib64

Once you’re done with that, close your text editor by hitting Control-X followed by
``Control-C`` (abbreviated ``C-x C-c``). You’ll be asked if you want to save the
file; answer yes.

As mentioned above, the .cshrc file is run automatically when the shell
starts, but you want to run it now. To do this, simply type

::

      source .cshrc

(You won’t have to do this step (or edit the .chsrc file) in the future
because the shell runs the file every time it starts up.) If it works,
you’ll just get another prompt. if you get any error
messages, that means something went wrong with the step.

Now we can start actually working with CUDA code. We’ll begin with a
program which add
two vectors. Copy the code into your directory from either the links on the home page of this module, or the text on the 'Add Vectors' page of the module.  

Let’s begin by compiling this program:

::

      nvcc -o addVectors addVectors.cu

``nvcc`` is the name of the compiler for CUDA, the ``-o addVectors`` part is
telling the compiler that you’d like to create an executable called
“addVectors”, and the last part is the name of the file to compile. If
you get any error messages, this probably means that there
is a problem with your .cshrc or with copying the file. Once you’ve
successfully compiled, you can run the program with the following:

::

      ./addVectors

You should get a printout with a time and a list of even numbers from 0
to 18.

Now let’s examine the code itself. Open the file. If you are using emacs, do this by typing ::

	emacs addVectors.cu

The file will display better if you put the program you are using
to edit the file into the mode for C programs. In emacs, do this by hitting
``Escape-X`` and then typing in ``c-mode`` (enter). You’ll see the first part of
the last word at the bottom of the screen (sort of status bar) change
from “Text” to “C”.

Right near the top of the file is the definition of the function kernel:

::

      __global__ void kernel(int* res, int* a, int* b) {
        //function that runs on GPU to do the addition
        //sets res[i] = a[i] + b[i]; each thread is responsible for one value of i

        int thread_id = threadIdx.x + blockIdx.x*blockDim.x;
        if(thread_id < N) {
          res[thread_id] = a[thread_id] + b[thread_id];
        }
      }

This mostly looks like Java except for some details of the top line.
``int\*`` is (for our purposes) equivalent to ``int[]`` in Java and the
``__global__`` part denotes this function as a kernel that can run on
the GPU. The first line of code in the function body sets ``thread_id`` as
a unique identifier for each thread. Its value is calculated as the
number of the thread within its block (``threadIdx.x``) plus the product of
its block number (``blockIdx.x``) and the size of each block (``blockDim.x``);
recall that threads are organized into blocks to simplify the
bookkeeping for the tremendous number of threads in CUDA programs. The
id value is then used as the index into the arrays so that each thread
performs exactly one of the additions in the array sum.

Next, let’s modify the CUDA code. Begin by changing the value of N
in the setting of the array length (right above the declaration of
``kernel``. In order for the program to take a measureable amount of time,
set this to 1 million (1000000). Since we don’t want to actually see the
vector sum (one million numbers would make quite a mess for output), go
down to the “verify results” part of the code (2nd to last “paragraph”
of code in the file). Change this for loop to only print if it finds an
index ``i`` such that ``res[i] != a[i]+b[i]``, i.e. the program has failed to
correctly add the vectors.

Once you’ve made these changes, it’s time to exit the editor and
recompile the program. To exit emacs, type ``C-x C-c`` again. Then type the
same compilation command as before (the one with ``nvcc``). Now when you run
the program, there shouldn’t be any output other than the time, which
will be significantly larger (about 4 milliseconds).

Let’s see how this time breaks down between the data transfer between
the main system (call the *host*) and the graphics card. Open the file
again and we’ll comment out the line that calls the kernel. To file this
line, we’ll use the search function in the text editor. If you're in emacs, type Control-s (``C-s``) and
then type ``kernel``. The cursor will jump to the first place that this
string appears in the file, which is the kernel’s definition. Hit ``C-s``
again to move to the next occurrence. This still isn’t the line so hit
it again. This is the correct line to terminate the search by hitting
enter.

Now comment out this line and the “verify” paragraph (down a couple of
paragraphs). Then exit the text editor, recompile, and run the program again. The
program is now transferring the data back and forth, but not actually
performing the addition. You’ll see that the running time hasn’t changed
much. This program spends most of its time transferring data because the
computation does very little to each piece of data and can do that part
in parallel.

To see this another way, open the file again and uncomment the kernel
call and the verify paragraph. Then comment out the lines that transfer
the data to the GPU; these are in the paragraph commented as
*“transfer a and b to the GPU”* (use the search function to find it). Then
modify the kernel to use ``thread_id`` instead of ``a[thread_id]`` and
``b[thread_id]``. (The program initializes ``a[i]`` and ``b[i]`` to both be ``i``; see
the “set up contents of a and b” paragraph.) The resulting program
should be equivalent to the original one except that instead of having
the CPU initialize the vectors and then copy them to the graphics card,
the graphics card is using its knowledge of their value to compute the
sum, thus avoiding the first data transfer. Recompile and rerun this
program; now the time is considerably less than the 4 milliseconds we
started with. (We’re no longer copying the two vectors, which are each a
million entries long...)

If you have additional time, copy the file ``divergence.cu`` from the home page. This file
contains two kernels, creatively named ``kernel_1`` and ``kernel_2``. Examine
them and verify that they produce the same result. The running
time is quite different however; change the call in ``main`` and look at the
difference in running time. This is caused by the fact that CUDA threads
operate in lockstep; each thread in a warp spends time for each instruction that
*any* thread in that warp wants to execute.
