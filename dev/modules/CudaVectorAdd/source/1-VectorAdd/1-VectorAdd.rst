*******************
Programming in CUDA
*******************

New Modifiers
#############

Open vectorAdd6.cu and look at the code. Notice that some of the functions have a ``__global__`` modifier in front of them. 
This denotes a function that runs on the machine that can be called anywhere in the program. 
There are two other modifiers, ``__device__`` and ``__host__`` that are not used in the program.
``__device__`` methods are run on the GPU and can only be called by ``__global__`` and ``__device__`` methods. 
By default, all other methods are ``__host__`` methods and can only be called by other ``__host__`` methods

Threads
#######

CUDA splits it's treads into three dimensional blocks which are arranged into a two dimensional grid.
Threads in the same block all have access to a local shared memory which is faster than the GPU's global memory. 
#TODO find a good graphic.
CUDA provides a handy type, dim3 to keep track of these dimensions you can declare dimensions like this ``dim3 myDimensions(1,2,3);`` 
Both blocks and grids use this type even though grids are 2D, just leave out the last argument or set it to one to create a grid dimension.
Each device has it's own limit on the dimensions of blocks, to find your device's information compile and run this program

Kernels
#######

CUDA threads are created by functions called kernels which must be ``__global__``.
Kernels are launched with an extra set of parameters enclosed by <<< and >>> the first argument is a dim3 representing the grid dimensions and the second is another dim3 representing the block dimensions.
You can also use ints instead of dim3s, this will create a nx1x1 grid.
After a kernel is launched it creates the number of threads specified and runs each of them.
Each thread has its own id represented by the variable threadIdx and a variable blockDim giving the size of the block
Both have three variables, x, y, and z representing the thread's coordinates with in the block and the max number of threads in each dimension respectively.
The block's coordinates within the grid are given by the variable blockIdx and the grid's dimensions are given by gridDim.
Both have 2 variable x and y representing the coordinates of th e block within the grid and the size of each dimension respectively
By using these variables we can create a unique id for each thread indexed from 0 to N where N is the total number of threads.
For a one dimensional grid and a one dimesional array this formula is ``blockIdx.x * blockDim.x + threadIdx.x``

Memory management in CUDA
#########################

CUDA 6 Unified Memory
*********************

.. warning:: This section is for CUDA 6 only. The following methods won't work on all devices. You can make and run this program to find out if your device can run this code

GPUs have their own dedicated RAM that is seperate from the RAM the CPU can use.
If we want both the CPU and GPU to be able to access a value we must tell our program to allocate unified memory.
We do this with the ``cudaMallocManaged`` function.
It works nearly identically to malloc but we have to pass in an address to a pointer as well as the size because cudaMallocManaged returns a cudaError_t instead of a pointer.
Make sure you read the next section too.
Even though unified memory will seem much simpler the exercises will require you to know how to use the regular methods as well.

Using cudaMalloc and cudaMemcpy
*******************************

