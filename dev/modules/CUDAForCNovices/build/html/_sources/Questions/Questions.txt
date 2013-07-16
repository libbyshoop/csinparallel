Questions
==========

* Describe the basic interaction between the CPU and GPU in a CUDA.

* The first activity in the CUDA lab involved commenting out various data transfer operations in the program.

	- What did this part of the lab demonstrate?

* Next, we compared the running time of two different procedures to run on the GPU. ::

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
	
* What did this part of the lab demonstrate?
