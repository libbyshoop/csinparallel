*******
Streams
*******

Virtual Memory and Paging
#########################

Before we entered the world of CUDA, we have learned something about virtual memory in computer systems. If you do not remember much, lets jog your memory a little bit. When you launch multiple programs in you computer, you are taking the risk of running out of memory. Although modern customer end computers usually have 4GB even more memory, it is still not enough for running many programs at the same time. Through virtual memory, or to be more specific, through the process of paging, a computer can store and retrieve data from secondary storage for use in main memory. Simply put, the operating system would transfer some of the data stored in memory that are not currently at use to secondary storage so that the system can free physical memory and proceed to other programs. 

Pageable Host Memory and Page-locked Host Memory
################################################

The cudaHostAlloc() Function
****************************

In previous examples, we have been using the function malloc() in standard C libraries to allocate memory on host memory. The malloc() function would allocate standard pageable host memory for our program. In this chapter, we introduce the cudaHostAlloc() function in CUDA library. Different from malloc() function, cudaHostAlloc() function would allocate page-locked host memory for our program. When you allocate pageable memory, you are taking the risk of operating system might page this memory out to secondary storage if it feels that the system is running out of physical memory. However, when you allocate page-locked memory, the operating system guarantees that it will never page this memory to secondary storage and ensure that this memory will always reside on physical memory. 

Knowing the difference, you may want to ask the question? Why do we need page-locked memory? Well, the answer to this question lies in how CUDA driver move data between different kind of memories. When you call cudaMemcpy() function from the host code, CUDA driver will start to transfer data from host memory to GPU memory, or vice versa. What happens is the driver will first move the data from a pageable system buffer to a page-locked system buffer. Then it will move the data from page-locked memory to the GPU. However, if we instruct the system to store the data on the page-locked system buffer in the first place, we can skip the first step and start from the second step. 

Therefore, if we allocate memory on pageable host memory, our data transfer speed is limited by the lower of the system front-side bus speed (first step) and PCI Express speed (second step). However, if we are using page-locked memory, we are limited only by PCI Express speed. The machine we are using is equipped with Intel Core 2 Quad processor with front-side bus speed between 8512MB/s to 10656MB/s, which is roughly 8GB/s to 10GB/s. The graphics card is installed on PCI Express x16 2.0 slot, which also has 8GB/s transfer speed. Therefore, if we are using page-locked memory instead of pageable memory, we can see the transfer time reduce to at most 50% of the original time.

CUDA Streams
############


