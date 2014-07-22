About The CUDA architecture
===========================

CPUs are designed to process several sequential commands as 
quickly as possible. While threading is an important technique,
creating threads is an expensive opertation and state of the art
consumer CPUs can't handle more than 8 threads effeciently.

GPUs on the other hand are designed to process several sequential
commands as quickly as possible. These commands tend to be
relatively simple and so GPUs use more, slower processors to
solve large problems, like for instance, generate 1,000,000
polygons and calculate which should be painted on the screen.
They do this by managing hundreds of threads which are much less
expensive to create than CPU threads.

Physical Architecture
#####################

NVIDIA CUDA enabled cards contain one or more Streaming
Streaming Multiprocessor (SM). Each multiprocessor has shared 
memory that the other SMs cannot access. Each Streaming
Multiprocessor is further devided into a number of CUDA cores.
The latest Kepler cards have 192 cores per SM. Each CUDA code
is capable of running the same instruction on a warp of threads
simultaneously. Warp is a term that comes from weaving and simply
means a group of threads. All CUDA architectures to date have
used a warp size of 32.
