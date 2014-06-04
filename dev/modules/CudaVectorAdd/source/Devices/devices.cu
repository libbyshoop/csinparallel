/**
*Created by Jeffrey Lyman for  use in COMP 240 at Macalester College
*Licensed under the GNU Public License
*
*This program prints out available NVIDIA graphics cards as well as
*their compute capability and whether they are compatible with CUDA 6's
*unified memory methods such as cudaMallocManaged
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(){
   int devCount;
   cudaGetDeviceCount(&devCount);
   printf("You have %d CUDA devices\n", devCount);
   cudaDeviceProp props;
   for (int i=0; i<devCount; i++){
        cudaGetDeviceProperties(&props, i);
        printf("\nDevice %d is named %s\n", i, props.name);
        printf("\tDevice compute capablitiy %d.%d\n", props.major, props.minor);
        printf("\tUse compiler flag -gencode arch=compute_%d%d,code=sm_%d%d\n",
                props.major, props.minor, props.major, props.minor);
 //cudaMallocMangaged etc. is only available for devices with compute
 //capablitity >=3
        if(props.major >= 3){
            printf("\tThis device can use CUDA 6 unified memory methods\n");
        } else {
            printf("\tThis device is incompatable with CUDA 6 unified memory methods\n");
        }
        //max thread/grid size
        printf("\tMax grid size %d x %d x %d\n",
                props.maxGridSize[0], props.maxGridSize[1],
                props.maxGridSize[2]);
        printf("\tMax block dimensions %d x %d\n",
                props.maxThreadsDim[0], props.maxThreadsDim[1]);
        printf("\tMax threads per block %d\n",props.maxThreadsPerBlock);
   }
   return 0;
}
