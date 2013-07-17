// Copyright 2003-2012 Intel Corporation. All Rights Reserved.
// 
// The source code contained or described herein and all documents related 
// to the source code ("Material") are owned by Intel Corporation or its
// suppliers or licensors.  Title to the Material remains with Intel Corporation
// or its suppliers and licensors.  The Material is protected by worldwide
// copyright and trade secret laws and treaty provisions.  No part of the
// Material may be used, copied, reproduced, modified, published, uploaded,
// posted, transmitted, distributed, or disclosed in any way without Intel's
// prior express written permission.
// 
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery
// of the Materials, either expressly, by implication, inducement, estoppel
// or otherwise.  Any license under such intellectual property rights must
// be express and approved by Intel in writing.



#include<stdio.h>
#include<stdlib.h>
#include"offload.h"
#include"time.h"
#include <sys/time.h>
#include<omp.h>

#define SIZE 1024*1024*64
#define PHI_DEV 0
#define ALPHA 1.3f
#define ITERS 10
#define NUM_THREADS 240

#define ALLOC alloc_if(1)
#define FREE free_if(1)
#define REUSE alloc_if(0)
#define RETAIN free_if(0)

#define VALIDATE

__declspec(target(mic)) static float *ina0, *inb0, *out0, *ina1, *inb1, *out1;

float *ref0,*ref1;

void do_async(void)
{
        //Send array 0 to the coprocessor
        #pragma offload target(mic:PHI_DEV)\
                in(ina0:length(SIZE) REUSE RETAIN)\
                in(inb0:length(SIZE) REUSE RETAIN)
        {
        }

        //Start the asynchronous transfer for array 1 and send signal "a1" when complete
        #pragma offload_transfer target(mic:PHI_DEV)\
                in(ina1:length(SIZE) REUSE RETAIN)\
                in(inb1:length(SIZE) REUSE RETAIN)\
                signal(&ina1)
        {
        }

        //Start simultaneous compute while the transfer for array 1 is occuring
        #pragma offload target(mic:PHI_DEV)\
                nocopy(ina0:length(SIZE) REUSE RETAIN)\
                nocopy(inb0:length(SIZE) REUSE RETAIN)\
                nocopy(out0:length(SIZE) REUSE RETAIN)
        {
                //Perform SAXPY operation
                #pragma omp parallel for firstprivate(ina0,inb0,out0)
                for(int i=0;i<SIZE;i++)
                {
                        out0[i]=ALPHA*ina0[i]+inb0[i];
                }
        }

        //Start the asynchronous transfer for out0
        #pragma offload_transfer target(mic:PHI_DEV)\
                out(out0:length(SIZE) REUSE RETAIN)\
                signal(&out0)
        {
        }

        //Start simultaneous compute while the transfer for out0 is occuring once you receive signal "array1" and transfer out1 back to host once done.
        #pragma offload target(mic:PHI_DEV)\
                nocopy(ina1:length(SIZE) REUSE RETAIN)\
                nocopy(inb1:length(SIZE) REUSE RETAIN)\
                out(out1:length(SIZE) REUSE RETAIN)\
                wait(&ina1)
        {
                //Perform SAXPY operation
                #pragma omp parallel for firstprivate(ina1,inb1,out1)
                for(int i=0;i<SIZE;i++)
                {
                        out1[i]=ALPHA*ina1[i]+inb1[i];
                }
        }
}

void do_offload(void)
{
	
	//Offload 1
	#pragma offload target(mic:PHI_DEV)\
		in(ina0:length(SIZE) REUSE RETAIN)\
                in(inb0:length(SIZE) REUSE RETAIN)\
                out(out0:length(SIZE) REUSE RETAIN)
	{
		//Perform SAXPY operation 
		#pragma omp parallel for firstprivate(ina0,inb0,out0)
		for(int i=0;i<SIZE;i++)
		{
			out0[i]=ALPHA*ina0[i]+inb0[i];
		}
	}

	//Offload 2
	#pragma offload target(mic:PHI_DEV)\
		in(ina1:length(SIZE) REUSE RETAIN)\
                in(inb1:length(SIZE) REUSE RETAIN)\
                out(out1:length(SIZE) REUSE RETAIN)
	{
		//Perform SAXPY operation
		#pragma omp parallel for firstprivate(ina1,inb1,out1)
                for(int i=0;i<SIZE;i++)
                {
                        out1[i]=ALPHA*ina1[i]+inb1[i];
                }
	}
}

int main()
{
	struct timeval start,end;
	double runtime;

	//Allocate arrays on host 
	ina0=(float*)malloc(sizeof(float)*SIZE);
	inb0=(float*)malloc(sizeof(float)*SIZE);
	ina1=(float*)malloc(sizeof(float)*SIZE);
	inb1=(float*)malloc(sizeof(float)*SIZE);
	out0=(float*)malloc(sizeof(float)*SIZE);
	out1=(float*)malloc(sizeof(float)*SIZE);
	ref0=(float*)malloc(sizeof(float)*SIZE);
	ref1=(float*)malloc(sizeof(float)*SIZE);	

	//Initialize host arrays
	srand(time(NULL));
	for(int i=0;i<SIZE;i++)
	{
		ina0[i]=(float)(rand()%100);
		inb0[i]=(float)(rand()%100);
                out0[i]=0.0f;;
                ina1[i]=(float)(rand()%100);
                inb1[i]=(float)(rand()%100);
                out1[i]=0.0f;	
		ref0[i]=ALPHA*ina0[i]+inb0[i];
		ref1[i]=ALPHA*ina1[i]+inb1[i];
	}

	//Allocate the arrays on coprocessor
	#pragma offload target(mic:PHI_DEV)\
		nocopy(ina0:length(SIZE) ALLOC RETAIN)\
		nocopy(inb0:length(SIZE) ALLOC RETAIN)\
		nocopy(out0:length(SIZE) ALLOC RETAIN)\
		nocopy(ina1:length(SIZE) ALLOC RETAIN)\
                nocopy(inb1:length(SIZE) ALLOC RETAIN)\
                nocopy(out1:length(SIZE) ALLOC RETAIN)
	{
		//Set the number of threads on the coprocessor
		omp_set_num_threads(NUM_THREADS);
	}


	gettimeofday(&start,NULL);
	for(int i=0;i<ITERS;i++)
	{
	do_offload();
	}
	gettimeofday(&end, NULL);
        runtime=(float) (end.tv_sec - start.tv_sec)+ (float) (end.tv_usec - start.tv_usec) * 1e-6;
        printf("Offload Runtime: %.3f\n",runtime/ITERS);
	
	//Validate the output
	#ifdef VALIDATE
	for(int i=0;i<SIZE;i++)
	{
		if(abs(ref0[i]-out0[i])>1e-6)
		{
			printf("ERROR:%d\n",i);break;
		}
	}
	#endif

	gettimeofday(&start,NULL);
        for(int i=0;i<ITERS;i++)
        {
        do_async();
        }
        gettimeofday(&end, NULL);
        runtime=(float) (end.tv_sec - start.tv_sec)+ (float) (end.tv_usec - start.tv_usec) * 1e-6;
        printf("Async Runtime: %.3f\n",runtime/ITERS);

	//Validate the output
	#ifdef VALIDATE
	for(int i=0;i<SIZE;i++)
        {
                if(abs(ref0[i]-out0[i])>1e-6)
                {
                        printf("ERROR:%d\n",i);break;
                }
        }
	#endif

	//Free the arrays on the coprocessor
	#pragma offload target(mic:PHI_DEV)\
                nocopy(ina0:length(SIZE) REUSE FREE)\
                nocopy(inb0:length(SIZE) REUSE FREE)\
                nocopy(out0:length(SIZE) REUSE FREE)\
                nocopy(ina1:length(SIZE) REUSE FREE)\
                nocopy(inb1:length(SIZE) REUSE FREE)\
                nocopy(out1:length(SIZE) REUSE FREE)
        {
        }

	return 0;
}
