/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* 
 * This copy of code is a derivative designed for educational purposes 
 * and it contains source code provided by NVIDIA Corporation.
 *
*/

#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

struct Sphere {

    float   r,b,g;// color of the sphere
    float   radius; 
    float   x,y,z;// coordinate of the center
    
    // will return the distance between imaginary camera and hit
    __device__ float hit( float ox, float oy, float *n ) {
        float dx = ox - x; // distance on x-axis
        float dy = oy - y; // distance on y-axis
        // if (dx*dx + dy*dy > radius*radius), ray will not hit sphere
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            // n is used in visual effect
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};

#define SPHERES 20

__global__ void kernel( Sphere *s, unsigned char *ptr ) {

    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // this is a linear offset into output buffer
    int offset = x + y * blockDim.x * gridDim.x;

    // shift the (x,y) image coordinate so that z-axis go through center
    float   ox = (x - DIM/2);
    float   oy = (y - DIM/2);

    float   r=0, g=0, b=0;// set the background to black
    float   maxz = -INF;
    for(int i=0; i<SPHERES; i++) {
        float   n;
        float   t = s[i].hit( ox, oy, &n ); // return the distance
        if (t > maxz) { 
            float fscale = n;// improve visual effect
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t; // update maxz everytime a smaller distance is found
        }
    } 

    // color the bitmap according to what the ray has 'seen'
    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
    Sphere          *s;
};

int main( void ) {

    // declare the data block and other needed variables
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char   *dev_bitmap;
    Sphere          *s;

    // allocate temp memory for the Sphere dataset on CPU
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );

    // initialize the Sphere dataset
    for (int i=0; i<SPHERES; i++) {
        temp_s[i].r = rnd( 1.0f );
        temp_s[i].g = rnd( 1.0f );
        temp_s[i].b = rnd( 1.0f );
        temp_s[i].x = rnd( 1000.0f ) - 500;
        temp_s[i].y = rnd( 1000.0f ) - 500;
        temp_s[i].z = rnd( 1000.0f ) - 500;
        temp_s[i].radius = rnd( 100.0f ) + 20;
    }

    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );

    // allocate memory for the Sphere dataset on GPU
    HANDLE_ERROR( cudaMalloc( (void**)&s, sizeof(Sphere) * SPHERES ) );

    // transfer the initialized Sphere dataset from CPU memory to GPU memory
    HANDLE_ERROR( cudaMemcpy( s, temp_s, sizeof(Sphere) * SPHERES,
                                cudaMemcpyHostToDevice ) );

    // generate a bitmap from our sphere data
    dim3    grids(DIM/32,DIM/32);
    dim3    threads(32,32);
    kernel<<<grids,threads>>>( s, dev_bitmap );

    // copy our bitmap back from the GPU for display
    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    // free CPU memory
    free( temp_s );

    // free GPU memory
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    HANDLE_ERROR( cudaFree( dev_bitmap ) );
    HANDLE_ERROR( cudaFree( s ) );

    // display
    bitmap.display_and_exit();
}

