*******************************
Ray Tracing and Constant Memory
*******************************

Acknowledgement
###############

The examples used in this chapter are based on examples in `CUDA BY EXAMPLE: An Introduction to General-Purpose GPU Programming`_, written by Jason Sanders and Edward Kandrot, and published by Addison Wesley.

Copyright 1993-2010 NVIDIA Corporation.  All rights reserved. 

This copy of code is a derivative based on the original code and designed for educational purposes only. It contains source code provided by `NVIDIA Corporation`_.

.. _`CUDA BY EXAMPLE: An Introduction to General-Purpose GPU Programming`: http://developer.nvidia.com/content/cuda-example-introduction-general-purpose-gpu-programming-0

.. _NVIDIA Corporation: http://www.nvidia.com

Basics of Ray Tracing
#####################

First of all, what is ray tracing. Well, ray tracing is how you reflect a scene consisting three-dimensional objects on a two dimensional image. This is similar to the games you play on your computer, except your games might use a different method. However, the basic idea behind is the same.

How does ray tracing work? It is actually pretty simple. In the two-dimensional image, you place a imaginary camera in there. Just like most real cameras, this imaginary camera contains light sensor as well. To produce a image, all we have to do is determine what light would hit our camera. The camera, on the other hand, would automatically record the color and light intensity of the ray hit it and produce exact same color and light intensity on the corresponding pixel.

Furthermore, deciding which ray would hit the camera is painstaking. So our clever computer scientist came up with an idea. Rather than deciding which ray would hit our camera, we can imagine shooting out a ray from our camera into the scene consisting three-dimensional objects. In other words, our imaginary camera is acting as an eye and we are now trying to find out what the eye is looking at. To seen what the eye is seeing, all we need to do is trace the ray shot out from the camera until it hits an object in our three-dimensional scene. We then record the color of the object and assign the color to the pixel. As you can see, most of the work in ray tracing is just deciding how the rays shot out and the objects in the scene would interact. 

Notes for Compile
#################

Before this chapter, we use the following code to compile CUDA code. 

**> nvcc -o example_name example_name.cu**

However, since we are using CUDA to produce images in this chapter, we need to use different code for compiling. Shown as follow

**> nvcc -lglut -o example_name example_name.c**


Ray Tracing Without Constant Memory
###################################

In our example, we will create a scene with 20 random spheres. They are placed in a cube with dimension 1000 x 1000 x 1000. The center of the cube is at the origin. All the spheres are random in size, position as well as color. We then place the camera on a random place on z-axis and fix it facing origin. Later on, all we need to do is to fire a ray from each pixel and keep tracing it until it hits one of the objects. We also need to keep track of the depth of the ray. Since one ray can hit more than one objects, we only need to record the nearest object and its color.

Ray Tracing Without Constant Memory source file:
:download:`ray_noconst.cu <ray_noconst.cu>`

Structure Code
**************

We first create a data structure Sphere. Just like standard C, you can also create data structures in CUDA C.

.. literalinclude:: ray_noconst.cu	
    :language: c
    :lines: 31-50

Inside the data structure, we stores the coordinate of the center of the Sphere as *(x, y, z)* and its color as *(r, g, b)*. You can see we also defined a method called hit. This method will decide whether the ray shot out from point *(ox, oy)* can hit the Sphere defined in the structure or not. The basic idea is simple, you can think of we project the sphere on our two-dimensional image. We first find out the distance between center of the Sphere and point *(ox, oy)* on the x-axis. We then do the same thing on the y-axis. Using Pythagorean theorem, we can find out the distance between center of the sphere and point *(ox, oy)*. If this distance is less than radius, then we are sure about the ray hitting the sphere. We then use this distance and the sphere's coordinate on z-axis to find out the distance between point *(ox, oy)* and sphere. On the other hand, it they don't intersect, we will assign negative infinity as the distance.

You may also noticed two other things left unexplained here. First, you can see that we add a qualifier **__device__** before the method definition. 

.. literalinclude:: ray_noconst.cu	
    :language: c
    :lines: 38

Well, the purpose of this qualifier is to tell the kernel that this method should executes on the device (our GPU) instead of on the host (our CPU). 

Second, you may also find the following line intrigued. 

.. literalinclude:: ray_noconst.cu	
    :language: c
    :lines: 45

The value *n* is used to provide a better visual effect. You can see that we defined it as the percentage of distance between point *(ox, oy)* and center of sphere out of the radius. We will add this value to later code so that you can see center of the circle clearer while the edge of the sphere dimmer.

Device Code
***********

.. literalinclude:: ray_noconst.cu	
    :language: c
    :lines: 54-85

On the GPU, we will assign each pixel a thread which is used for ray tracing computation. Therefore, in the first several lines of code, 

.. literalinclude:: ray_noconst.cu	
    :language: c
    :lines: 56-58

we first need to map each thread's threadIdx and blockIdx to the pixel position on the bitmap, which is represented by *(x, y)*. Then, we need to create a linear offset so that when the kernel is coloring the pixel, the kernel need to know exactly which pixel it will color.

Then we shift image coordinate by *DIM/2* on the x-axis and *DIM/2* on the y-axis as well. We need to do this because the center of the bitmap is not the origin. We need the center of the bitmap to match origin's position so that the z-axis can go through the center of image.

.. literalinclude:: ray_noconst.cu	
    :language: c
    :lines: 62-64

After the preparations, we can start our ray tracing program. We first set the *(r, g, b)* values for each pixel to be 0. We would have black background if the ray does not hit any object. Then we declare and initialize the variable *maxz*, which would hold the nearest distance between the pixel and one of the objects. Later on, each thread will call the method defined in the Sphere data structure. The method would use the *(ox, oy)* parameter passed by the thread to first decide whether one object will intersect the ray or not and second decide the distance if they intersect. The method will loop over all 20 spheres. 

.. literalinclude:: ray_noconst.cu	
    :language: c
    :lines: 72-75

In the several lines of code above, you can see that we assign the actual *(r, g, b)* value according to the *(r, g, b)* value in the structure. We also multiplied a constant *fscale* to it. When we see a sphere from above, the nearest point aligned with your eye and the sphere center will be closer to you. On the other hand, the edge of the sphere will appear to be a little bit far away. When we multiply *fscale* to the *(r, g, b)* values, what we are trying to do is to create this effect.

.. literalinclude:: ray_noconst.cu	
    :language: c
    :lines: 80-84

The last few line would be just color the the bitmap. Nothing needs to be clarified in these lines of code.

Host Code
*********

.. literalinclude:: ray_noconst.cu	
    :language: c
    :lines: 93-161

There is nothing worth mentioning about the host code. You first declare the data block and the variables. Then you allocate memory on both CPU and GPU for those variables. Then you can initialize some variables, the 20 spheres in this case on the CPU and then transfer them to the GPU memory. Later on you can call the kernel invocation code and let GPU finish the hard work. Finally, you transfer the bitmap back to CPU and display the bitmap.

.. figure:: RayTracing.png
    :width: 500px
    :align: center
    :height: 500px
    :alt: alternate text
    :figclass: align-center

    A screenshot from the ray tracing example

Performance
***********

We conducted 5 tests and the results are below.
 * 1. 6.7 ms
 * 2. 6.8 ms
 * 3. 6.8 ms
 * 4. 6.7 ms
 * 5. 6.8 ms
 * Average: 6.76 ms

Constant Memory
###############

We have mentioned that there are several types of memory in CUDA architecture. Till now, we have seen global memory and shared memory. This time, we will explore the characteristics of constant memory.

By its name, constant memory is designed to store variables that will not change when the kernel is executing commands. Constant memory is located in global memory, which means constant variables are stored in the global memory as well. However, constant variables are cached for higher access efficiency. Just like shared memory, there is always price come with faster access speed. The CUDA architecture provides only 64KB of space for global memory. Therefore, constant memory is not designed to store large dataset.

Ray Tracing With Constant Memory
################################

In the example of ray tracing, we will see how to improve program efficiency by using constant memory. We do this by store 20 sphere object in the constant memory for faster access. In our example, every pixel of the image needs to access 20 sphere objects over the course of kernel execution. If we have a bitmap of the size *1024x1024*, we are looking at over one million times of access for each of the sphere.

Ray Tracing With Constant Memory source file:
:download:`ray.cu <ray.cu>`

Constant Memory Declaration
***************************

.. literalinclude:: ray.cu	
    :language: c
    :lines: 54

This line of code shows you how to declare variables in constant memory. The only difference is that you have to add **__constant__** qualifier before the declaration.

Structure & Device Code
***********************

The device code and the code to create structure are exactly the same as the version not using constant memory.

Host Code
*********

Most of the host code is the same as the version not using constant memory. There are only two different places. First, since we have already prepared spaces in constant memory for Sphere dataset, we do not use the command *cudaMalloc()* and *cudaMemcpy()* anymore to allocate it in global memory anymore. Second, we use the following code to copy initialized Sphere dataset to the constant memory.

.. literalinclude:: ray.cu	
    :language: c
    :lines: 124-126

Performance
***********

We conducted 5 tests and the results are below.
 * 1. 6.2 ms
 * 2. 6.1 ms
 * 3. 6.3 ms
 * 4. 6.4 ms
 * 5. 6.4 ms
 * Average: 6.28 ms

Due to the small bitmap size we are using, the improvement is not significant.

