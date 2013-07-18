Introduction
============

What is River Trail
-------------------

River Trail is an extension of JavaScript written by Intel and Mozilla to enable data-parallelism in web applications. It compiles units of code with the OpenCL kernel compiler to parallelize them. It also adds a specialized object, the ParallelArray object, to JavaScript.

What It Looks Like
----------------------
::
    var first = new ParallelArray(1,2,3,4);

    /* Elemental functions */
    var incd = first.map(function f(val) {return val+1;} );

    /* Collective functions */
    var summed = incd.reduce(function f(a, b) {return a + b;}); 

How It Works
------------

When ParallelArray object functions (such as map, reduce, scatter, scan, and filter) are called, an OpenCL kernel is executed and returned. OpenCL executes these kernels on as many devices as it is able to control, which may include multiple CPUs, vector cores, and GPUs. So, the computation is distributed to each core and each partition of the graphics card.

Benefits
--------

The River Trail extension increases speed (often significantly) by leveraging more of a computer's available resources. It is especially useful in graphics, a major computational area for JavaScript, where it allows much speed up for certain operations.

Also, as you will see in the next module, the River Trail extension is fairly easy for users to obtain. It is currently included in the Mozilla Firefox nightly build and will hopefully be standardized soon.
