*****************
Program Structure
*****************

:download:`Download Pandemic-All.zip <Pandemic-All.zip>`

The overall structure of the program is pretty complicated. This is mainly because we can build 6 versions of the program at the same time. The six versions are:

* Serial Version
* OpenMP Version
* MPI version
* CUDA version
* MPI + OpenMP heterogeneous (hybrid) version
* MPI + CUDA heterogeneous (hybrid) version

There are in total 8 files in this program.

+--------------+------------------------------------------+-------------------+
| File Name    | Functions                                |Usage              |
+==============+==========================================+===================+
|  Pandemic.c  | Holds All the function calls             |All versions       |
+--------------+------------------------------------------+-------------------+
|  Defaults.h  | Data structure and default values        |All versions       |
+--------------+------------------------------------------+-------------------+
| Initialize.h | Initialize the runtime environment       |All versions       |
+--------------+------------------------------------------+-------------------+
| Infection.h  | Find and share all infected persons      |All versions       |
+--------------+------------------------------------------+-------------------+
|   Display.h  | Display everyone's state and location    |If use display     |
+--------------+------------------------------------------+-------------------+
|   CUDA.cu    | Use CUDA for core operations             |2 CUDA versions    |
+--------------+------------------------------------------+-------------------+
|    Core.h    | Use serial or OpenMP for core operations |non-CUDA versions  |
+--------------+------------------------------------------+-------------------+
|  Finalize.h  | Finalize the run time environment        |All versions       |
+--------------+------------------------------------------+-------------------+

Program Structure
#################

.. figure:: Structure.png
    :align: center
    :width: 488px
    :height: 600px
    :alt: alternate text
    :figclass: align-center

    Overall Program Structurer

The rest of the module will go through each of the code files. We can start with the *Pandemic.c* file.

Pandemic.c
##########

At the very beginning of the file, we need to include all the necessary code files. This program is designed to build six versions of the program, and some versions need some specific code file while others don't. This means that the code file inclusion procedure needs to be handled delicately. A good reference of which file is included in which version is the **Usage** column of the file table in previous section.

We first include four files that are needed for all versions.

.. literalinclude:: Pandemic.c	
    :language: c
    :lines: 20-23

Then, if we are using display, we include the display code file.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 25-27

If we are not using CUDA, we include core functions file.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 29-31

If we are using CUDA, we include CUDA functions file.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 33-35

main()
*******

This function is the backbone of the whole program. It first initialize all the data structures need.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 39-46

Then it will initialize the runtime environment by calling **init()** function.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 48-50

Then we start the simulation. A for loop wraps around most of the functions, where the each iteration of the loop represents a day passing.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 54-56, 97

Inside the for loop, we first find all data related to the infection.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 57-60

Then, if display is enabled, we display the infection status. In other words, we display everyone's location and their states of infection.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 63-73

After display, if we are using CUDA for core operations, we can call **cuda_run()** function

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 77-81

else, if we are **NOT** using CUDA, we can call four core functions in *Core.h** code file.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 84-94

This is the end of the loop.

Finally, after the loop, we can display the results and finalize the runtime environment.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 99-103