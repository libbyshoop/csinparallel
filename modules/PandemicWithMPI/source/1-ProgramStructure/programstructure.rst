*****************
Program Structure
*****************

:download:`Download Pandemic-MPI.zip <Pandemic-MPI.zip>`

There are in total 7 files in this program.

+--------------+------------------------------------------+
| File Name    | Functions                                |
+==============+==========================================+
|  Pandemic.c  | Holds All the function calls             |
+--------------+------------------------------------------+
|  Defaults.h  | Data structure and default values        |
+--------------+------------------------------------------+
| Initialize.h | Initialize the runtime environment       |
+--------------+------------------------------------------+
| Infection.h  | Find and share all infected persons      |
+--------------+------------------------------------------+
|   Display.h  | Display everyone's state and location    |
+--------------+------------------------------------------+
|    Core.h    | Use serial or OpenMP for core operations |
+--------------+------------------------------------------+
|  Finalize.h  | Finalize the run time environment        |
+--------------+------------------------------------------+

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

At the very beginning of the file, We first include four files that are needed for all versions.

.. literalinclude:: Pandemic.c	
    :language: c
    :lines: 14-18

Then, if we are using display, we include the display code file.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 20-22

main()
*******

This function is the backbone of the whole program. It first initialize all the data structures need.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 26-32

Then it will initialize the runtime environment by calling **init()** function.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 34-36

Then we start the simulation. A for loop wraps around most of the functions, where the each iteration of the loop represents a day passing.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 40-42, 70

Inside the for loop, we first find all data related to the infection.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 43-47

Then, if display is enabled, we display the infection status. In other words, we display everyone's location and their states of infection.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 49-59

After display, we can call four core functions in *Core.h** code file.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 61-69

This is the end of the loop.

Finally, after the loop, we can display the results and finalize the runtime environment.

.. literalinclude:: Pandemic.c
    :language: c
    :lines: 72-76