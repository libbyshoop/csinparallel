**************************
Resource and Customization
**************************

Source Code
###########

:download:`Download GameOfLife.zip <GameOfLife.zip>`

:download:`Download StatKit.zip <StatKit.zip>`

**Please put the StatKit folder outside of GameOfLife folder.**

Prerequisite on Hardware and Software
#####################################

In order to compile and run the code, you will need the following hardware and software:

* Machines with multicore processors (Intel i7 or Xeon) 

* Cluster (LittleFe, Selkie by Macalester College, or Helios by St. Olaf College)

* Nvidia's Graphics Card (Tesla, Geforce, Quadro)

* OpenMP library for OpenMP operations

* OpenMPI library for MPI operations (Normally OpenMPI should be installed on clusters. However, you can install it on a stand alone machine as well. It would only simulate MPI operations through TCP/IP)

* CUDA library for CUDA operations. 

Customize Makefile
##################

The source code is compiled and tested on LittleFe and *Macalester College*'s Nuggle server. Therefore, most of the library path in the makefile is customized for these machines. Before compiling or running the code, you will need to customize the makefile for your own installation.

**C Compiler**

We use *gcc* and *icc* as standard compiler.

**OpenMP Macro**

We discovered that different C compilers has different OpenMP macros. *gcc* uses -fopenmp while *icc* uses -openmp. Make sure the macro you put in matches your C compiler.

**CUDA Library Path**

You will need to change the CUDA library path to match your own machine.

**machinefile**

MPI requires a machinefile for *mpirun* command. You will need to customize machinefile to match your own installation.

Notes
#####

1. LittleFe has Nvidia's Ion graphics card, which is a rather low end product. Running CUDA only version and CUDA-MPI hybrid version with display will cause problems because limited GPU resources. We recommend running these versions with **--no-display** option.

2. If you are running MPI on a single machine instead of on a cluster, MPI+OpenMP hybrid version can be really slow due to MPI and OpenMP are using multicore architecture in the same time.




