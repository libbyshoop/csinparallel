.. Cuda Vector Addition documentation master file, created by
   sphinx-quickstart on Mon Jun  2 15:58:17 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Timing CUDA Operations
================================================

This module was created by Joel Adams of Calvin College and extended and adapted
for CSInParallel by Jeffrey Lyman in 2014 (JLyman@macalester.edu) 

The purpose of this document is to teach students the basics of 
CUDA programming and to give them an understanding of when it is
appropriate to offload work to the GPU.

Through completion of Vector Addition, multipliction, square root,
and squaring programs, students will gain an understanding of 
when the overhead of creating threads and copying memory is worth
the speedup of GPU coding.

**Prerequisites**

- Some knowledge of C coding and using makefiles.

- An ability to create directories and use the command line in unix.

- Access to a computer with a reasonably capable GPU card.

**Contents**

This activity contains three parts, linked below.  First there is a short introduction to setting up code in CUDA to run on a GPU. Then you will try running vector addition code on your GPU machine. Lastly, you will experiment with various types of operations and large sizes of arrays to determine when it is worthwhile to use a GPU for general-purpose computing.

.. toctree::
    :maxdepth: 1

    0-Introduction/Introduction 
    1-VectorAdd/VectorAdd
    2-MoreExercises/MoreExercises


.. comment
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

