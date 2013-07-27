***********************************************
Finding the Area Under the Curve Using Striping
***********************************************

As stated before blocking may not always be desirable. In case of the problem of finding the area under the curve which way of data separation we choose doesn’t have an impact on the performance of our program. In some other situations, however, it is possible that we encounter a non-uniform distribution of data. In this case we might employ striping rather than blocking. Striping can be done for the nodes only, for the threads only or both. The main difference in striping and blocking is that this time the nodes do not require a subset of their own rectangles. Instead, the loop will go through all the rectangles.

In the following activities you will be asked to improve the get_rectangle_area(struct area_t * area) function. For each activity it is recommended that you create separate versions of the function. 

Before you write your functions, there are several things you need to do. These instructions are valid only for MPI striping, and any form of hybrid srtiping. For OpenMP only striping, do not change the way you compile the program and do not add any additional header files. 

Make sure you download the MPEarea.h file and add it to the folder where the rest of your code is. 
Then add the following lines to the init(struct area_t * area) function between setup_rectangles_improved(area); and sum_rectangles(area); in area.h:

.. literalinclude:: area.h
	:language: c
	:lines: 94-123

Some notes about compiling your improved versions. When compiling make sure you specify make [target] STRIPING=1. This is done so that you don’t have to remove all functions and header files related to X windowing. In the activities descriptions there are further instructions about compiling.
In addition, you will be needing some guidelines to call the proper drawing function in your new version. Just like we were calling draw_rectangle(area,current_left, current_height) you will be calling a similar function in each iteration. To draw each rectangle you need to call a function from the MPEarea.h file called draw_rectangle_mpe(struct area_t * area, double current_left, double current_height, int color_index). The color_index is an integer that specifies an index from a color array. This index varies depending on whether you use OpenMP, MPI or both. 
If you use OpenMP or hybrid striping specify a variable in your function such as:

.. literalinclude:: area.h
	:language: c 
	:lines: 179

and pass this variable to the draw_rectangle_mpe function.
If you use MPI only then just pass 0 into the draw_rectangle_mpe function, the color assignment for each process is already taken care of for you.
