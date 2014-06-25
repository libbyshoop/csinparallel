=======================================
Task Decomposition Algorithm Strategies
=======================================

All threaded programs have some form of task decomposition, that is, delineating which threads will do what tasks in parallel at certain points in the program. We have seen one way of dictating this by using the master-worker implementation, where one thread does one task and all the others to another.  Here we introduce a more general approach that can be used.

14. Task Decomposition Algorithm Strategy using OpenMP section directive
************************************************************************

*file: openMP/14.sections/sections.c*

*Build inside 14.sections directory:*
::
	make sections

*Execute on the command line inside 14.sections directory:*
::
	./sections

This example shows how to create a program with arbitrary separate tasks that run concurrently.  This is useful if you have tasks that are not dependent on one another.

.. literalinclude::
	../patternlets/openMP/14.sections/sections.c
    :language: c
    :linenos:



