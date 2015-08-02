

****************************
Parallel Processes in Python
****************************

*Author*: 

| Steven Bogaerts
| Department of Computer Science
| DePauw University
| 602 S. College Ave, Greencastle, IN  46135, U.S.A.

For Instructors
###############

This module examines some key concepts of parallelism that have particularly accessible Python implementations, making it suitable for an introductory computer science course (CS1). It considers how to create processes to work in parallel, how to use locks to manage access to shared resources, and how to pass data between processes. This is done using the ``multiprocessing`` module in Python. 

*Note:* The code examples in this module conform to **Python 2**.

For Students
############

You are about to study some key ideas of parallel computation. More specifically, we'll build on the knowledge of Python you've already gained, to see how to create programs that can make use of special computer hardware to literally do multiple things at the same time. We'll learn how to do this in Python using the ``multiprocessing`` module, but the ideas we explore here are broadly applicable. Thus they will serve a nice foundation for more study in future courses.

Start at the beginning section linked in the Contents below and work your way through, trying examples and stopping to complete code that you are asked to work on along the way.  You should work through this tutorial by having your Python programming environment available as you read-- this is designed for you to learn by doing. 


Learning Goals
##############

Base Knowledge
--------------

Students should be able to:

-  Describe what a process is and how a running program can involve many
   processes.

-  Describe why order of execution of multiple idependent processes is not guaranteed.

-  Describe the challenge of shared resources in parallel computation.

-  Describe the purpose of interprocess communication.

Conceptual/Applied Knowledge
----------------------------

Students should be able to:

-  Write embarrassingly parallel programs in Python using the
   multiprocessing module.

-  Manage access to a shared resource via a lock.

-  Write programs involving simple interprocess communication via a
   shared queue.

Prerequisites and Assumed Level
###############################

At a minimum, students should have introductory-level experience in
printing, variables, tuples, and writing and using functions with
keyword arguments. Some basic understanding of object-oriented
programming is also helpful: classes, methods, object state,
constructors, and method invocation. For example, it would be quite
sufficient for students to be able to write a “Student” class with major
and GPA fields and supporting accessors and mutators.

Contents
########


.. toctree::
	:maxdepth: 2

	ProcessesBasics
	MultipleProcesses
	MultipleProcesses2
	ExecutionOrderAndResources
	LockingExercise
	Communication
	QueueExerciseHint
	QueueExerciseSolution
	Coordination
	Coordination2
	



.. comment
	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`

