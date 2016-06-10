.. Parallel Patternlets documentation master file, created by
   sphinx-quickstart on Sat Jan  5 09:59:20 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**********************
Parallel Patternlets
**********************

**Last Updated:** |date|

This document contains simple examples of basic elements that are combined to form patterns often used in programs employing parallelism.  We call these examples *patternlets* because they are deliberately trivial, small, yet functioning programs that illustrate a basic shell of how a particular parallel pattern is created in a program.  They are starting points you can use to create realistic working programs of your own that use the patterns.  Before diving into the examples, first there will be some background on parallel programming patterns.

.. toctree::
	:maxdepth: 1

	PatternsIntro
	MessagePassing/MPI_Patternlets
	SharedMemory/OpenMP_Patternlets



.. comment
	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`

.. |date| date::
