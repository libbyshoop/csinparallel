.. CDS_java documentation master file, created by
   sphinx-quickstart on Fri Jul 13 11:33:43 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Concurrent Data Structures in Java
====================================

**Prologue**

This lab activity asks you to complete some code provided so that it is able to crawl the web from a given starting URL.  You will start by completing a sequential version of this code.  Then you will work on a version that uses the libraries from java.util.concurrent to create a multi-threaded version.

**Prerequisites**

Some existing knowledge of the Java data structures LinkedList and ArrayList is necessary.

**Code**

Please download :download:`ConcurrentDataStructures.jar <code/ConcurrentDataStructures.jar>`. Inside the src directory are two packages: one for the original sequential web crawler called lab/spider, and another for the threaded spider called lab/concurrentSpider. The first chapter linked below starts you out with the sequential spider using conventional Java data structures to hold the URLs it encounters. You will then continue on to create the threaded version, starting with this code provided.

.. toctree::
   :maxdepth: 1

   TheSpiderLabonecrawler/TheSpiderLabonecrawler
   URLSpider/URLSpider
   SpiderLabpart2/SpiderLabpart2


.. comment
	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`

