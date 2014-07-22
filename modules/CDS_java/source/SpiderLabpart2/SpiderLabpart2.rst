********************
Improving the Spider
********************

First Question: How much work is there?
#######################################

Once you have a completed working spider, let’s examine how much
work it has to do.  Try some experiments in which you continue
using increasing values of maxUrls in the Spider.  Please note that
you can provide this value in its constructor.  Add a method to the
Spider that enables you to ask how many pages are still left to
work on in the ‘work’ queue.  You may also want to add a method to
know how many pages have been finished.

Change the RunSpider class to run some experiments with different
values of maxUrls by executing several Spiders.  For each value of
maxUrls, report on how much work is left to do.  How quickly is our
Spider overloaded with work?

Multiple Spiders to the rescue
##############################

Now let’s examine how we can use multiple spiders working at the
same time on this problem.  Your instructor will take a moment to
explain how we will use a technique called threads to run many
spiders at the same time, each of who will access the work,
finished, and urlCounter queue.  Then you will try this out below.

There is now a new lab.concurrentSpider package in our shared
space.  Examine the RunThreadedSpider class.  Note that we now use
a Java class called a Thread to begin running multiple instances of
the Spider in many Threads.  The Spider is now in a class called
ConcurrentSpider, and implements an interface called Runnable.

A key feature of concurrently running Spiders is that they must
share the same data structures in order to work together.  To do
this, we need to place the data structures they are working on in
one class and create one instance of that class in
RunConcurrentSpider.  Then each new ‘Runnable’ ConcurrentSpider
will receive a reference to that class of shared data structures.
We provide a class called lab.concurrentSpider.SharedSpiderData for this purpose.

First try:  share our original data structures?
###############################################

We could attempt to use the original LinkedList and ArrayList data structures and share those among the threads.  However, these are not 'thread safe', that is they are not guaranteed to behave properly when multiple threads are accessing and updating them at the same time.

Second try: concurrent data structures
######################################



To ensure our code will work correctly using multiple threads, we will
use the new Java Concurrent Data
Structures from the package java.util.concurrent.  Begin with the
file SharedSpiderData to see the types of shared, thread-safe data
structures we will use for this version of the multi-threaded
crawler.

To Do
**********

Finish the classes called ConcurrentSpider and RunThreadedSpider.  You will need to discover what methods on the concurrent data structures (ArrayBlockingQueue, ConcurrentLinkedQueue) are available for adding and removing elements.

.. topic:: Try This:

	* You can try using different numbers of threads, depending on how much your machine can handle.
	* Experiment with this variable found in ComcurrentSpider:  `maxurls`     If you double it, how many new urls were encountered?  Now that you have all these spider threads, you can likely scrape more URLs.
	* Experiment with the BEGNNING\_URL variable found in RunSpider by choosing some other pages of interest to you as starting points.
 


















