********************
Improving the Spider
********************

First Question: How much work is there?
#######################################

Once you have a completed working Spider, let’s examine how much
work it has to do.  Try some experiments in which you continue
using increasing values of maxUrls in the Spider.  Please note that
you can provide this value in its constructor.  

**TO DO:**  Add a method to the
Spider that enables you to ask how many pages are still left to
work on in the ‘work’ queue.  You may also want to add a method to
know how many pages have been finished.

**TO DO:**  Change the RunSpider class to run some experiments with different
values of maxUrls by executing several Spiders, one right after the other,
with increasing numbers of URLs.  For each value of
maxUrls, report on how much work is left to do.  How quickly is our
single Spider overloaded with work?

Multiple Spiders to the rescue
##############################

Now let’s examine how we can use multiple spiders working at the
same time on this problem.  Your instructor will take a moment to
explain how we will use a technique called threads to run many
spiders at the same time, each of who will access the work,
finished, and urlCounter queue.  Then you will try this out.

**Note:** more details about the implementation can be found on the
next page, which you can get to by following 'next' in the upper and lower
right of this page or going to the 'Next Topic' in the menu on the left.

There is a new lab.concurrentSpider package included in the code.
Examine the **RunThreadedSpider** class.  Note that we now use
a Java class called a Thread to begin running multiple instances of
the **ConcurrentSpider**, one each per Thread that is started.  
The Spider is now in this class called
ConcurrentSpider, and implements an interface called Runnable.

A *key feature* of concurrently running Spiders is that they must
share the same data structures in order to work together.  To do
this, we need to place the data structures they are working on in
one class and create one instance of that class in
RunConcurrentSpider.  Then each new ‘Runnable’ ConcurrentSpider
will receive a reference to that class of shared data structures.
We provide a class called **SharedSpiderData** for this purpose.

Can we share our original data structures?
###############################################

We could attempt to use the original LinkedList and ArrayList data structures 
and share those among the threads.  However, these are not 'thread safe', 
that is they are not guaranteed to behave properly when multiple threads are 
accessing and updating them at the same time.

Guaranteed implementation: Use concurrent data structures
##########################################################

To ensure our code will work correctly using multiple threads, we will
use the new Java Concurrent Data
Structures from the package java.util.concurrent.  Begin with the
file SharedSpiderData to see the types of shared, thread-safe data
structures we will use for this version of the multi-threaded
crawler.

To Do
**********

Finish the class called ConcurrentSpider so that it uses the new concurrent data structures when scraping
the pages and keeping track of what has finished.  You will need to discover what methods on the concurrent data structures (ArrayBlockingQueue, ConcurrentLinkedQueue) are available for adding and removing elements.

You will also find it useful to include the ability to have the RunThreadedSpider class be able to determine how much
overall work was completed.

.. topic:: Try This:

	* You can try using different numbers of threads, depending on how much your machine can handle, by changing the NUM_THREADS variable in the RunThreadedSpider class.
	* Experiment with the parameter found in ComcurrentSpider constructor:  `maxUrls`     If you double it, how many new urls were encountered?  Now that you have all these spider threads, you can likely scrape more URLs.
	* Experiment with the BEGNNING\_URL variable found in RunSpider by choosing some other pages of interest to you as starting points.
 


















