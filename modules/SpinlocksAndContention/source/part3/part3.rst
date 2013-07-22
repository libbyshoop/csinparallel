.. role:: uline

**************************
Getting More Sophisticated
**************************

The spinlock constructions we've looked at some far provide first-come-first-served (FIFO) access with little contention, making them useful in many applications. In `real-time systems`_, however, threads may require the ability to *abort*, canceling a pending request to acquire a lock in order to meet some other real-time deadline. Example reasons for giving up waiting for a lock could include a tiemout or a database transaction aborted by the user. Let's look at implementing the option to abort on the locks we've examined so far:

Aborting a :uline:`backoff lock` request is trivial: if the request takes too long, simply return from the ``lock()`` method. The lock's state is unaffected by any thread's abort. A nice property of this lock is that the abort itself is immediate: it is wait-free, requiring only a constant number of steps, and there is no cleanup.

The :uline:`queue lock` is a little trickier, though: if we simply quit, the next thread in line will starve. 

Under normal circumstances, recall, the queue lock works like this:

.. image:: images/normalqueue.gif
	:width: 750px
	:height: 316px
	:scale: 40%
	:alt: normal queue behavior.
	:align: center

However, if we try to have a node in the middle abort, the subsequent threads run into problems:

.. image:: images/abortqueue.gif
	:width: 750px
	:height: 316px
	:scale: 40%
	:alt: queue with an aborting node
	:align: center
	
Instead, we need a graceful way out. Removing a node in a wait-free manner from the middle of the list is difficult if we want that thread to reuse its own node, though. Instead, the aborting thread marks the node so that the successor in the list will reuse the abandoned node and wait for that node's predecessor. Let's go through it one step at a time:

.. image:: images/clhabort1.png
	:width: 597px
	:height: 367px
	:scale: 40%
	:alt: queue with an aborting node
	:align: center





.. _real-time systems: http://en.wikipedia.org/wiki/Real-time_computing