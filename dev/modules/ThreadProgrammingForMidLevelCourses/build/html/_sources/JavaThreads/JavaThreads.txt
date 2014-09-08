************
Java Threads
************

We can use the Java API to parallelize programs. We will focus on a class for writing multi-threaded programs called ``java.lang.Thread``, which deals with individual threads of execution running concurrently.

After we create a ``Thread`` object, it will start execution when we call its ``start()`` method. Calling this method marks the ``Thread`` object ready, and it returns so that the caller can continue execution. This ``Thread`` that we just started will execute its ``run()`` method so that it runs concurrently to the caller. The caller (or possibly some other piece of code) could wait for the ``Thread`` to finish by calling the ``Thread``'s ``join()`` method, which will block the caller until the ``Thread``'s ``run()`` method returns.

A ``Thread`` object's default ``run()`` method will do nothing and return immediately, so we need to specify what a ``Thread`` will do. We could do this by creating a class extending ``Thread`` and then override ``run()``. Alternatively, we could create a class implementing the ``Runnable`` interface, which contains a single method ``run()``. Then we can pass a ``Runnable`` object to the constructor of ``Thread``, which will cause that ``Thread``'s ``run()`` method to invoke the ``run()`` method of the ``Runnable`` object. This lab will use the second approach.

We can use ``Thread`` objects to easily create a program with multiple threads, but this comes with some complications. The different ``Threads`` will usually need to communicate or share data in order to accomplish a shared goal, which may seem simple because they share and address space and they can share variables. However, we need to make sure to avoid race conditions where one thread accesses data in the middle of the process of another thread changing that data. Java provides a simple way to avoid this situation. The keyword ``synchronized`` creates a code block that locks at the beginning of a block, runs the code in the block, and then releases the lock. For example:

.. code-block:: java
	
	synchronized(obj) {
	   ...
	}

acquires the lock attached to ``obj``, runs the block, and then releases the lock. We can use an object from any class. If our thread tries to acquire a lock that a different thread already holds, our thread will block until the other thread releases the lock. We can also use ``synchronized`` to modify a method signature, as in:

.. code-block:: java
	
	public synchronized void method() {
	   ...
	}

which would work the same as putting the method body inside of a ``synchronized`` block which locks on this. The class ``java.util.concurrent`` has some more general synchronization primitives, such as semaphores, but this module will not use these.
