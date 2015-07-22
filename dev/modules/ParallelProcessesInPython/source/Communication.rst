Communication
=============

In most real-world applications of parallelism, some amount of
communication between processes is required. We have already seen one
way in which processes can communicate: a parent process can send data
to children through the args parameter of the Process constructor. Now
we are ready to look at a more flexible means of communication.

The Queue class (pronounced like the letter “Q”) defines a Queue object
that a parent can pass to children so that multiple processes have
access to it. A queue can be thought of as a collection of data. Any
process can put data onto the queue using the put method, and take data
off the queue using the get method. Thus one process could do a put, and
another a get, in order to transmit data. If a process attempts a get
when there is nothing on the queue, then the process will wait (*block*)
on the line of code where the get occurred until some other process does
a put on the queue.

Let’s look at this in the following example:

::

    import time

    def greet(q):
        print "(child process) Waiting for name..."
        name = q.get()
        print "(child process) Well, hi", name

    def sendName():
        q = Queue()
        
        p1 = Process(target=greet, args=(q,))
        p1.start()
        
        time.sleep(5) # wait for 5 seconds
        print "(parent process) Ok, I'll send the name"
        q.put("Jimmy")



Note the
use of a queue for communication between processes, which in this case is the variable ``q``, which is a Python multiprocessing  ``Queue`` object. When ``sendName`` is
run, the following output results:

::

        (child process) Waiting for name...
        (parent process) Ok, I'll send the name
        (child process) Well, hi Jimmy

At the start of the sendName function, the Queue constructor is called,
with the resulting Queue object stored in the variable q. This object is
passed to the child process. So the child process is in fact using the
same queue as the parent. The child is started, and then the parent does
nothing for 5 seconds, via the time.sleep(5) command. In the meantime,
since the child has started, the first print in greet is executed,
followed by the call to get. The child’s get is a *blocking* command.
This means that the child process will go to sleep until it has a reason
to wake up – in this case, that there is something to get off the queue.
Since the parent sleeps for 5 seconds, the child ends up blocking for
approximately 5 seconds as well. Finally the parent process sends the
string “Jimmy”, the child process unblocks and stores “Jimmy” in the
variable name, and prints its final message.

.. topic:: Try the code

    :download:`Download sendName.py <code/sendName.py>` and try the above example on your system.


Extended Communication Via a Queue 
-----------------------------------

Let’s try some quick practice now that you’ve worked through the
previous example. Copy the code above as a basis for ``greet2`` and
``sendName2``. Modify the code so that ``greet2`` expects to receive 5 names,
which are sent by ``sendName2``. Each function should accomplish this by
sending/receiving one name at a time, in a loop. Spend some time on this
before moving on.

