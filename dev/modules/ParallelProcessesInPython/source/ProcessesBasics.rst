********************************
Basics of Processes with Python
********************************


Think about a time you worked with other people on some task. For
example, you might have worked with some friends to shovel a driveway or
complete a class project. You split the task into pieces, and each
person worked at the same time to get the job done more quickly than
would be possible by yourself. This is parallelism. In computing,
*parallelism* can be defined as the use of multiple processing units
working together to complete some task. There are many different kinds
of hardware that can serve as a “processing unit”, but the principle is
the same: a task is broken into pieces in some way, and the processing
units cooperate on those pieces to get the task done.

Every computing device has a central processing unit (CPU) that handles
the running of a program. Have you heard of desktop computers or mobile
devices being described as “dual-core” or “quad-core”? This is a
reference to the number of processing units available on the CPU of that
device. A computer with a dual-core CPU has two cores – two processing
units – capable of working at the same time. Similarly, a quad-core CPU
has four cores.

The challenge is that these cores don’t get used to their greatest
benefit automatically. Programs need to be written in a particular way
to make effective use of the available cores. In this course module,
we’ll explore the use of the multiprocessing module in Python to write
programs that can execute on multiple cores.

Before we dive into programming, let’s consider what a *process* is.
While the details can be rather complex and dependent on many factors,
the big picture is simple. We can think of a process as a running
program. A process has to keep track of what line in the code will be
executed next, and what variable values are set. On a single-core
processor, only one process actually runs at a time. This is in contrast
to a multicore processor, in which multiple processes can be executed
literally at the same time (limited by the number of cores, of course).

Making a Process
----------------

We are now ready to work with the ``multiprocessing`` module itself. Let’s
consider the code below. First note that we
use ``from multiprocessing import *`` to gain access to the multiprocessing
module. This will give us access to many useful tools, including the
``current_process`` function and the ``Process`` class used in this
example.

::

    from multiprocessing import *

    def sayHi():
        print "Hi from process", current_process().pid

    def procEx():
        print "Hi from process", current_process().pid, "(parent process)"

        otherProc = Process(target=sayHi, args=())

        otherProc.start()


This code follows a common pattern: a *parent* process creates one or
more *child* processes to do some task. In this example, suppose we call
``procEx``. The first line in that function prints a simple message about
what process is running. This is done by calling the function
``current_process`` that is defined in the multiprocessing module. The
``current_process`` function returns a Process object representing the
currently running process. Every Process object has a public field
called **pid**, which stands for “process identifier”. Thus
``current_process().pid`` returns the pid for the currently running
process. This is what gets printed.

Proceeding to the next line of the ``procEx`` function, observe that the
``Process`` constructor is called, passing two arguments by name. The
purpose of this constructor call is to create a new ``Process`` object to be
executed. The ``target`` argument specifies what function should be executed
when the process under construction is actually started. The ``args``
argument is a tuple of the arguments to pass to the target function; since the ``sayHi`` target function
takes no arguments, args is an empty tuple in this example.

It is important to note that by calling the ``Process`` constructor, we have
*created* a ``Process`` object, but we have not yet *started* a new process.
That is, the process exists, but is not available to be run yet. The
process is actually started with the last line of ``procEx``. The ``start``
method is defined in the ``Process`` class. It changes the state of the
``Process`` object on which it is called, such that the process is made
available for execution.

So to summarize, there are two steps to make a child process do some
task: A ``Process`` object must be created using the constructor, and then
started using the ``start`` method.

So what does the child process do? It executes the ``sayHi`` function, as
specified in the target argument of the ``Process`` constructor call. Thus
it simply prints a message showing its pid. Note we use the same
``current_process().pid`` code here as in the parent, but this code will be
executed by the child process, not the parent, and so the pid will be
different. If you call the ``procEx`` method, you should receive output
similar to the following:

::

        Hi from process 3988 (parent process)
        Hi from process 4828

Of course, your pids will likely be different, since these numbers are
arbitrarily assigned by the operating system.

.. topic:: Try the code

	:download:`Download basic1.py <code/basic1.py>` and try the above example on your system.

Making Multiple Processes
-------------------------

Let’s extend what we’ve just looked at a little bit with a short
exercise. Copy the code from the previous example and modify it
to create three processes, each of which says “hi”. Try this on your own
now, before reading on.
