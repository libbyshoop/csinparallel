****************************************
Execution Order and Resource Contention
****************************************

In addition to a pid, we also have the option of naming each child
process. Any provided name is stored in the public name field defined in
the ``Process`` class. For example, consider the following code:


::

    def sayHi3(personName):
        print "Hi", personName, "from process", current_process().name, "- pid", current_process().pid

    def manyGreetings3():
        print "Hi from process", current_process().pid, "(parent process)"
        
        personName = "Jimmy"
        for i in range(10):
            Process(target=sayHi3, args=(personName,), name=str(i)).start()

If we run ``manyGreetings3``, the parent process
says “Hi”, and then creates and starts ten child processes. Each child
process runs ``sayHi3``, which requires a personName argument, specified in
the args parameter of the Process constructor call. We also include one
new argument in the Process constructor call: ``name``. This ``name`` argument should be a
string, and in this example we just use the string representation of the
loop index variable i. Thus when a child process executes ``sayHi3``, it has
access to the ``personName`` given as an argument, and also has access to
the ``name`` field provided in the call to the ``Process`` constructor.

Try to predict what will happen when you run the ``manyGreetings3`` function. Your first guess might be the following
(with arbitrary pids, of course):

::

        Hi from process 3988 (main process)
        Hi Jimmy from process 0 pid 5164
        Hi Jimmy from process 1 pid 5236
        Hi Jimmy from process 2 pid 6884
        Hi Jimmy from process 3 pid 3652
        Hi Jimmy from process 4 pid 1060
        Hi Jimmy from process 5 pid 1767
        Hi Jimmy from process 6 pid 5812
        Hi Jimmy from process 7 pid 4732
        Hi Jimmy from process 8 pid 3564
        Hi Jimmy from process 9 pid 4332

It’s possible that the processes will print out very nicely and in order
like the above, but it’s extremely unlikely. First note that each core
of the CPU is a *scarce resource*, meaning there aren’t typically enough
cores for every process to use one whenever it wants. On a quad-core
system, for example, up to four processes can execute at once. If there
are more than four processes wanting to execute, some will need to wait.

So the operating system maintains a list of waiting processes. When a
core becomes available, the operating system chooses another process to
execute on that core. But the process created first is not necessarily
the next one chosen. That is, just because we *create and start* the
processes in the order 0 through 9 in our program, it doesn’t mean that
the operating system will choose them to *execute* in the order 0
through 9. So, for example, we might expect output like this:

::

        Hi from process 3988 (parent process)
        Hi Jimmy from process 8 pid 3564
        Hi Jimmy from process 2 pid 6884
        Hi Jimmy from process 6 pid 5812
        Hi Jimmy from process 0 pid 5164
        Hi Jimmy from process 3 pid 3652
        Hi Jimmy from process 9 pid 4332
        Hi Jimmy from process 4 pid 1060
        Hi Jimmy from process 7 pid 4732
        Hi Jimmy from process 1 pid 5236
        Hi Jimmy from process 5 pid 1767

In fact, any ordering of the child processes’ messages is possible. The
only thing we know for certain is that the message from the parent
process will show up first, since our code specifies that that should
happen before we create any child processes. There are ways to ensure
that certain tasks are completed before other tasks, as we’ll see later
in this module. But by default, 
*child processes execute in arbitrary order*, and parallel programs must be designed with this in mind.

Unfortunately, we still haven’t captured what will likely happen when we
run the code given above. Go ahead and run it now and
see. Results will vary, but you may see something very garbled up like
the following:

::

    Hi from process 3988 (main process)
    Hi HJHimmyiHii   Jimmy  from process 0 -JfJ pid 5164
    immyrom process 4 - pid 4332
    immy from process  from process 7 - pid8  5236- pid
     3564
    Hi Jimmy from process 1 - pidH 6884
    i Jimmy from process 5 - pid 3652
    Hi Jimmy from process 5 - pid 1060
    Hi Jimmy from process 2 - pid 176
    Hi Jimmy from process 3 - pid 5812
    Hi Jimmy from process 9 - pid 4732

What’s going on? The first thing to realize is that the CPU cores are
not the only scarce resource here. Standard output – where printing
occurs – is also scarce. More specifically, standard output is a single
*shared* resource that multiple processes are trying to access at the
same time. So the processes have to take turns. As it is, our program is
not forcing the processes to take turns in any reasonable way. How can
we fix this? We’ll use a *lock*.

Using a Lock to Control Printing
--------------------------------

One excellent way to begin our study of locks is by analogy to a concept
from the novel by William Golding (or the 1963 and 1990 film
adaptations). The novel tells the story of a group of boys shipwrecked
on a deserted island with no adult survivors. Before an eventual
breakdown into savagery, the boys conduct regular meetings to decide on
issues facing the group. The boys quickly realize that, left unchecked,
such meetings will be unproductive as multiple boys wish to speak at the
same time. Thus a rule is developed: Only the boy that is holding a
specially-designated conch shell is allowed to speak. When that boy is
finished speaking, he relinquishes the conch so that another boy may
speak. Thus order is maintained at the meetings as long as the boys
abide by these rules. We can also imagine what would happen if this
conch were not used: chaos in meetings as the boys try to shout above
each other. (And in fact this does happen in the story.)

It requires only a slight stretch of the events in this novel to make an
analogy to the coordination of multiple processes accessing a shared
resource, like standard output. In programming terms, each boy is a
separate process, having his own things he wishes to say at the meeting.
But the air around the meeting is a shared resource - all boys speak
into the same space. So there is contention for the shared resource that
is this space. Control of this shared resource is handled via the
single, special conch shell. The conch shell is a *lock* – only one boy
may hold it at a time. When he releases it, indicating that he is done
speaking, some other boy may pick it up. Boys that are waiting to pick
up the conch are not allowed to say anything – they just have to wait
until whoever has the conch releases it. Of course, several boys may be
waiting for the conch at the same time, and only one of them will
actually get it next. So some boys might have to continue to wait
through multiple speakers.

The following code shows the analogous idea in Python. 

::

    def sayHi4(lock, name):
        lock.acquire()
        print "Hi", name, "from process", current_process().pid
        lock.release()

    def manyGreetings4():
        lock1 = Lock()
        
        print "Hi from process", current_process().pid, "(main process)"
        
        for i in range(10):
            Process(target=sayHi4, args=(lock1, "p"+str(i))).start()


At
the start of ``manyGreetings4``, the constructor of the ``Lock`` class is
called, with the resulting object stored in the variable ``lock1``. This
single ``Lock`` object, along with a distinct name, is passed to each of the child
processes. Each child process wants to print something when it executes ``sayHi4``. But
print writes to ``stdout`` (standard output), a single resource that is
shared among all the processes. So when multiple processes all want to
print at the same time, their output would be jumbled together were it
not for the lock, which ensures that only one process is able to execute
its print at a time.

How does the lock accomplish this? Through the use of the acquire and
release methods, both defined in the Lock class. Suppose process
:math:`A` acquires the lock and begins printing. If processes :math:`B`,
:math:`C`, and :math:`D` then execute their acquire calls while
:math:`A` has the lock, then :math:`B`, :math:`C`, and :math:`D` each
must wait. That is, each will *block* on its acquire call. Once
:math:`A` releases the lock, one of the processes blocked on that lock
acquisition will arbitrarily be chosen to acquire the lock and print.
That process will then release the lock so that another blocked process
can proceed, and so on.

Note that the lock must be created in the parent process and then passed
to each child – this way each child process is referring to the same
lock. The alternative, in which each child constructs its own lock,
would be analogous to each boy bringing his own conch to a meeting.
Clearly this wouldn’t work.

As in the previous example, the order of execution of the processes is
still arbitrary. That is, the acquisition of the lock is arbitrary, and
so subsequent runs of the code are likely to
produce different orderings. It is not necessarily the process that was
created first, or that has been waiting the longest, that gets to
acquire the lock next.

.. topic:: Try the code

	:download:`Download manyGreetings4.py <code/manyGreetings4.py>` and try the above example on your system.

You try it: Digging Holes
-------------------------

Let us now try an exercise extending the concept of locks above. Imagine
that you have 10 hole diggers, named :math:`A`, :math:`B`, :math:`C`,
:math:`D`, :math:`E`, :math:`F`, :math:`G`, :math:`H`, :math:`I`, and
:math:`J`. Think of each of these as a process, and write a function
assignDiggers() that creates 10 processes with these worker names
working on hole 0, 1, 2, ..., 9, respectively. Each one should print a
message about what it’s doing. When you’re done, you should get output
like the following (except that the order will be arbitrary):

::

        Hiddy-ho!  I'm worker G and today I have to dig hole 6
        Hiddy-ho!  I'm worker A and today I have to dig hole 0
        Hiddy-ho!  I'm worker C and today I have to dig hole 2
        Hiddy-ho!  I'm worker D and today I have to dig hole 3
        Hiddy-ho!  I'm worker F and today I have to dig hole 5
        Hiddy-ho!  I'm worker I and today I have to dig hole 8
        Hiddy-ho!  I'm worker H and today I have to dig hole 7
        Hiddy-ho!  I'm worker J and today I have to dig hole 9
        Hiddy-ho!  I'm worker B and today I have to dig hole 1
        Hiddy-ho!  I'm worker E and today I have to dig hole 4

Try to complete this exercise before moving on.

