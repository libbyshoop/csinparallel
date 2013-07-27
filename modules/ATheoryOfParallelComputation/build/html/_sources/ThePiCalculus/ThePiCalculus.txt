************************************
The pi-calculus, informally
************************************

* A *calculus* is a method or computation based on symbolic manipulation.

  * In *differential calculus*, symbolic manipulations involve an operator :math:`\frac{d}{dx}` that satisfies rules such as the following:

  .. math::

     {d\over dx}(f+g) = ({d\over dx}f) + ({d\over dx}g)\\
     {d\over dx}(f\cdot g) = ({d\over dx}f) \cdot g + f \cdot ({d\over dx} g)

  * In *integral calculus*, symbolic manipulations involve an operator :math:`\int ...\,dx` that satisfies rules such as the following:

  .. math::

     \int f+g\,dx = \int f\,dx + \int g\,dx \\
     \int f\cdot ({d\over dx}g)\,dx = f\cdot g - \int ({d\over dx}f)\cdot g\,dx

  * In the :math:`\lambda`-calculus, symbolic manipulations involve an operator :math:`\lambda` that has its manipulation rules, involving operations such as substitution of variables and applying functions to particular "input" values (function calls).

* The operators and manipulation rules for a calculus may have useful concrete applications. For example, the differential calculus rules are satisfied by certain continuous mathematical functions, where the operator :math:`\frac{d}{dx}` represents the rate of change of those functions.

  We typically think of :math:`\frac{d}{dx}` as operating on those functions, although the differential calculus rules are actually abstract and might be applied to other entities than functions.

* The :math:`\pi`-calculus has six operators. We think of them as operating on *sequential processes*, i.e., running computer programs, although they are abstract and can be used without any particular concrete application.

  * The *concurrency operator* :math:`P|Q` (pronounced *"P par Q"*) may be thought of as two processes *P* and *Q* executing in parallel (e.g., simultaneously on separate cores or on different computers).

  * The *communication operators* may be thought of as sending and receiving messages from one process to another, across a communication channel that is used only by those two processes (i.e., a *dedicated* communication channel in the language of CS).

    * The output prefixing operator :math:`\bar{c} \langle x \rangle . P` (pronounced "output x along c (then proceed with P)") may be thought of as send a message *x* across a channel *c*, then proceeding to carry out process *P*. Here, the channel c may be thought of as starting from this process to another.

      Channels such as c may be set up between any two processes, but those two processes are then uniquely determined for c, and may not be changed later. Channels provide for a single communication in one direction only, specified when the channel is created.

      The "dot" that appears in this notation indicates the boundary between one step and a next step in a process.

    * The *input prefixing* operator :math:`c(y).P` (pronounced "Input y along c") may be thought of as waiting to receive a value from the channel *c*, and once a value is received, storing that value in *y* and proceeding to carry out process *P*.

  * The replication operator :math:`!P` ("bang P") may be thought of as creating a new process that is a duplicate of *P*.

    This sort of an operation is quite realistic in parallel computing. For example, a *web server* is a program that receives requests for particular web pages and responds by sending those web pages. Web servers must be capable of handling multiple responses at the same time, because some web pages may take a significant amount of time to prepare and deliver, and it would be undesirable for one user to be delayed by another user's request. Therefore, a web server system may start up a new duplicate process for handling each request it receives. (Students who have studied operating systems will also see an analogy between the system call fork() and this replication operator.)

    In the :math:`\pi`-calculus, arbitrarily many duplicate processes are created by a single application of the replication operator.

  * The name allocation operator :math:`(\nu{\it c}).{P}` ("new c in P") may be thought of as allocating a new constant communication channel *c* within the process *P*. The symbol :math:`\nu` is the Greek letter nu, pronounced like "new".

  * The alternative operator :math:`P+Q` ("P plus Q") represents a process capable of taking part in exactly one alternative for communication. That process cannot make the choice among its alternatives; that selection among alternatives cannot be determined until it occurs, and once determined, any remaining alternatives have lost their chance and will never occur. (These restrictions on the alternative operator are not strictly necessary for :math:`\pi`-calculus to work, but they simplify the theory.)

* Besides these operations, there is one constant process 0 that does nothing. For example, we might write :math:`\bar{c} \langle x \rangle . 0` for a process that sends one message across a channel *c*, then does nothing more.

###########################
Pi Calculus and Parallelism
###########################


* Observe that all of the operations have to do with entire processes or with communication among processes. For example, there are no arithmetic operations such as multiplication, nor any operations related to applying (i.e., calling) functions, nor a way to store values in memory (assignment). The :math:`\pi`-calculus is entirely concerned with communication among processes that are executing in parallel.

  However, a theory of sequential processes, such as automata or the :math:`\lambda`-calculus, can be used in conjunction with :math:`\pi`-calculus in order to model both the parallelism of communication and sequential algorithms that take place between communication events.

  In our examples, we will use an informal notation for the sequential aspects of a process for readability and convenience, but we will use the :math:`\pi`-calculus formalism carefully in matters of parallelism and communication between processes.

