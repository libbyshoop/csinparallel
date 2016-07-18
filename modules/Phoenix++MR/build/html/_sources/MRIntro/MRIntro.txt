What is MapReduce?
==================


Motivation
------------

In today's world, multicore architectures are ubiquitous. But, the majority of 
programs that people write are still serial. Why is that? While some people may 
be unaware that they can leverage the multiple cores on their computers, the 
truth is that parallel computing is very difficult. In many cases, the 
programmer must consider many factors that have nothing to do with problem he 
or she is trying to parallelize. For example, to implement a program in Pthreads, 
a programmer must physically allocate, create and join any threads they want to 
use. The programmer must also be aware data races, and use synchronization 
constructs as necessary.

This is not unique to Pthreads. In MPI for example, you have to explicitly 
specify what messages you want to send to what node, and how to synchronize 
messages. As you can imagine, this creates a lot of overhead for the 
programmer. As those who have programmed previously in Pthreads, MPI, or OpenMP 
can attest, debugging parallel programs can be very difficult! When things 
execute in parallel, they execute non-deterministically. This non-determinism 
can cause a lot of headaches.

As multicore and parallel systems became more prevalent, computer scientists 
began to ask the question if parallel computing is harder that it needs to be. 
Some libraries such as OpenMP "hide" some of the work required with threads 
through the use of pragmas. Hiding implementation details is known as 
abstraction. However, even with abstraction, the programming still has to worry 
a lot about the "management" aspects of the program. Furthermore, similar 
applications can be parallelized in the way. Researchers began exploring ways 
to create an automated framework for parallel computing.

Enter MapReduce
----------------


In 2004, Jeffrey Dean and Sanjay Ghemawhat of Google published a paper on the 
MapReduce paradigm. Google uses MapReduce as the backbone of its search engine, 
and uses it for multiple operations. It is important to note that Google did 
*not* invent MapReduce; the paradigm has existed for decades in functional 
languages. However, the paper's release was a watershed moment in parallel 
computing, and spelled the beginning of an upsurge in interest in the paradigm 
that has led to many innovations over the last decade.

Google's implementation of MapReduce is closed source and proprietary. In 2006, 
work on the Hadoop project was started by Doug Cutting, an employee of Yahoo!. 
Hadoop is named after a plush toy elephant belonging to Cutting's son, and the 
eponymous elephant features prominently in the Hadoop logo. Over the last six 
years, Hadoop has been widely adopted by many tech giants, including Amazon, 
Facebook and Microsoft.

It is important to note that both Google's implementation of MapReduce and 
Hadoop MapReduce were designed for very large datasets, on the order of 
hundreds of gigabytes and petabytes. The goal is to efficiently streamline the 
processing of these large numbers of documents by distributing them over 
thousands of machines. Note that for smaller datasets, the system may have 
limited benefit; the Hadoop Distributed File System (HDFS) can prove to be a 
bottleneck. However, the concept of MapReduce is still very attractive to 
programmers with smaller datasets or more limited computational resources, due 
to its relative simplicity.

.. note:: Want to play with a Hadoop system on the web? Check out WebMapReduce!  
          Access the module at this link. 

Phoenix and Phoenix++
----------------------


In 2007, a team at Stanford University led by Christos Kozyrakis began 
exploring how to implement the MapReduce paradigm on multi-core platform. Their 
thread-based solution, Phoenix, won best paper at HPCA'07, and has been cited 
over 800 times. An update on Phoenix (Phoenix 2) was released in 2009. In 2011, 
Phoenix++ was released. A complete re-write of the earlier Phoenix systems, 
Phoenix++ enables development in C++, and significantly modularizes and 
improves the performance of the original code base.

.. note:: This entire module is based on Phoenix++ the latest release of 
          Phoenix. Please note that if you are interested in using the earlier 
          modules, these instructions may not directly apply.

