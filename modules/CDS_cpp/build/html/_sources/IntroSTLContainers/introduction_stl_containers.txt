Introduction to STL containers
===============================



*Note:* This reading refers to topics that we will not pursue in
homework, but which may arise in general C++ coding. Such topics will be
marked\ **(Extra)**.

Introduction
------------

The C++
`Standard Template Library <http://www.cplusplus.com/reference/stl/>`_\ *(STL)*
is a software library of algorithms, data structures, and other features
that can be used with any pre-defined C++ type and with user-defined
types (that provide members such as a copy constructor and an assignment
operator).

The STL uses a C++ feature called the *template* to substitute
types throughout a class or function definition, which enables a
programmer to avoid rewriting definitions that differ in the type being
used.

For example, our IntArray class for homework provides a class
structure around an array of int, with error checking, default
initialization for array elements, whole-array assignment, etc. To
define a class LongArray or FloatArray, we would only need to replace
the type name int by a different type name long or float, etc. Templates
make it possible to define a class Array<T> in which a type name T is
used thrroughout the definition of that class, so that the notation
Array<int> would specify a class that behaves just like IntArray,
Array<float> would specify FloatArray, etc. Here, the angle brackets
<...> are the syntax for invoking a template, and they enclose a type
name to use throughout a definition.

A programmer can define her/his own templated classes. For
example, here is a complete program with a definition of a templated
structPair<T> that represents an group of two elements of the same type,
together with an example use of that template.

::

	#include <iostream>
     using namespace std;
    
     template <class T>
     struct Pair {
       T x;
       T y;
    
       Pair(T val1, T val2) { x = val1;  y = val2; }
     };
    
    
     int main() {
       Pair<float> p(3.14, -1.0);
    
       cout << p.x << ", " << p.y << endl;
     }


The STL *containers* are templated classes defined in the STL
that are designed to hold objects in some relationship. Examples:

* The array classes we have implemented in homework are examples of containers, although STL's array-like container vector<T> has some added useful features such as an ability to resize incrementally that our array classes did not include.

* Another STL container is called a queue<T>, and contains a linear list of elements that are added at one end and removed at another, comparable to a check-out line of customers at a grocery store. Unlike a vector, the elements of a queue do not necessarily appear consecutively in memory.

* **(Extra)** Another example is the map<K,T> container, which includes an operator[] that accepts "indices" of a type K (not necessarily an integer type) and associates values of type T with those "index" values.

This is not a complete list of STL containers, but illustrates
some commonly used ones. See
`www.cplusplus.com.reference.stl <http://www.cplusplus.com/reference/stl/>`_
for documentation of all STL containers


The STL container vector
------------------------

STL's vector<T> is a templated class that behaves like an
"elastic array," in that its size can be changed incrementally. For
example, consider the following example program:

::
    #include <iostream>
    using namespace std;
    #include <vector>
    int main() {
      vector<int> vec(4);
      vec[1] = 11;
      vec.at(2) = 22;
      vec.push\_back(33);
      for (int i = 0;  i < vec.size(); i++)
        cout << vec.at(i) << endl;
    }

The preprocessor directive

::

	#include <vector>

tells the compiler what it needs to know to compile with vectors

The first statement is a variable definition that defines vec to
be a vector of four integers. Each integer location is initialized at 0
by default.

The third statement uses the at() method of vector, and behaves
exactly like operator[] except for a different style of error handling
(at throws "exceptions" instead of crashing the program).

The call to vector's push\_back() method appends a new element 33
to the end of vec

The resulting vector vec contains five int elements: 0, 11, 22,
0, and 33. Those values will be printed by the final loop.

A similar exercise could have been programmed with float, Dog, or
another type.

**(Extra)** In the "exception" style of error handling used by
methods such as at(), a runtime error (i.e., while the program is
running, such as index out of bounds for a vector object) creates an
object called an *exception* that includes information about that error.
We say that the error condition *throws* the exception object. C++
provides an optional try/catch feature for capturing thrown exceptions
and taking action on them; otherwise, throwing an exception causes a
program to crash.

The vector method push\_back() enables a programmer to extend the
length of a vector object by one element, as shown above. That new
element is added at the "back" or highest-indexed end of that vector.

Another vector method pop\_back() enables a programmer to delete
the last element of a vector, thus decreasing its length by 1. The
methodpop\_back() requires no argument and returns no value.

The vector index operator [ ] and the method at() both provide
immediate access to any element in a vector.

The method size() returns the number of elements currently in a
vector object.

The method back() returns the last element of a non-empty vector.
Thus, vec.back() returns the same value as vec[vec.size()-1]
orvec.at(vec.size()-1) .

The vector container provides methods for inserting or removing
values at locations other than the back of a vector object, called
insert()and erase(). However, these methods are not as efficient as
push\_back() and pop\_back(). This is because vector elements are stored
consecutively in memory, so inserting or removing an element at a
position other than the back requires copying all element values from
that position through the back value.

**(Extra)** Note that the methods *insert()* and *erase()*
require iterator objects to specify position within a vector object. An
*iterator* is an object that contains a pointer and has methods for
certain pointer operations, such as a method next() for advancing to the
next element in an array or vector.

.. seealso::
	`www.cplusplus.com/reference/stl/vector <http://www.cplusplus.com/reference/stl/vector>`_ for a reference on vectors.

The STL container queue
-----------------------

STL's queue<T> is a templated class that is a *FIFO (first-in
first-out)* container. This means that it is capable of holding an
indeterminate number of elements of a particular type, organized in an
ordered list, with each element added at one end of the list (the
*back*) and removed at the other end (the *front*).

Whereas STL's vector templated class has many methods, a queue
has only six specified methods
(see `www.cplusplus.com/reference/stl/queue <http://www.cplusplus.com/reference/stl/queue>`_):

* push(), which adds an element at the end of a queue,

* pop(), which removes an element from the beginning of a queue,

* front(), which returns the element at the front of a non-empty queue (next to be popped),

* back(), which returns the element at the back of a non-empty queue (most recent to be pushed),

* empty(), which returns Boolean True if there are no elements in a queue and False otherwise, and

* size(), which returns the number of elements currently in a queue.

Here is a code example of using a queue.
::

	#include <iostream>
	    using namespace std;
	#include <queue>
	int main() {
	   
	      queue<float> q;
	      q.push(1.23);
	      q.push(4.56);
	      q.push(7.89);
	      while (!q.empty()) {
	        cout << q.front() << endl;
	        q.pop();
	      }
	}

The output from this code should be the numbers 1.23 *then* 4.56
*then* 7.89, one per line.

An STL vector could be used in a situation where a *queue* would
be appropriate (e.g., simulating a process comparable to a grocery-store
checkout line), using the vector methods push\_back(), front(), and
erase() (to remove the front element). But a queue can be implemented
more efficiently than a vector for this purpose, avoiding the copying of
elements that are needed for vector's erase() method.

**(Extra)** The underlying data structure for a queue can be
specified when that queue is created, using a second template
parameter.

On the other hand, a queue provides no index or at() operator for
accessing an element other than the front or back elements. The ability
to access arbitrary element locations (e.g., via indices) is called
*random access*, and if random access is needed, a vector may be more
desirable than a queue.

As with vector, the templated container class queue can accept a
user-defined type for its elements. Thus

::

	queue<Dog> ,  queue<const Dog\*> ,

and other types may be used.


 
