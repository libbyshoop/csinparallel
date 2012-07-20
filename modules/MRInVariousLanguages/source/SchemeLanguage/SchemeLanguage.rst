*******************************************
Example: WebMapReduce using Scheme language
*******************************************

* **WebMapReduce (WMR)** is a strategically simplified interface for performing map-reduce computing developed by students from St. Olaf College.  While initially supporting Scheme, the platform currently supports several high-level languages, including Python, C++, and Java.

* Wmr_scm.pdf includes specs for the functions provided in this Scheme WMR interface. :download:`download Wmr_scm.pdf <Wmr_scm.pdf>`

* This implementation of the Scheme interface for entering mappers and reducers uses an iterator for providing values in a reducer. Each call of a reducer receives all the key-value pairs for a particular key, and the two arguments for that reducer are that key and an iterator for obtaining the values.

    **Iterator** - an (object-oriented programming) object that enables a programmer to obtain each value in a collection as a sequence of values, encapsulating the internal representation of that collection.  We may visualize an iterator as a "dispenser" of values, providing one value at a time until all are exhausted.

* Iterators are used to provide reducer values because when there are very many key-value pairs, the total size of the collection of values may exceed the size of main memory. 

* As the spec indicates, the second argument of a reducer is a ``WmrIterator`` object (we'll call it ``iter``), and that object ``iter`` has two methods:

  * The call (``iter 'has-next``) returns true if a next element exists, false otherwise.
  * The call (``iter 'get-next``) delivers the next element from the iterator, and advances that iterator (so a next call to the iterator will return a fresh value, if available); this call returns false if there is no next value.
