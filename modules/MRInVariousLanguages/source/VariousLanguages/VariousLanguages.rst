*********************************
WebMapReduce in various languages
*********************************

The following subsections include the word count mapper and reducer implementations for WMR in several languages:

Scheme
######

Word count for WMR in Scheme language (spec is found on :download:`Wmr_scm.pdf <Wmr_scm.pdf>`)

mapper
------

.. literalinclude:: MR.scheme
    :language: scheme
    :lines: 1-11

reducer
-------

.. literalinclude:: MR.scheme
    :language: scheme
    :lines: 13-24

.. note:: 

  For this WMR interface for Scheme (see spec for details):

  * As indicated before, the mapper and reducer in this Scheme interface are functions.

  * String manipulation functions are primitive in Scheme, so a library function ``split`` is provided for this interface that allows one to specify delimiters by a regular-expression pattern. Type conversion is provided in Scheme through convenient (though long-named) functions ``number->string`` and ``string->number``.

  * We use Scheme-style objects as implemented at St. Olaf for the iterator for a reducer, as described above.

C++
###

Word count for WMR in C++ language (C++ style iterators, spec is found on :download:`Wmr_cpp.pdf <Wmr_cpp.pdf>`)

mapper
------

.. literalinclude:: MR.cpp
    :language: cpp
    :lines: 1-14

reducer
-------

.. literalinclude:: MR.cpp
    :language: cpp
    :lines: 16-29

.. note::

   for this WMR interface for C++ (see spec for details):

  * The ``mapper`` and ``reducer`` are methods of classes Mapper and Reducer, respectively.

  * Strings are split using the method ``Wmr::split()`` of a (predefined) library class ``Wmr.`` Rather than splitting on arbitrary regular expressions, the (required) second argument of ``Wmr::split()`` is a string of characters, any one of which counts as a delimiter. Type conversion between numbers and strings is not convenient in C++, so helper methods are provided.

  * C++-style iterators are used in the reducer method. In this style of iterator, ``operator*`` delivers the current value, ``operator++`` is used to advance to the next value, and the end of an iterator is detected by comparing that iterator for equality with the special iterator value ``WmrIterator::end``.

Java
####

Word count for WMR in Java language (Java style iterators, spec is found on :download:`Wmr_java.pdf <Wmr_java.pdf>`)

mapper
------

.. literalinclude:: MR.java
    :language: java
    :lines: 1-11

reducer
-------

.. literalinclude:: MR.java
    :language: java
    :lines: 13-24

.. note::

  for this WMR interface for Java (see spec for details):

  * The mapper and reducer are again methods of classes ``Mapper`` and ``Reducer``, respectively, as for C++.

  * Java provides useful string manipulation methods. Type conversion is provided in the Java libraries, but is inconvenient.

  * Java style iterators are used for the reducer. These have methods ``hasNext()`` which returns ``false`` when no new values exist in an iterator, and ``next()`` which returns the next unseen value and advances that iterator.

Python
######

Word count for WMR in Python3 language (Python3 style iterators, spec is found on :download:`Wmr_jpy3.pdf <Wmr_py3.pdf>`)

mapper
------

.. literalinclude:: MR.py
    :language: python
    :lines: 1-4

reducer
-------

.. literalinclude:: MR.py
    :language: python
    :lines: 6-10

.. note:: 

  Notes for this WMR interface for Python3 (see spec for details):

  * The mapper and reducer for this interface are functions, as was the case for Scheme.

  * Python provides many useful string manipulation methods for string objects, as well as convenient type conversion functions ``int()`` and ``str()``.

  * The reducer uses a Python-style iterator, which may be used conveniently in a ``for`` loop construct. 

Comparison
##########

For comparison, here is an implementation of word count mapper and reducer for Java using Hadoop map-reduce directly, without using WMR.

.. literalinclude:: Comparison.java
    :language: java
    :lines: 1-66













