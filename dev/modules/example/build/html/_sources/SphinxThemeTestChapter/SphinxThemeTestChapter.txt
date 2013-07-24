**************
Example Module
**************

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

This is for bold **Lorem ipsum**.

This is for italic *Lorem ipsum*.

This is a subtitle
##################

This is a subsubtitle
*********************

How to include Code
*******************

.. literalinclude:: code.h  
    :language: c
    :lines: 1-24

How to create a list
********************

* This is a bulleted list.
* It has two items, the second
  item uses two lines. (note the indentation)

1. This is a numbered list.
2. It has two items too.

#. This is a numbered list.
#. It has two items too.

Colored Blocks
**************

.. seealso:: This is a simple **seealso** note.

.. note:: This is a simple **note** note.

.. warning:: This is a simple **warning** note.

.. topic:: Dig Deeper:

    Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

:Comments:

    Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

This is a Table
***************

========================= =========================== ===========================
Header 1                  Header 2                    Header 3
========================= =========================== ===========================
line 1 column 1           | line 1 column 2           | line 1 column 3   
line 2 column 1           | line 2 column 2           | line 2 column 3  
line 3 column 1           | line 3 column 2           | line 3 column 3  
========================= =========================== ===========================

How to create hyperlinks
************************
This is a link to Google_.

A really good site to find color schemes ColorHexa_

.. _Google: http://www.google.com/
.. _ColorHexa: http://www.colorhexa.com/

How to include Math
*******************

The classic model of the computer, first established by John von
Neumann in the 20:math:`{}^{th}` century, has a single CPU connected to
memory. 

Speedup(N) = :math:`N–(1–P)*(N–1)`

.. math::

    n_{\mathrm{offset}} = \sum_{k=0}^{N-1} s_k n_k

How to include image
********************

.. image:: image1.jpg
    :width: 640px
    :align: center
    :height: 360px
    :alt: alternate text

How to include figure
*********************

.. figure:: image1.jpg
    :width: 640px
    :align: center
    :height: 360px
    :alt: alternate text
    :figclass: align-center

    figure are like images but with a caption and whatever else you wish to add

glossary, download and field list
*********************************

.. glossary::
    apical
        at the top of the plant.

:download:`download code.h <code.h>`

:Whatever: this is handy to create new field

footnote and citations
**********************

This is that a test for footnote, like [1]_ may be defined at the bottom of the page:

This is a test for citation references, like [CIT2002]_ may be defined at the bottom of the page:

.. [1]
    Note that we are using footnote

.. [CIT2002]
    This is a citation