.. role:: red

*******************
New Module Template
*******************

The main title of the page goes in between lines of asterisks the same length as the title, and text of an introductory persuasion can go below that before subheadings begin (but is not necessary). You can view the code for this page (or even copy the file to use as a template) at ``/csinparallel/docs/developers/source/newmoduletemplate/newmoduletemplate.rst``.

Subheadings
###########

First-level subheadings get underlined with pound signs; again, make sure that the line is the same length as the line of text above it.

Further layers of subheadings are distinguished by underlining them with something else. Sphinx is very flexible about which you use, interpreting levels based on the order they're introduced, so just be consistent.

The Sub-subheading
******************

This one was underlined with asterisks.

The Sub-sub-subheading
----------------------

This one was underlined with dashes, but we could have swapped the last two and created the same results.

These appear in the table of contents on the left but not on our homepage, because our index.rst file has a toctree maxdepth of 1 (only the first-level headings are shown). **This is a CSinParallel convention.**

Deeper subheadings are possible; should you find yourself in need of them, first question your design and then look up the documentation.


Simple Text Formatting 
######################

Inline Formatting
*****************

*Italics* can be made by surrounding text with single asterisks; **bold**, double. To include just a ``line or two of code``, enclose it in double backticks (``). 

Asterisks or backticks that could be confused with markup can be escaped with a *back*\ slash. These markups can't be nested or start or end with a space, and there must be whitespace on either side of them (the workaround for this, such as the word backslash above, is to use a backslash-escaped space: ``*back*\ slash``\ .)

Larger Blocks
*************

Inline formatting is a start, but here are some bigger guns.

Code Blocks
-----------

.. code-block:: html
	:emphasize-lines: 1
	:linenos:

	This is a code block.
	You can specify the language for coloring purposes,
	and there are several other options (such as the highlighted line above and the line numbers at left).

- Code blocks are begun with the directive :literal:`.. code-block::` (then specify the language - anything supported by `Pygments`_). 

- Code should be tabbed under the directive; leaving a newline will escape the code block. `Here`_ is a list of available options.

Lists
-----

Creating the unordered list above was simple: leaving a newline before and after each element (including the first and last), simply put an asterisk or hyphen before each element. To create an ordered list, add ``#.`` before each element, and they will be automatically numbered.

Two Kinds of Boxes
------------------

:First: To add a colored block containing a word of your choice, such as this, put it between two colons (e.g. ``:Comment:`` or ``:Observe:``) and then the material to be included. Leave a line of whitespace to escape the colored block.


.. note:: For text encased in the colored box, such as this, there are three options: ``.. note::`` (grey box), ``.. warning::`` (red box), and ``.. seealso::`` (yellow box).

.. _Pygments: http://pygments.org/languages/

.. _Here: http://sphinx-doc.org/markup/code.html


More Advanced Formatting: Custom Inlines
****************************************

To create your own custom text formatting labels (such as to make :red:`colored text`\ ), there are three distinct pieces you'll need. These are rendered in the HTML as ``<span>``\ s.

#. At the top of the .rst files you want to use your custom labels in, define them as **roles**. For example, to create the red text above, this file begins with ``.. role:: red``.

#. In your ``default.css_t`` file, add CSS class instructions for your format. To create the red text, this project's ``default.css_t`` contains 

	.. code-block:: css

		.red {
			color:red;
			font-weight:bold;
		}

#. To use your new label, put the class name between colons and then the element to be formatted between backticks. The red text above looks like this: ``:red:`colored text```.

Another Subheading: Adding Visuals
##################################

.. figure:: sampleimage.png
    :width: 401px
    :align: center
    :height: 368px
    :alt: Alt-text for sample image
    :figclass: align-center
    :target: http://docutils.sourceforge.net/docs/ref/rst/directives.html#figure

    This is a figure, so here is its caption. It is separated from the fields by a line of whitespace but is still tabbed.

**Equations** also fall under visuals. You can insert clips from screenshots as images or figures if you like, or you can type standard ``LaTeX`` expressions like this: 
``:math:`<expression>``` (substituting your own expression for <expression> but keeping the backticks).
