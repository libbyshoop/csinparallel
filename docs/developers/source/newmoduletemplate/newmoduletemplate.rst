*******************
New Module Template
*******************

The main title of the page goes in between lines of asterisks the same length as the title, and text of an introductory persuasion can go below that before subheadings begin (but is not necessary).

A Subheading: Formatting Text
#############################

First-level subheadings get underlined with pound signs; again, make sure that the line is the same length as the line of text above it.

:note:
	To add a note or comment, add the label ``:note:`` or ``:comment:``, and then tab the material to be included. Leave a line of whitespace to escape the note or comment block.

*Italics* can be made by surrounding text with single asterisks; **bold**, double.


Another Subheading: An Image!
#############################

.. figure:: sampleimage.png
    :width: 401px
    :align: center
    :height: 368px
    :alt: Alt-text for sample image
    :figclass: align-center
    :target: http://docutils.sourceforge.net/docs/ref/rst/directives.html#figure

    This is a figure, so here is its caption. It is separated from the fields by a line of whitespace but is still tabbed.

A Third Subheading: Sub-subheadings
###################################

Further layers of subheadings are distinguished by underlining them with something else. Sphinx is very flexible about which you use, interpreting levels based on the order they're introduced, so just be consistent.

The Sub-Subheading
******************

This one was underlined with asterisks.

The Sub-sub-subheading
----------------------

This one was underlined with dashes, but we could have swapped the last two and created the same results.

These appear in the table of contents on the left but not on our homepage, because our index.rst file has a toctree maxdepth of 1 (only the first-level headings are shown). **This is a CSinParallel convention.**