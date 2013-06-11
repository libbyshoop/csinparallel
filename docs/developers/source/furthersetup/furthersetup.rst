***********************
Editing Your New Module
***********************

Most of these sections describe CSinParallel conventions that differ from Sphinx defaults; please follow them.

index.rst
#########
This file becomes the index of your site; it is how to order and link together the chapters of your module. But first, follow these instructions:

- Replace "Welcome to YourModuleName's documentation" with "Your Module Name".
- Delete "Contents"
- Change ``:maxdepth: 2`` to ``:maxdepth: 1``
- Delete ``Indices and tables`` and the line of equal signs below it.
- Comment out all of the refs by adding a line that says  ``.. comment`` and tabbing them all below it so it looks like this:

.. code-block:: rst

	.. comment
		* :ref:`genindex`
		* :ref:`modindex`
		* :ref:`search`

Be sure to put a newline after this block.

Once you've generated some other pages, you'll add them on lines tabbed underneath the ``:toctree:`` directive.

Changing CSS Files
##################

