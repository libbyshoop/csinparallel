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

Once you've generated some other pages, you'll add them on lines tabbed underneath the ``:toctree:`` directive. These should be relative to the folder containing index.rst and added as **document names**, meaning no *trailing or leading slashes* and no *file extensions* (e.g. documents/index, not /documents/index.rst )

Creating Content
################

Each module is a separate distinct head of a Sphinx documentation tree. This way each one can stand alone and be used independently. It may be useful to look at the other modules as examples of the document structure: they're like books, with each chapter in a different folder.

The next section describes the process of using git as you develop.  The section following that can be used as a template for making pages of your module.
