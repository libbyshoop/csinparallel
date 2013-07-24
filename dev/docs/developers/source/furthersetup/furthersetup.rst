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

Changing CSS Files
##################

The default template is defined in the default.css file, located in ``~/github/csinparallel/modules/YourModuleName/build/html/_static/``. 

In order to use your own template (to make any changes to your module's CSS), you have to create a default.css_t file and put it into the following directory: ``~/github/csinparallel/modules/YourModuleName/source/_static/``

All existing modules have had the changes described at the end of the section made to them. You can make these changes manually or just copy a default.css_t from any existing module's ``/source/_static/`` directory into the directory given above.

:Warning: Note that the extention is css_t, not css - you have to make sure you have css_t in the extension, not the filename.

In all existing modules, this

.. code-block:: css
	
	tt {
  		background-color: #ecf0f3;
  		padding: 0 1px 0 1px;
  		font-size: 0.95em;
	}

has been changed to this:

.. code-block:: css
	
	tt {
  		background-color: #ecf0f3;
  		padding: 0 1px 0 1px;
  		/*font-size: 0.95em;*/
  		font-family:"Lucida Console", Monaco, monospace;
	}

Creating Content
################

Each module is a separate distinct head of a Sphinx documentation tree. This way each one can stand alone and be used independently. It may be useful to look at the other modules as examples of the document structure: they're like books, with each chapter in a different folder.

<<<<<<< HEAD
The next section describes the process of using git as you develop.  The section following that can be used as a template for making pages of your module.
=======
The next section can be used as a template for making pages of your module.
>>>>>>> origin/dani-dev
