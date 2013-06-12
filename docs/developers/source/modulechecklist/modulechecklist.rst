****************************************************
Checklist: Reminders for Experienced Module Creators
****************************************************

If you've already developed a module or two, here is a cheat sheet to help you make sure you've made all the important changes.

- Your **index.rst** file should be titled with the name of the module, have a toctree maxdepth of 1, not contain the word Contents, and have the refs commented out.

- Your **Makefile** should have ::

		$(MAKE) -C $(BUILDDIR)/latex all-pdf
		tar -czf $(BUILDDIR)/latex.tar.gz $(BUILDDIR)/latex

in the latexpdf script.

- Your **conf.py** file should have been edited by ``confscript.py``. 

- You should have a **default.css_t** file in ``/modules/YourModuleName/source/_static/`` that has either been copied from an existing module or has had the following changes made manually.

This

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
