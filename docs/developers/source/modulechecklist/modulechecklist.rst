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
	:Warning: If you're not sure, open this file and check rather than just running it again. An easy check is to see if the ``html_logo`` field is filled.