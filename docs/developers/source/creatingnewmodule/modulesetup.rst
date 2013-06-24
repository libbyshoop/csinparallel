***********************
Setting Up A New Module
***********************

If you've already got Python, a Git client, and the Sphinx package installed, then begin here!
:note: If your content is already in LaTeX form, check out this page instead.

Creating a Module
#################

Find your GitHub directory (on Windows, it usually puts itself in ``My Documents``). If you forked the repository, you'll have a directory called csinparallel.Navigate to ``csinparallel/modules``, and enter ``sphinx-quickstart``
to begin. Next, follow these directions below to go through the setup process.

Setup
*****

Bolded answers require the user to type something, not just hit enter.

> Root path for the documentation [.]: **YourModuleNameInCamelCase** (e.g. DistributedMemoryProgramming)

> Separate source and build directories (y/N) [n]: **y**

> Name prefix for templates and static dir [_]: [hit enter]

> Project name: **Name of the module in plain English** (e.g. GPU Programming or Distributed Memory Programming)

> Author name(s): **CSInParallel Project**

> Project version: **1**

> Project release [1]: [hit enter]

> Source file suffix [.rst]: [hit enter]

> Name of your master document (without suffix) [index]: [hit enter]

> Do you want to use the epub builder (y/N) [n]: [hit enter]

> autodoc: automatically insert docstrings from modules (y/N) [n]: [hit enter]

> doctest: automatically test code snippets in doctest blocks (y/N) [n]: [hit enter]

> intersphinx: link between Sphinx documentation of different projects (y/N) [n]: [hit enter]

> todo: write "todo" entries that can be shown or hidden on build (y/N) [n]: [hit enter]

> coverage: checks for documentation coverage (y/N) [n]: [hit enter]

> pngmath: include math, rendered as PNG images (y/N) [n]: **y**

> mathjax: include math, rendered in the browser by MathJax (y/N) [n]: [hit enter]

> ifconfig: conditional inclusion of content based on config values (y/N) [n]: [hit enter]

> viewcode: include links to the source code of documented Python objects (y/N) [n]: [hit enter]

> Create Makefile? (Y/n) [y]: [hit enter]

> Create Windows command file? (Y/n) [y]: [hit enter]

Modifying the conf.py File
##########################

In ``/GitHub/csinparallel/modules/YourModuleName/source``, run the configuration script. To do this, copy the file :download:`confscript.py <confscript.py>` into this folder and double click on it, or enter ``python confscript.py`` into a terminal or command line.  Follow the directions to enter your module's name (in plain English) and a short description when prompted.

Optional: Enabling LaTeX
************************

Most module editors will not need to create LaTeX versions of their modules. If you really want to, though, you'll need to open conf.py and change ``extentions = ['sphinx.ext.pngmath']`` to the following: 

 ::

    extensions = ['sphinx.ext.pngmath']

    if 'Darwin' in platform.uname()[0]:
	    pngmath_latex = ''
	    pngmath_dvipng = ''
    elif 'Linux' in platform.uname()[0]:
	    pngmath_latex = ''
	    pngmath_dvipng = ''
    elif 'Windows' in platform.uname()[0]:
            pngmath_latex = ''
            pngmath_dvipng = ''  

and then find your computer's tex paths and add them to the appropriate fields.

Making the Module
#################

Open the Makefile (not make.bat) and find the latexpdf entry. Add ``tar -czf $(BUILDDIR)/latex.tar.gz $(BUILDDIR)/latex`` after ``$(MAKE) -C $(BUILDDIR)/latex all-pdf``.

You now have enough of a start to see results! From ``csinparallel/modules/YourModuleName``, enter ``make html``. This will render HTML files from your ``source`` folder and place them in your ``build`` folder. Click on one to open it! 

The next section of this tutorial will outline how to begin to make changes to what you see here.
