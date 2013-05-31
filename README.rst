*************************
How to make a new module?
*************************

Creating a Module using Sphinx
##############################

Open up a terminal and type in following command:

::

  $ cd ~/github/csinparallel/modules

  ~/github/csinparallel/modules$ sphinx-quickstart

And then follow the following answers.

::

  > Root path for the documentation [.]: The name of your module

  > Separate source and build directories (y/N) [n]: y

  > Name prefix for templates and static dir [_]: [hit enter]

  > Project name: Name of the module (e.g. GPU Programming or Distributed Memory Programming)

  > Author name(s): CSInParallel Project

  > Project version: 1

  > Project release [1]: 1

  > Source file suffix [.rst]: [hit enter]

  > Name of your master document (without suffix) [index]: [hit enter]

  > Do you want to use the epub builder (y/N) [n]: [hit enter]

  > autodoc: automatically insert docstrings from modules (y/N) [n]: [hit enter]

  > doctest: automatically test code snippets in doctest blocks (y/N) [n]: [hit enter]

  > intersphinx: link between Sphinx documentation of different projects (y/N) [n]: [hit enter]

  > todo: write "todo" entries that can be shown or hidden on build (y/N) [n]: [hit enter]
 
  > coverage: checks for documentation coverage (y/N) [n]: [hit enter]
 
  > pngmath: include math, rendered as PNG images (y/N) [n]: y

  > mathjax: include math, rendered in the browser by MathJax (y/N) [n]: [hit enter]

  > ifconfig: conditional inclusion of content based on config values (y/N) [n]: [hit enter] 

  > viewcode: include links to the source code of documented Python objects (y/N) [n]: [hit enter]

  > Create Makefile? (Y/n) [y]: [hit enter]

  > Create Windows command file? (Y/n) [y]:y

Modify the conf.py File
#######################

#. go to /github/csinparallel/modules/YourModuleName/source

#. Open up the conf.py file and make the following changes

* Change 

  :: 

    import sys, os
   
  to following:
  
  ::
  
    import sys, os, platform

* Change

  :: 

    extentions = ['sphinx.ext.pngmath'] 

  to following:

  ::

    extensions = ['sphinx.ext.pngmath']

    if 'Darwin' in platform.uname()[0]:
	    pngmath_latex = '/usr/local/texlive/2011/bin/x86_64-darwin/latex'
	    pngmath_dvipng = '/usr/local/texlive/2011/bin/x86_64-darwin/dvipng'
    elif 'Linux' in platform.uname()[0]:
	    pngmath_latex = '/usr/bin/latex'
	    pngmath_dvipng = '/usr/bin/dvipng'
    elif 'Windows' in platform.uname()[0]:
            pngmath_latex = ''
            pngmath_dvipng = ''

* Change 

  ::
    
    copyright = u'2012, CSInParallel' 

  to following

  ::

    copyright = u'This work is licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License'

* Comment out 

  ::
   
     version = '1'

* Comment out    

  ::
   
     release = '1'

* Comment in and then change 

  ::

    html_title = None 

  to following

  ::
   
    html_title = 'Your Module Name' (including the single quotation marks)

* Comment in and then change 

  ::

    html_logo = None 

  to following

  ::

    html_logo = '../../../images/CSInParallel200wide.png' (including the single quotation marks)

* Comment in and then change 

  :: 
  
    html_show_sourcelink = True 

  to following

  ::

    html_show_sourcelink = False

* Add following

  ::

    'releasename': '', 'classoptions': ',openany,oneside', 'babel' : '\\usepackage[english]{babel}'

  to

  ::

    latex_elements = {

    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
    }

* Find the following

  ::

    latex_documents = [
      ('index', 'GPUProgramming.tex', u'GPU Programming',
       u'CSInParallel Project', 'manual'),
    ]

  and delete the word "Documentation"

* Find the following

  ::

    man_pages = [
       ('index', 'YourModuleName', u'Your Module Name Documentation',
        [u'CSInParallel Project'], 1)
    ]

  and delete the word "Documentation"

* Find the following

  ::

    texinfo_documents = [
      ('index', 'GPUProgramming', u'GPU Programming',
       u'CSInParallel Project', 'GPUProgramming', 'One line description of project.',
       'Miscellaneous'),
    ]

  and delete the word "Documentation"

Modify the index.rst File
#########################

1. go to /github/csinparallel/modules/YourModuleName/source
2. open up the index.rst file and make the following changes

* Delete "Welcome to YourModuleName's documentation!" and change it "Your Module Name"

* Delete "Contents"

* Change the 

  ::

    :maxdepth: 2

  to 

  ::

    :maxdepth: 1

* Delete 

  :: 

    Indices and tables
    ==============================

* Comment out the refs like

  ::

    .. comment 
	    * :ref:`genindex`
	    * :ref:`modindex`
	    * :ref:`search`

Modify the Makefile file
########################

1. go to /github/csinparallel/modules/YourModuleName
2. open up makefile file in an editor and make the following changes

* find latexpdf entry

* add "tar -czf $(BUILDDIR)/latex.tar.gz $(BUILDDIR)/latex"(without quote sign) after "$(MAKE) -C $(BUILDDIR)/latex all-pdf"

* make sure you pressed a tab to make the line you added to line up with others instead using a bunch of spaces!!

Build the html
##############

In your linux or mac terminal, or your windows command line, go to your module's root directory.

::
  
  $ cd ~/github/csinparallel/modules/yourmodulename

Then excute make html command

::

  ~/github/csinparallel/modules/yourmodulename$ make html

This will build the html using our modified conf.py, index.rst and Makefile files.

Using your own template
#######################

1. The default template is defined in the defualt.css file. You can access this file by cd into its directory.

::

  $ cd ~/github/csinparallel/modules/YourModuleName/build/html/_static

2. In order to use your own template, you have to create a default.css_t file and put it into the following directory.

::

  $ cd ~/github/csinparallel/modules/YourModuleName/source/_static

For all existing modules, we made some small changes to the template. You will find details at the end of the section. If you would like to use our template, you can copy the defualt.css_t from any existing modules and put it into the above directory of your module. Just go through the follwoing steps.

	1. go to ~/github/csinparallel/modules/AnyExistingModule/source/_static
	
	2. you will see a default.css_t file. 

	3. copy that file and put it into ~/github/csinparallel/modules/YourModuleName/source/_static

Note that the extention is css_t, not css, you have to make sure you have css_t in extension, not filename.

Note that We recommnand you take the defualt.css and modify it to create your own template.

3. About the changes we made

We changed

::

  tt {
    background-color: #ecf0f3;
    padding: 0 1px 0 1px;
    font-size: 0.95em;
  }

to the following

::

  tt {
      background-color: #ecf0f3;
      padding: 0 1px 0 1px;
      /*font-size: 1.35em;*/
	font-family:"Lucida Console", Monaco, monospace;
  }




































    
