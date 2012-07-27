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

  > Name prefix for templates and static dir [_]: defualt

  > Project name: Name of the module (e.g. GPU Programming or Distributed Memory Programming)

  > Author name(s): CSInParallel Project

  > Project version: 1

  > Project release [1]: 1

  > Source file suffix [.rst]: defualt

  > Name of your master document (without suffix) [index]: defualt

  > Do you want to use the epub builder (y/N) [n]: defualt

  > autodoc: automatically insert docstrings from modules (y/N) [n]: defualt

  > doctest: automatically test code snippets in doctest blocks (y/N) [n]: defualt

  > intersphinx: link between Sphinx documentation of different projects (y/N) [n]: defualt

  > todo: write "todo" entries that can be shown or hidden on build (y/N) [n]: defualt
 
  > coverage: checks for documentation coverage (y/N) [n]: defualt
 
  > pngmath: include math, rendered as PNG images (y/N) [n]: y

  > mathjax: include math, rendered in the browser by MathJax (y/N) [n]: defualt

  > ifconfig: conditional inclusion of content based on config values (y/N) [n]: defualt 

  > viewcode: include links to the source code of documented Python objects (y/N) [n]: defualt

  > Create Makefile? (Y/n) [y]: defualt

  > Create Windows command file? (Y/n) [y]:n

Modify the conf.py File
#######################

#. go to /github/csinparallel/modules/YourModuleName/source

#. Open up the conf.py file and make the following changes

* Change

  :: 

    extentions = ['sphinx.ext.pngmath'] 

  to following:

  ::

    extensions = ['sphinx.ext.pngmath']

    if 'Darwin' in os.uname()[0]:
	    pngmath_latex = '/usr/local/texlive/2011/bin/x86_64-darwin/latex'
	    pngmath_dvipng = '/usr/local/texlive/2011/bin/x86_64-darwin/dvipng'
    elif 'Linux' in os.uname()[0]:
	    pngmath_latex = '/usr/bin/latex'
	    pngmath_dvipng = '/usr/bin/dvipng'

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
   
    html_title = 'Your Module Name'

* Comment in and then change 

  ::

    html_logo = None 

  to following

  ::

    html_logo = '../../../images/CSInParallel200wide.png'

* Comment in and then change 

  :: 
  
    html_show_sourcelink = True 

  to following

  ::

    html_show_sourcelink = False

* Add following to 

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

Using your own template
#######################

We made some modification on the html template. To be exact, we modified the default.css file and put in to the _static folder in source directory to let Sphinx use it when building html.

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

You can also create your own template. 

How to tell Sphinx to use your template
***************************************

1. go to ~/github/csinparallel/modules/AnyExistingModule/source/_static

2. you will see a default.css_t file. 

3. copy that file and put it into ~/github/csinparallel/modules/YourModuleName/source/_static





































    
