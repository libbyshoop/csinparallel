*******************************
What You'll Need to Get Started
*******************************

Before you can start writing modules, you'll need some software. 

The GitHub Repository
#####################

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8766cf121e46c568468d697515e36d67e1be51f7
CSinParallel modules are housed in a `github repository`_. (If you don't have a github client, `start with that`_).  This repository contains all the modules as individual Sphinx projects.
To work on the modules or create new ones, you will need to be a *collaborator* on this project.  Once you are set up as a collaborator, you will eventually clone the repository on your own machine, which you can do with the github client (there is one for Windows and for Macs).  If you are working on a linux machine, you will use the
command line to run git commands.

For reference, the `git manual pages <http://git-htmldocs.googlecode.com/git/git.html>`_ are extensive and `this git tutorial <http://www.atlassian.com/git/tutorial>`_ 
provides easy access to most commands.

<<<<<<< HEAD
=======
CSinParallel modules are housed in a `github repository`_. (If you don't have a github client, `start with that`_).  Fork the repository, which contains all the modules as individual Sphinx projects.
>>>>>>> origin/dani-dev
=======
>>>>>>> 8766cf121e46c568468d697515e36d67e1be51f7

.. _github repository: https://github.com/libbyshoop/csinparallel

.. _start with that: https://help.github.com/articles/set-up-git

Additional Software
###################

A Text Editor
*************

You'll need an editor that can work with restructured text (.rst, a simple markdown language). 

:note: We advise avoiding notepad++ on Windows; Sublime Text or gedit are good options.

Python
******

You'll need a version of Python 2 (not Python 3), preferably 2.7. You can download one from `the Python site`_. 

If you already have Python but aren't sure which version, enter ``python -v`` into a terminal or command line to check.

easy_install
************

Finally, you'll need the ``easy_install``
Python program, which you can `download here`_.
	
	On a Mac:
		- ``sudo sh setuptools-0.6c9-py2.7.egg --prefix=~``
	On a PC:
		- grab the 32-bit executable for Python 2.7 from `here`_
<<<<<<< HEAD
<<<<<<< HEAD
		- You'll also need to add your Python Scripts directory to your PATH environment variable (in System Properties, click the Advanced tab, then Environment Variables, then add C:\\Python27\\Scripts\\ to the Variable value for Variable Path). Be sure to close and restart the command prompt in order for these changes to take effect.
=======
		- You'll also need to add your Python Scripts directory to your PATH environment variable (in System Properties, click the Advanced tab, then Environment Variables, then add C:\Python27\Scripts\ to the Variable value for Variable Path). Be sure to close and restart the command prompt in order for these changes to take effect.
>>>>>>> origin/dani-dev
=======
		- You'll also need to add your Python Scripts directory to your PATH environment variable (in System Properties, click the Advanced tab, then Environment Variables, then add C:\\Python27\\Scripts\\ to the Variable value for Variable Path). Be sure to close and restart the command prompt in order for these changes to take effect.
>>>>>>> 8766cf121e46c568468d697515e36d67e1be51f7

		:note: If you've got more than one version of Python, make sure you add this to the list **before** any paths including other versions of Python.

.. _the Python Site: http://www.python.org

.. _here: https://pypi.python.org/pypi/setuptools#files

.. _download here: http://pypi.python.org/pypi/setuptools



Sphinx
######

In command line (Mac) or command prompt (Windows), enter ``easy_install -U Sphinx``. (If this doesn't work, make sure you've restarted your terminal or command prompt).

Test by entering ``sphinx-build``, which should return a help message.
