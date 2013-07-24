
=================================================
The CSinParallel Module Development Process
=================================================


Developers will work from the csinparallel github repository.  You will need to
contact a project leader if you are not yet set up as a collaborator on this repository.
You can `visit the project on github <https://github.com/libbyshoop/csinparallel>`_.
You will see there the following directory structure:

::

	csinparallel/modules/_____Module Name Here_______/source
	                                                 /build
	            /images
	            /docs
	            /dev/modules/_____Module Name Here_______/source
	                                                     /build
	                /images
	                /docs
	            /README.rst

Modules that are ready for production will be in the top directory shown above.
Modules that are under development will be under the directory named 'dev'.


The central repo on github.com holds two main branches
	1. master :	origin/master
	2. develop :	origin/develop

Development Process
===================

**ALL** work on modules, both new ones and updates, should be done on the develop branch.
The way of work is the following:

- cd to your directory where you've cloned the project repository.
- Switch your local copy to the develop brnach.
- Checkout a new branch of the develop branch with -b option and choose a name for your branch.

Here is an example, with git commands, where 'ls-dev' is the name for a branch to develop from. This assumes your are in the directory where your clone copy 
of the csinparallel repository resides.

When developing a module, you will checkout off the develop branch
of the csinparallel repository:
::
	git checkout -b ls-dev develop

Now you can do your work:

- commit to your branch as you go along, 
- occasionally push up to github on your branch off develop (so you are saving a copy
in the repository),
- and occasionally pull down other's work from the develop branch and merge it with yours.


.. note:: These above steps can be done with the github client tool by committing and doing 'sync' operations, or using the command line.  Below are the command-line equivalents. See also: the `git manual pages <http://git-htmldocs.googlecode.com/git/git.html>`_ and `this git tutorial <http://www.atlassian.com/git/tutorial>`_.

From the command line:
::
	git add <file or directory>
	git commit 

	git pull origin develop

	git push origin ls-dev

Eventually, you will want to place your work on the develop branch so that
other people you are working with will be able to see it on a web server. The next few steps describe this. If you are using a Mac or linux, you can *Use the command line for this set of steps.*  However, if you are using the PC client software from github, you can switch to the develop branch in it and there should be an option to merge your branch into the develop branch, which is the equivalent of the following steps.

With this next command, go back to the develop branch:
::
	git checkout develop

From the develop branch, merge your sub-branch changes:
::
	git merge --no-ff ls-dev
	
Push the changes back up to github on the develop branch:
::
	git push origin develop


The 'develop' branch can be housed on a web server separately from
the master branch.  This will be updated on a regular basis so that
developers can share their resulting work woth others.  We envision this
site should be password protected for developer access.

The master branch will be updated **only by csinparallel personnel**.
The master branch will contain the production modules
and those files that are in the develop branch when we update it.

An example of a reStructuredText document
==========================================

The next section contains a .rst file that shows many of the tags and formatting characters that your will likely need for creating a chapter page of a module.



CSinParallel production version creation
========================================

When a module on the develop branch is ready for production, we will:

- checkout a sub-branch of the develop branch
- 'git move' the module from the 'dev' directory to the csinparallel/modules directory
- build all the versions needed (html, latex, pdf, word)
- commit and push the new module in its new location
- switch to the master branch
- merge the develop branch into the master branch
- push the changes up to the master branch
- update the official web site containing csinparallel modules (including removal ofthe 'dev' subdirectory)

Other Notes
===========

We will *not* use .gitignore to ignore the build directory in github.  
Instead, developers will be able to push their build subdirectories
into the 'dev' subdirectory.  These can then be used immediately on the 
development web servers for the project.




