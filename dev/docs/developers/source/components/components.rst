**********************
Components of a Module
**********************

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8766cf121e46c568468d697515e36d67e1be51f7
Sphinx, check. Github, check. CSS, check. You can create the content of the module
that can be used by instructors for courses. However, there are more elements to a module, some of which a module author/developer is responsible for, and some of which the CSinParallel project is responsible for, and some of which is a joint effort.

Here are descriptions of the various module elements, organized by what's expected of developers, what gets done by CSinParallel, and what could go either way or be shared.
<<<<<<< HEAD
=======
Sphinx, check. Github, check. CSS, check. Now...what actually goes in a module? Here are descriptions of the various sections, organized by what's expected of developers, what gets done by CSinParallel, and what could go either way or be shared.
>>>>>>> origin/dani-dev
=======
>>>>>>> 8766cf121e46c568468d697515e36d67e1be51f7

Author's Responsibilities
#########################

Title
*****

Unless you started with this page, you've had to enter a title for your module in several places. If you did start here, check out the list of existing modules to see what kinds of titles they tend to have (and to make sure you're not duplicating an existing one!).

Summary
*******

Used on the website, this is a short description of the content, language, objectives, and length of your module. See the list of modules on the website for examples.

Context for Use
***************

Where would you teach this module? Consider the following: what course or level? with what other topics? in what kind of setting (lab, classroom, independent reading)? what technology and background knowledge would students need? 

Learning objectives
*******************

What do you want students to *know* and *be able to do* by the end of each lesson and after completing the module? How will you *assess* whether they've gained these competencies?

With those things in mind, write objectives with **active, measurable** language. For example, rather than *Students will be more knowledgeable about race conditions,* here are some more useful alternatives:
	- Students will **identify** race conditions in provided code.
	- Students will orally **define** and **give an example** of a race condition to a classmate.
	- Students will **describe** (and/or **implement**) a safer alternative to code containing potential race conditions.

The level of proficiency you expect of students is at your discretion, given their expertise and time constraints. Measurable doesn't necessarily have to mean a lengthier or more involved project: `here`_ is a list of action verbs for each stage of `Bloom's Taxonomy`_. 

.. _here: http://uwf.edu/cutla/SLO/ActionWords.pdf

.. _Bloom's Taxonomy: http://en.wikipedia.org/wiki/Bloom's_Taxonomy

Create module content
*********************

<<<<<<< HEAD
<<<<<<< HEAD
Initially, this can be in any format. We can assist you from there to make publication-quality versions using sphinx by creating .rst files, or you can get started on this yourself.  A `tool called pandoc <http://johnmacfarlane.net/pandoc/>`_ works quite well for converting your original content into a .rst file that you can work with.

For the actual .rst files, see the previous template section for eaxamples of rst formatting. How exactly you format your content depends on who you envision the audience being: are you creating material students would interact with directly, or that instructors would absorb and teach from?
=======
For the actual .rst files, see the template section. How exactly you format your content depends on who you envision the audience being: are you creating material students would interact with directly, or that instructors would absorb and teach from?
>>>>>>> origin/dani-dev
=======
Initially, this can be in any format. We can assist you from there to make publication-quality versions using sphinx by creating .rst files, or you can get started on this yourself.  A `tool called pandoc <http://johnmacfarlane.net/pandoc/>`_ works quite well for converting your original content into a .rst file that you can work with.

For the actual .rst files, see the previous template section for eaxamples of rst formatting. How exactly you format your content depends on who you envision the audience being: are you creating material students would interact with directly, or that instructors would absorb and teach from?
>>>>>>> 8766cf121e46c568468d697515e36d67e1be51f7


Write teaching notes and tips
*****************************

Anticipated sticky spots for students or things that took longer than you might think? Useful things to know about the software involved? Metaphors you've successfully used to explain the content? Jot it all down - the less other instructors have to try to read your mind or reinvent the wheel, the more everyone benefits.

Create supporting exercises
***************************

If you've got ideas for in-class activities, homework assignments, or quiz/exam questions, write them up!

Shared responsibilities
#######################

Determine metadata from controlled vocabulary
*********************************************
These tags are used to search the list of existing modules. If you'd like to add a term, contact rab@stolaf.edu or shoop@macalester.edu.

- Languages Supported: Scheme, Python, C++, C, Java, Any 

- Relevant Parallel Computing Concepts: Data Parallelism, Task Parallelism, Message Passing, Shared Memory, Distributed

- Recommended Teaching Level: Introductory, Intermediate, Advanced, Any 

- Possible Course Use: Introduction to Computer Science, Hardware Design, Software Design, Algorithm Design, Parallel Computing Systems, Programming Languages 

Create assessment questions
***************************

How will students demonstrate that they've learned what you expect them to learn? Writing learning objectives may have inspired thoughts in this area; if not, CSinParallel writers can lend a hand.

Create publication-quality content on csinparallel github repo
**************************************************************

Again, a shared responsibility - if you've already got material that's ready to go or you have time to flesh out your own outlines, excellent. If not, we can help you out.

Maintenance
***********

Will you have time to keep an eye on your module and make sure that the content doesn't get outdated when the next version of a language is released?

CSinParallel responsibilities
#############################

Publish onto csinparallel.org
*****************************

<<<<<<< HEAD
<<<<<<< HEAD
We will usually take care of final publication of your module onto csinparallel.org.  As you make changes over time (to the github version on the master branch), these will be seen when we update the web server holding the latest version of the master branch.

Test and review; curation
*************************

We will be glad to assist with review and curation of the module content, and testing it out to make sure that it seems to be ready for classroom use.

Develop assessment instruments
******************************

As part of the CSinParallel project, we will assess most modules as they are used by
instructors in courses.  We will develop assessment instruments for this purpose.

=======
=======
We will usually take care of final publication of your module onto csinparallel.org.  As you make changes over time (to the github version on the master branch), these will be seen when we update the web server holding the latest version of the master branch.

>>>>>>> 8766cf121e46c568468d697515e36d67e1be51f7
Test and review; curation
*************************

We will be glad to assist with review and curation of the module content, and testing it out to make sure that it seems to be ready for classroom use.

Develop assessment instruments
******************************
<<<<<<< HEAD
>>>>>>> origin/dani-dev
=======

As part of the CSinParallel project, we will assess most modules as they are used by
instructors in courses.  We will develop assessment instruments for this purpose.

>>>>>>> 8766cf121e46c568468d697515e36d67e1be51f7
