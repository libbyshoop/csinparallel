============================
Local Cluster Configurations
============================

There are two local clusters at Macalester College, one is Selkie, and the other is LittleFe. Here we want to give you a sense of how the two clusters work.

LittleFe
--------
LittleFe is a 6-node distributed memory cluster. LittleFe is able to have its nodes work simultaneously. It uses Message Passing Interface (MPI) library to distribute the task to each node, and let each node work their own task. After they finish, we can combine the result from each node to get the overall output. Each node of LittleFe has nVidia GT 218 GPU, which enables LittleFe to do GPU programming, CUDA C. All the nodes connect via the network router.

LittleFe is Linux-based, which is provided by Bootable Cluster CD (BCCD). BCCD also provides some cool applications that can be ran using MPI such as GalaxSee, Life, and Parameter-space .etc. Because of LittleFe's ability, we have also been working on Heterogeneous programming model, which combines CUDA and MPI. You will see in the next module. In this module, we will be using LittleFe for our MPI programming activities. Below is a picture of LittleFe.

.. image:: images/LittleFe.jpg
	:width: 450px
	:align: center
	:height: 350px
	:alt: LittleFe

.. centered:: Figure 1: LittleFe

How to log in onto LittleFe
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to run a program on LittleFe, we can remotely log in onto LittleFe by using the following command: ::

	ssh bccd@141.140.151.98

When it prompts for a password, enter: ::

	bccd

You are then successfully logging in onto LittleFe, and are ready to run your program.	

Selkie
------


How to log in onto Selkie
^^^^^^^^^^^^^^^^^^^^^^^^^