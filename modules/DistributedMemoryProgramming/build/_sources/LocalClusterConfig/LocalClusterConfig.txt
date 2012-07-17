============================
Local Cluster Configurations
============================

There are two local clusters at Macalester College, one is Selkie, and the other is LittleFe. Here we want to give you a sense of how the two clusters work.

LittleFe
--------
LittleFe is a 6-node distributed memory cluster. Each node is a motherboard consists of a CPU, I/O devices, and memory .etc. It uses Message Passing Interface (MPI) library to distribute the task to each node, and let each node work their own task simultaneously. Then we can combine the result from each node to get the overall output. Each node of LittleFe has nVidia GT 218 GPU, which enables LittleFe to do GPU programming, CUDA C. All the nodes connect via the network router.

LittleFe is Linux-based operating system, which is provided by Bootable Cluster CD (BCCD). BCCD also provides some cool applications that can be ran using MPI such as GalaxSee, Life, and Parameter-space .etc. Because of LittleFe's ability, we have also been working on Heterogeneous programming model, which combines both CUDA and MPI. You will see in the next module. In this module, we will be using LittleFe for our MPI programming activities. Below is a picture of LittleFe.

.. image:: images/LittleFe.jpg
	:width: 450px
	:align: center
	:height: 350px
	:alt: LittleFe

.. centered:: Figure 1: LittleFe


Selkie
------

Selkie is a virtual cluster, which was built in summer 2011, at Macalester College by a research group. The hardware platforms of Selkie are Apple iMacs, 16 mid-2010 models and 28 new mid-2011 models. VirtualBox product was used for virtual machines. The operating system configuration on each virtual machine is Ubuntu linix Server version 10.04. 


How to log in onto your local clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your instructor will provide the instructions of how to log in onto your local clusters.

