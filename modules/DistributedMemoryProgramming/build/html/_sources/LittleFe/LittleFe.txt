===========
LittleFe
===========

LittleFe is a 6-node distributed memory cluster. LittleFe is able to have its node work simultaneously. It uses MPI library to distribute the task to each node, and let each node work their own task. After they finish, we can combine the result from each node to get the overall output. LittleFe is a linux based machine. Each node of LittleFe has nVidia GT 218 GPU, which enables LittleFe to do GPU programming, CUDA C. All of them connect via the network router. 

LittleFe is a based Linux cluster, which is provided by Bootable Cluster CD (BCCD). BCCD also provides some cool applications that can be ran using MPI such as GalaxSee, Life, and Parameter-space .etc. Because of LittleFe's ability, we have also been working on Heterogeneous programming model, which combines CUDA and MPI. You will see in the next module. In this module, we will be using LittleFe for our parallel programming activities. Below is a picture of LittleFe: 

.. image:: images/LittleFe.jpg
	:width: 450px
	:align: center
	:height: 350px
	:alt: LittleFe

.. centered:: Figure 1: LittleFe