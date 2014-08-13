Setting up a Jetson Tegra Cluster
=================================

Introduction
-------------
These instructions are based on the cluster we set up at Macalester College during 
the summer of 2014. The cluster consists of 6 Jetson Tegra Tk1's. The Jetson Tegra 
is an development board made by Nvidia with embedded applications in mind. Because
the Jetson is a small, relatively afforadable board with powerful Nvidia graphics,
they are idea for setting up a cluster for teaching purposes. The Jetson is an ARM 
board and runs the Ubuntu ARM port. Because of this, our cluster is bit less stable
then an x86 system. Also, the binaries available in the repositories are limited 
and we had to compile a fair bit of the software we use from source. 

Prerequisites
-------------
The particular hardware you use is flexible. You need an ethernet switch, USB hub,
and a USB ethernet adapter with linux drivers for the cluster. The number of Jetson 
Tegra Tka1's is up to you and you can always add more later. Adding a SATA hard drive
is not strictly needed but is a good idea.

These instructions assume a basic knowledge of the Unix command line. 

|Cluster1|

|Cluster2|


.. |Cluster1| image:: cluster1.jpg
.. |Cluster2| image:: cluster2.jpg
