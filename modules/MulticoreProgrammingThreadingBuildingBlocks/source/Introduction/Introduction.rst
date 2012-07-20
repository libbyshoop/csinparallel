************
Introduction
************

Intel Corporation has set up a special remote system that allows faculty and students to work with computers with lots of cores, called the *Manycore Testing Lab (MTL)*. In this lab, we will create a program that intentionally uses multi-core parallelism, upload and run it on the MTL, and explore the issues in parallelism and concurrency that arise.

.. topic: Extra:

    Comments in this format indicate possible avenues of exploration for people seeking more challenge in this lab.

Requirements
############

It is recommended that you work on a non-lab machine (preferably a personal laptop) for this lab; you can still use the lab computers in order to develop our multi-core program, but we need to connect to the Intel MTL system using a non-lab computer. This is because the Cisco VPN software for connecting to the MTL blocks all other network access, so we can't use it on the lab computers or it would interfere with all other uses of those computers.

If you choose to use your own computer for this lab, you will need the following materials:

* A C++ compiler, if you do not have one on your computer already.  GNU's gcc compiler is one such example, and is available on Windows, Mac OS, and Linux.

* A text editor or program to write and save C++ programs

* Cisco VPN client, which we will use to access the MTL. Instructions for installing Cisco VPN client for your machine is detailed in the next section.

* A terminal. Linux and Mac OS have UNIX-based terminals by default. For Windows, you also need ssh and scp capabilities. Putty and Cygwin are two ways to get these capabilities. With Putty_, you'll need both **putty.exe** and **pscp.exe**. Ask for help if you need it.

.. _Putty: http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html


