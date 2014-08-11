Configure MPICH
===============

1. Compile and Install Mpich
----------------------------

We based what we did off of the following tutorial with some differences:
https://help.ubuntu.com/community/MpichCluster

Since the mpich binaries are not available for arm we have to compile them ourselves:
First, we create a user for mpich on each node:
::

  sudo useradd -d /home/mpiu mpiu
  su mpiu
  mkdir /home/mpiu/shared

Then we compile and install MPICH on the head node.
::

  wget http://www.mpich.org/static/downloads/3.1.1/mpich-3.1.1.tar.gz #check for newest version
  tar -xvf mpich-3.1.1.tar.gz
  cd mpich-3.1.1
  ./configure --disable-f77 --disable-fc --disable-fortran --prefix=/home/mpiu/shared/mpich-install
  make
  sudo make install

Note that the options besides --prefix disable Fortran. If you need Fortran, you will have
to obtain a Fortran compiler and configure MPICH with support for it. 

2. Setup SSH keys
-----------------

Then generate an ssh key for mpiu (leave the passphrase blank) on the head node:
::

  ssh-keygen 
  cat id_rsa.pub > authorized_keys

Then copy the folder .ssh to each of the workers (eg rsync -a -e ssh ~/.ssh mpiu@tegra2:/home/mpiu). Do the same for root (passwordless authentication for root means if you have root access on one node, you can get it on all of them. We considered that acceptable since we are using the same passwords on the different nodes anyway). 

3. Configure Sharing Across the Cluster
---------------------------------------

Nfs-server is not available from the repositories and I had dependency issues compiling for arm. sshfs can be used instead
User Cluster SSH to install sshfs on the worker nodes:
::

  sudo apt-get install

Next we need add mpiu to the fuse group on the worker nodes. This is necessary for users to mount sshfs shares.
::

  sudo usermod -a -G fuse mpiu

Comment in the line allow_other in /etc/fuse.conf.
Run:
::

  crontab -e

as the user mpiu and add the following command. This mounts the directory containing the MPICH install
at startup. 
::

  @reboot sshfs -o allow_other mpiu@tegra1:/home/mpiu/shared /home/mpiu/shared
Sudo to root, run "crontab -e" again, and add the following line: 
::


  @reboot sshfs -o allow_other,nomap=ignore,idmap=file,uidfile=/etc/sshfs.uidmap,gidfile=/etc/sshfs.gidmap root@tegra1:/mnt/hd /mnt/hd

We mounted an external hard drive at /mnt/hd. This share is used to hold the home directories of the student users. 
The files sshfs.uidmap and sshfs.gidmap map usernames on the workers to their uid/gid on the head node. Our sshfs.uidmap looked like:
::

  ubuntu:1000
  mpiu:1001

Our gid map looked like:
::

  video:44
 
Finally we need to add mpich and cuda to the PATH on the worker nodes and head node:
add the following lines to /etc/profile on all nodes:
::

  PATH=/home/mpiu/shared/mpich-install/bin:/usr/local/cuda-6.0/bin:$PATH
  LD_LIBRARY_PATH=/hom/mpiu/shared/mpich-install/lib:/usr/local/cuda-6.0/bin:$LD_LIBRARY_PATH

Congratulations, you should have a working jetson cluster with mpich and cuda working!
The final section explains how to configure user management scripts and is useful
