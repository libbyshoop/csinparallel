Configuring User Account Management Scripts
===========================================

Our Jetson cluster is going to be used by students to run their MPICH
and Cuda code. To automate the process of creating user accounts and
giving them priveleges on the cluster, we have written several scripts for
creating users across the cluster and sharing their home directories. 

The scripts are somewhat dependant on our particular cluster setup, but if 
you have followed this guide, (in particular if you have setup the sshfs 
shares the same way) you should be able to use these scripts without modification. 

The script give_cluster_priveleges creates a user that already exists on the
head node on each of the worker nodes, creates an ssh key for passwordless 
into each of the worker nodes, and adds the user to the video group which is
necessary for running Cuda code.

However, you do not need to run this script directly. Rather, you can user the 
scripts create_instructor and create_course to create an instructor user and 
an arbitrary number of student users. See the README for more detailed instructions.

You can download the scripts here:
:download:`new_accounts.tar <new_accounts.tar>`

