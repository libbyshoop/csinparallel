.. Building a Raspberry Pi documentation master file, created by
   Rohit Bagda on July 12 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********************************
Building a Raspberry Pi Cluster
***********************************
**Last Updated:** |date|

This module acts as a tutorial to help users build a cluster of Raspberry Pis for Parallel Programming and Distributed Processing. The guide is a step by step procedure to build a simple 4-node Raspberry Pi 3 cluster with a shared network file system. Note that clusters can be of any length n≥2.

Materials Needed
#################
1.  4 x `Raspberry Pi 3`_
2.  4 x `Micro SD Card`_ (Preferably 8GB or higher) and a `Card Reader`_
3.  4 x `Ethernet Cable`_ (Preferably 1 ft. long)
4.  4 x `USB to Micro USB cable`_ for Raspberry Pi Power (Preferably 1 ft. long)
5.  1 x `6 Port USB Battery Charger`_ (60W 2.4 amps per port or 12 amps overall)
6.  1 x `5 Port Ethernet Switch`_
7.  1 x `Monitor`_, `Keyboard and Mouse`_
8.  1 x `4 Port HDMI Switch`_ (Optional, but recommended)
9.  1 x `4 Port USB Switch`_ for Keyboard and Mouse (Optional, but recommended)
10. 1 x `4 Node Acrylic Raspberry Pi Stand`_ (Optional, but recommended)


.. _Raspberry Pi 3: https://www.amazon.com/Raspberry-Model-A1-2GHz-64-bit-quad-core/dp/B01CD5VC92/ref=sr_1_3?s=pc&ie=UTF8&qid=1499808173&sr=1-3&keywords=raspberry+pi+3

.. _Micro SD Card: https://www.amazon.com/SanDisk-MicroSDHC-Standard-Packaging-SDSDQUAN-008G-G4A/dp/B00M55C0VU/ref=sr_1_5?s=electronics&ie=UTF8&qid=1499808284&sr=1-5&keywords=micro+sd+card+8+gb

.. _Card Reader: https://www.amazon.com/Anker-Portable-Reader-RS-MMC-Micro/dp/B006T9B6R2/ref=sr_1_1?s=electronics&ie=UTF8&qid=1499808439&sr=1-1-spons&keywords=sd+card+reader&psc=1

.. _Ethernet Cable: https://www.amazon.com/Cable-Matters-5-Pack-Snagless-Ethernet/dp/B00C4U030G/ref=sr_1_1?s=electronics&ie=UTF8&qid=1499808595&sr=1-1&keywords=ethernet+cable+1ft

.. _USB to Micro USB cable: https://www.amazon.com/Sabrent-6-Pack-Premium-Cables-CB-UM61/dp/B011KMSNXM/ref=sr_1_3?s=electronics&ie=UTF8&qid=1499808702&sr=1-3&keywords=usb+to+micro+usb+cable+1ft

.. _6 Port USB Battery Charger: https://www.amazon.com/Anker-PowerPort-High-Speed-Charging-VoltageBoost/dp/B00P936188/ref=sr_1_1?s=electronics&ie=UTF8&qid=1499808936&sr=1-1&keywords=anker+6+port+usb+charger

.. _5 Port Ethernet Switch: https://www.amazon.com/TRENDnet-Unmanaged-Gigabit-GREENnet-Desktop/dp/B002HH0W5W/ref=sr_1_4?s=electronics&ie=UTF8&qid=1499808983&sr=1-4&keywords=greennet+switch

.. _Monitor: https://www.amazon.com/GeChic-2501C-Portable-Monitor-Inputs/dp/B00H4MWMWQ/ref=sr_1_1?s=electronics&ie=UTF8&qid=1499809271&sr=1-1-spons&keywords=portable+monitor&psc=1

.. _Keyboard and Mouse: https://www.amazon.com/Keyboard-Jelly-Comb-Rechargeable-Wireless/dp/B01NCW2JR9/ref=sr_1_4?s=electronics&ie=UTF8&qid=1499809455&sr=1-4&keywords=jelly+comb+keyboard+mouse+combo

.. _4 Port HDMI Switch: https://www.amazon.com/LinkS-Switcher-Supports-Wireless-function/dp/B01FHILVX4/ref=sr_1_18?ie=UTF8&qid=1499806934&sr=8-18&keywords=4+port+hdmi+switch+with+remote

.. _4 Port USB Switch: https://www.amazon.com/dp/B01CU4QD1I?psc=1

.. _4 Node Acrylic Raspberry Pi Stand: https://www.amazon.com/Raspberry-Pi-Complete-Stackable-Enclosure/dp/B01LVUVVOQ/ref=sr_1_12?ie=UTF8&qid=1499805835&sr=8-12&keywords=acrylic+raspberry+pi+case

Overview:
##########
Before you begin please read Helpful Ideas and Notes so that you can configure this cluster as you want it to be.

* Set up Physical Cluster
* Set up Operating System
* Software Installation and Wireless Configuration
* SSH Remote Login Setup
* Set Up and Mount Network File System (NFS)
* Broadcasting and Shutting Down
* Time and Date Synchronization
* Helpful Ideas and Notes

Setting up the Physical Cluster
###############################
An acrylic 4-node case can be used to stack the RPi’s on top of each other. For the display and input devices we will use a 4 Port HDMI switch and a 4 Port Keyboard/Mouse switch in order to make it convenient for us to switch between nodes as we work on each one of them. We use a monitor which has USB cable for power and an HDMI to Mini-HDMI for the Input Signal. The RPi’s are then connected to a 5 port Ethernet Switch which will be used to set up a network between the cluster. A USB to Barrel connector will be needed to power the Ethernet Switch. Lastly, we will connect each of the RPi’s to power using USB to Micro-USB cables. All the components can be powered using a regular wall charger but in order for the cluster to be more compact and consisting of less wires we will use a 6 port USB Battery Charger to power the Monitor, Ethernet Switch and  the 4 RPi’s.

Setting up Rasbian OS
#########################
**Please Note that these following set of instructions were made on Rasbian GNU/Linux 8.0 (jessie) and may or may not work on previous versions of Rasbian**

To set up the OS we will need 4 MicroSD cards (preferably 8 GB or higher). These Micro-SD Cards will need to be flashed with fresh Rasbian images. To do this we will require a PC and an SD Card reader.

Our Cluster will have a head node on which all our files will be stored and mounted for the other nodes. Essentially this will be our primary work station throughout. Hence we will require a completely functional GUI and hence we will flash the Micro-SD Card of the head node with the latest version of the Rasbian Pixel. Since we will be able to remotely login into the worker nodes, we will use Rasbian Lite so as to make things run faster. So we will download the `two Rasbian images`_ on our computer which can read a Micro-SD card, and flash each of the cards with it’s respective Rasbian images using a software called `Etcher`_. Using `Win32 Disk Imager`_ is also feasible.

.. _two Rasbian images: https://www.raspberrypi.org/downloads/raspbian/

.. _Etcher: https://etcher.io/

.. _Win32 Disk Imager: https://sourceforge.net/projects/win32diskimager/



Each node will have a user ‘pi’ and a password ‘raspberry’.

Software Installation and Wireless Setup
#########################################
Since head node is the only RPi with a GUI we will only need to access the Internet on that. But before we disable the Wifi on the worker nodes, we should connect the Ethernet switch to the wall so that each of the RPi’s have a wired Internet connection temporarily in order to install software we will require later for SSH, NFS Mounting and MPI.

Firstly, we will run updates and upgrades using the following commands::

  sudo apt-get update
  sudo apt-get upgrade

Then we will install mpich2 which will help us run MPI (Message Passing Interface) code on the cluster, SSH (Secure Shell) which will be required for remote login and NTP (Network Time Protocol) which we will need to synchronize date and time across the cluster::

  sudo apt-get install mpich2
  sudo apt-get install openssh-server
  sudo apt-get install ntp

Next, on the head node we will be installing NFS kernel server.::

  sudo apt-get install nfs-kernel-server

And on the worker nodes we will be installing NFS Common::

  sudo apt-get install nfs-common

With this we have installed all the required software for us to run MPI Code on the cluster. If you wish to install any other software then you should install it now because we will be removing internet from the worker nodes and setting up wireless only on the head node.

To set up wireless and we will need to edit two files using a terminal text editor (nano is the simplest one) :
/etc/network/interfaces and
/etc/wpa_supplicant/wpa_supplicant.conf

::

  cd /etc/wpa_supplicant
  sudo nano wpa_supplicant.conf

.. literalinclude:: ../source/wpa_supplicant.conf

In order for us to establish a network between the cluster through ethernet cables we need static ip addresses. To do this, we edit the interfaces file on the head node. It is good practice to make a copy of the file before we make changes to it. This can be done by doing the following:

::

  cd/etc/network
  sudo cp interfaces interfaces.orig

Now, we will edit the interfaces file

::

  cd /etc/network
  sudo nano interfaces

.. literalinclude:: ../source/interfaces_head

Next, we will have to edit the interfaces file (after making a copy like above) on each of the worker nodes to the following:

.. literalinclude:: ../source/interfaces_worker

Note: the ip address below will change according to the order of the 4 nodes, i.e. the 2nd node will have the address 192.168.1.11, the 3rd will have 192.168.1.12 and the 4th will have 192.168.1.13 as its address.

SSH Remote Login Setup
#######################
Before we begin we want to assign names to each of the nodes. To do this we changed the hostname and hosts file.

::

  sudo nano /etc/hostname

The hostname file will contain the name of the local host i.e. the name of that node. We called the head node “head” and the worker nodes ‘node1’, ‘node2’ and “node3’ respectively.

Next,
We add the static ip addresses of the nodes in the hosts file.
::

  sudo nano /etc/hosts

.. literalinclude:: ../source/hosts

The above process has to be repeated on all 4 nodes.

Since we want the cluster nodes to communicate with each other without having to ask for permission each time we will set up SSH (Secure Shell) remote login. In order to do this, we will generate 2048 bit RSA key-pair on the head node and then copy the SSH ID in all 4 nodes.
::

  ssh-keygen -t rsa -b 2048

Note: The ssh key should be stored in the .ssh folder and need not require a pass phrase. While generating the keys the user will be prompted for destination folder and a pass phrase. The user can just hit return thrice.

Now, we will copy the SSH ID to all the nodes.

::

  ssh-copy-id head.
  ssh-copy-id node1
  ssh-copy-id node2
  ssh-copy-id node3

Next, we will need to SSH into all the nodes including itself. For the first time we will require a password (raspberry) when we try to do so.

::

  ssh head

We notice that we are re-logged in into the head node. We need to exit this connection before we SSH into all the other nodes i.e. ssh into node1, node2 and node3 from the head node and exit each time. This will update the known_hosts file in the .ssh directory. The known_hosts file keeps track of the cluster nodes which have permissions to ssh into nodes without a password.

Our last step for this setup will be to copy known_hosts, id_rsa public and private keys from the .ssh folder in the head node to the .ssh folder of all the other nodes. We can do this using secure copy.

We use the following steps on the head node to do this:
::

  cd .ssh

Run the following commands: ::

    scp known_hosts pi@node1:.ssh
    scp id_rsa pi@node1:.ssh
    scp id_rsa.pub pi@node1:.ssh

Repeat for node2 and node3

After this we will be able SSH to and from any node in the Cluster.

Mount Network File System
##########################

The Network File System (NFS) mounting is crucial part of the cluster set up in order for all the nodes to have one common working directory. We will be taking advantage of the nfs-kernel-server and nfs-common which we had installed earlier. To begin we will create a directory called cluster_files on the head node. Essentially it can be named anything as long as it's consistent throughout all the other nodes. We will make this directory in the home directory so that it is easily accessible.

::

    cd
    sudo mkdir cluster_files

Moving forward, we will have change the /etc/exports file on the head node in order for the above directory be accessible by the worker nodes

::

    sudo nano /etc/exports

We will add the following line at the end of the above file

.. literalinclude:: ../source/exports

Next we will be working on the worker nodes. Since we have SSH set up, we can simply ssh into each node instead of switching displays.

::

  ssh node1

On each node we will have to mount the cluster_files directory of the head node. For this to work first we will make the cluster_files directory in each of the worker nodes like we did for the head node. After this is done, we will add the following line to the fstab file in each node. This will assign the path to the cluster_files directory on the head node.

Note:
Please make sure that you do not alter anything in the rest of the fstab file because this might ruin your entire system. Again, for safety, we will make a copy of the original by doing the following:

::

  cd /etc
  sudo cp fstab fstab.orig

Now, we can go forward and edit the fstab file.
::

  sudo nano /etc/fstab

.. literalinclude:: ../source/fstab

Next up, we will be making a new shell script called rpcbindboot. This script will mount the NFS automatically each time the cluster is booted.
This script will be have to be made and executed in the /etc/init.d directory.

::

  cd /etc/init.d
  sudo touch rpcbindboot
  sudo nano rpcbindboot

The file should be contain the following:

.. literalinclude:: ../source/rpcbindboot

After saving the file, we will need to execute this script.
we run the following command to do so:

::

  chmod +x rpcbindboot

Our Last step is to make sure that the nfs-common and rpcbind have the same run level as nfs-kernel-server.
We have a look into /etc/init.d/nfs-kernel-server and find its start level is **2** **3** **4** **5**. However, nfs-common and rpcbind have differenct run level.

Have a look at these files' runlevel

/etc/init.d/nfs-kernel-server

/etc/init.d/nfs-common

/etc/init.d/rpcbind

This is the runlevel of nfs-kernel-server

::

  ### BEGIN INIT INFO
  # Provides:          nfs-kernel-server
  # Required-Start:    $remote_fs nfs-common $portmap $time
  # Required-Stop:     $remote_fs nfs-common $portmap $time
  # Should-Start:      $named
  # Default-Start:     2 3 4 5
  # Default-Stop:      0 1 6
  # Short-Description: Kernel NFS server support
  # Description:       NFS is a popular protocol for file sharing across
  #                    TCP/IP networks. This service provides NFS server
  #                    functionality, which is configured via the
  #                    /etc/exports file.
  ### END INIT INFO

We need to change the Default-Start: in rpcbind and nfs-common to 2 3 4 5 as well. Again we will use nano to do this.

::

  sudo nano /etc/init.d/nfs-common
  sudo nano /etc/init.d/rpcbind

Next, we update-rc.d the changed init scripts with defaults. In order to do this we will need to remove and add them again.
::

  update-rc.d -f rpcbind remove
  update-rc.d rpcbind defaults

  update-rc.d -f nfs-common remove
  update-rc.d nfs-common defaults

  update-rc.d -f nfs-kernel-server remove
  update-rc.d nfs-kernel-server defaults

Now our NFS mounting should be complete. In order to check this we will first restart our cluster. Once the cluster is restarted we will simply create a file using the touch command in the cluster_files directory and see if it can be accessed through the worker nodes. Follow the steps below to do this:

::

  cd cluster_files
  touch foo

Now, we ssh into any worker node and see if the file 'foo' is present or not

::

  ssh node1
  cd cluster_files
  ls

We notice that the file 'foo' is in fact present. Now if we remove the file in the cluster_files directory in node1, it should automatically be removed from the head node. Let's try this:

::

  cd cluster_files
  rm foo
  exit

Now we are back to the head node. If we check the cluster_files directory on the head node now, the file 'foo' will not be there.

Congratulations, you have a working Raspberry Pi Cluster!

Broadcasting and Shutting Down
##############################

If you wish to run a specific command to all the nodes without having to type it on each node individually, we can create a simple shell script for broadcasting commands to all the nodes.

::

  cd
  sudo nano bcast.sh

.. literalinclude:: ../source/bcast.sh

We save this file and then make it executable.
::

  chmod +x bcast.sh

Now we can simply use this script by running the following command:

::

  ./bcast.sh #YourCommand

For example, "./bcast.sh date" will give you the date and time across all the nodes of the cluster.

Unfortunately, you cannot shutdown the cluster using this command because the Raspberry Pi shuts down so quickly that it cannot communicate back to the head node that it actually did shut down. To solve this you can make another similar shell script for shutting down the cluster.

::

  cd
  sudo nano shutdown_cluster

.. literalinclude:: ../source/shutdown_cluster

Again, we save this file and then make it executable.
::

  chmod +x bcast.sh

Now, to shutdown we can simply type the following:
::

  ./shutdown_cluster

Time and Date Synchronization
###############################

We can synchronize the date and time across each node of the cluster to the head node. To do this first, we will need to reconfigure timezone data on each node
To do this run the following:
::

  sudo dpkg-reconfigure tzdata

Follow the instructions to select your timezone on each node.

Next, we will need to edit the ntp.conf file on the head node which will act as our server node:

::

  sudo nano /etc/ntp.conf

Once the file is open, add the following line to the end of the file.

::

  server 127.127.1.0
  fudge 127.127.1.0 stratum 10

Now, restart NTP:
::

  sudo /etc/init.d/ntp restart

Now on each of the worker nodes, we will set the time server as the head node. To do this you simply SSH into each of the worker nodes and edit the ntp configuration file as follows:

::

  sudo nano /etc/ntp.conf

Now add the following line to the end of the file:

::

    server head iburst

Now remove the following lines from the above file by commenting out with a # character at the beginning of the line as follows:

::

  #server 0.debian.pool.ntp.org iburst
  #server 1.debian.pool.ntp.org iburst
  #server 2.debian.pool.ntp.org iburst
  #server 3.debian.pool.ntp.org iburst

Lastly, we will need to restart the NTP like we did for the head node:
::

  sudo /etc/init.d/ntp restart


Now you should have the time synchronized across all the nodes.

Helpful Ideas and Notes
##############################

* If you intend on building a cluster containing many nodes you can use a pen drive to store all the files we have edited in the head and one of the worker nodes and simply copy and replace them in the rest of the nodes making sure to change the static ip address and the hostname.

* If you intend on making multiple clusters of each of n nodes, a faster approach is to make images of each of the n nodes in one cluster using `Win32 Disk Imager`_ and flash new Micro-SD cards with these working images. The Micro-SD cards with these images can simply be inserted into a new cluster of Raspberry Pis without having to configure anything.

* The Rasbian Pixel can be used on all of the nodes of the cluster if you wish to have GUI on each one of them. We recommend that you use Rasbian Lite for the worker nodes as you may not need to use its GUI and they can be easily accessed through SSH via the head node. This will help processes run faster on the worker nodes as the processors will not need to drive the GUI on them.

* External drives are automatically mounted on Rasbian Pixel but in order to access an external drive on Rasbian Lite, you will first need to create a mount point for it and then mount it. This can be done using the following commands after you have inserted the Pen Drive into the Pi:
::

    sudo mkdir /media/usb
    sudo mount /dev/sda1 /media/usb

* If you wish to mount an external hard disk/pen drive with a higher storage capacity as your network file system it is possible to do that too. Since the head node will have GUI on it you simply insert the drive and find it's path and then edit the exports file on the head node. Since the NFS will be an external drive the exports file will be a bit different. We will have to specify the access for each of the nodes:

::

  sudo nano /etc/exports

If you had already edited the exports file as we had shown above then remove the line you had added then replace it with the following lines at the end of the file. If not, then simply add these:

::

  /media/usb node1(rw,sync,no_root_squash,no_subtree_check)
  /media/usb node2(rw,sync,no_root_squash,no_subtree_check)
  /media/usb node3(rw,sync,no_root_squash,no_subtree_check)

Note: You may need to format the external drive so that it is compatible with Linux systems before you are able to mount it.

* We recommend that you use the 4 Port HDMI and 4 Port USB switch because it makes it really convenient to switch to a different node without having to change the display HDMI or the USB jacks for the keyboard in the cluster. This is extremely helpful because we have to repeat several steps in each node of the cluster.

* The Raspberry Pi Boards act strangely when it comes to switching them on. If you want the board to connect to a display, you have to make sure that the HDMI is hooked up before you turn the power on. Otherwise you won't be able to see desktop on the screen.

* If you already have data on your Pi and software installed, it is highly recommended that you not only back up your data but also create an image of the existing version of your Pi.

* While building this cluster, the reason why we did not set up Wifi on each node is because the issue we faced where we had to register the MAC Hardware Address of each node our college gadgets' wifi portal. Hence we decided just to have the head node access to the Internet. In the long run this might not be a good idea for those who will want install software in the future as the worker nodes will not have access to Internet. If your Wireless Network allows you to simply connect to a network using a username and password then you should go ahead and connect each of the nodes to the network. You will then be able to run updates and upgrades regularly and install software anytime you want. To do this you will have to follow the same wireless setup as we did for the head node i.e. you will need to edit the wpa_supplicant.conf file. One other thing you will need to do is to have the worker nodes contain the same interfaces file as the head node but with their respective static IP addresses of course. Basically, add the wlan0 and wlan1 interface to the interfaces file.


.. toctree::
  :caption: Table of Contents
  :titlesonly:

.. comment
	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`

.. |date| date::
