Configuring the Networking on a Jetson Cluster
==============================================

1. (Optional but recomended) Install Cluster SSH
------------------------------------------------- 

This step isn't necessary but will save you alot of time configuring and 
maintaining the cluster. Clusterssh allows you to control multiple ssh 
terminals at once. Since ARM binaries aren't available in the repositories, we need 
to download the sources and compile them. Obtain the sources for the 
dependencies from the following web pages: http://packages.ubuntu.com/trusty/perl-tk
http://packages.ubuntu.com/trusty/libx11-protocol-perl .
Untar and cd to each directory and type:
::

  $ perl Makefile.PL
  $ make
  $ sudo make install

Obtain the source code for cluserssh from: http://sourceforge.net/projects/clusterssh/
Untar and cd to the directory and type:
::

  ./configure
  make
  sudo make install

Now you can run:
::

  cssh hostname1 hostname2 … #controls multiple ssh terminal at once

2. Configure the Internal Network on the Cluster
------------------------------------------------
 
We assume that eth0 is the interface of the internal network of the cluster and 
that the head node has an additional interface, eth1, with internet access.

First we configure static ip for the cluster's internal network. On the head node run:
::

  $sudo vim /etc/network/interfaces

And then add the following:
::

  auto eth0
  iface eth0 inet static
  address 192.168.0.101
  netmask 255.255.255.0
  gateway 192.168.0.101

Do the same on the worker except for incrementing the address field for each 
worker (ie the line for the first worker should be "address 192.168.0.102").
Restart the worker nodes. The internal network should now be working so you 
can use Cluster SSH to configure the workers all at once from now on. 

3. Configure head node to act as a gateway
------------------------------------------

In order we the worker nodes to have internet, we need to configure the head 
node to act as a gateway. For more details, see the ubuntu internet gateway 
section of the following guide:
https://help.ubuntu.com/community/Internet/ConnectionSharing

First we enable nat on the head node:
::

  sudo iptables -A FORWARD -o eth1 -i eth0 -s 192.168.0.0/24 -m conntrack --ctstate NEW -j ACCEPT
  sudo iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
  sudo iptables -t nat -F POSTROUTING
  sudo iptables -t nat -A POSTROUTING -o eth1 -j MASQUERADE

The first rule allows forwarded packets (initial ones). The second rule allows forwarding of established connection packets (and those related to ones that started). The third rule does the NAT.
We save these settings with:
::

  sudo iptables-save | sudo tee /etc/iptables.sav

And then add this line to /etc/rc.local:
::

  iptables-restore < /etc/iptables.sav

To enabled packet forwarding, uncomment the following line in /etc/sysctl.conf:
::

  #net.ipv4.ip_forward=1

Then reboot the head node to the settings take effect:
::

  sudo reboot

We need to configure DNS servers for the workers to use.
Add the following lines to /etc/resolv.conf on each node:
::

  nameserver 8.8.8.8
  nameserver 8.8.4.4

4. Edit the Hosts File
----------------------

Add names for your cluster nodes into the host file (use cluster ssh to do it across the network)
::

  $ sudo vim /etc/hosts

Yours should look something like this:
::

  127.0.0.1 localhost 
  127.0.0.1 tegra-ubuntu
  192.168.0.101 tegra1
  192.168.0.102 tegra2
  ...

The worker nodes should now have internet.
