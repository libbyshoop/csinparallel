**********************************
Preparing your machine for the lab
**********************************

**Carry the steps in this section before the lab, if possible, on a laptop you can bring with you to the lab session.** 

The Cisco VPN client software is free, easy to install, easy to use, and does not interfere with your computer except while you are connected to the MTL. Here are download links (install before the lab if possible):

* `Windows (2000,XP,Vista,7)`_

  Download the from this link and run the executable to install the Cisco VPN client.

* `Mac OS 10.4 and later`_

  Download from this link, which creates a pseudodisk on your desktop, then open that pseudodisk and follow the instructions. (You can delete the pseudodisk when you're done.)

  You can use a UNIX terminal window to access ssh and scp.

*  For Linux and other versions of Windows and Macintosh, you can see the `University of Ghent page`_ that these downloads were obtained from. Choose among the "Cisco VPN clients without config file".

Once you have installed the Cisco VPN client (and have ssh and scp installed), configure the Cisco VPN client to access the multicore testing lab, as follows.

1. Start up the Cisco VPN client.

  .. note:: This is installed as a software application. *Don't* look for it under system network connection options, like other VPN systems.

2. Create a new connection, with the following connection information:

  * *Connection entry:* Choose a name, perhaps "Intel MTL VPN"
  * *Description:* Optional
  * *Host:* **192.55.51.80**
  * Select *Group Authentication* (probably the default)
  * *Name* is **VPN2**
  * *Password* is sent to you separately.

  Save this connection to finish creating it.

3. Now try connecting on your new VPN connection entry. 

  * If this succeeds, you will find that *none of your usual network services work*. For example, your browser won't be able to find any pages (thus, you'll have to use a different machine while you're connected to the MTL if you need to access network services).

  * If your new VPN connection fails, recheck the settings you entered, or seek help.

4. Finally, disconnect from your new VPN connection entry. This will give you your usual network capabilities back, etc. 

.. _`Windows (2000,XP,Vista,7)`: http://www.cs.stolaf.edu/pub/vpnclient-win-msi-5.0.06.0160-k9.exe

.. _`Mac OS 10.4 and later`: http://helpdesk.ugent.be/vpn/download/vpnclient-darwin-4.9.01.0080-universal-k9-5-10.dmg

.. _`University of Ghent page`: http://helpdesk.ugent.be/vpn/en/akkoord.php
