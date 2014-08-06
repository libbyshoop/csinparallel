Installing and Configuring Cuda
===============================


1. Register a Nvidia Developer Account
--------------------------------------

Create a nvidia developer account and register as a cuda developer. The approval process takes a while.
You can go to https://developer.nvidia.com/jetson-tk1-support and double check on your ‘my account’ page 
to see that the registration was succesful,

2. Download and Install Cuda
----------------------------

Download the cuda toolkit from this site https://developer.nvidia.com/jetson-tk1-support. You can just download it on the head node and then use ‘python -m SimpleHTTPServer’ to serve contents of the current directory on a HTTP server. Then download it onto each of the nodes with
:: 

    wget 192.168.0.101:8000/cuda-repo-l4t-r19.2_6.0-42_armhf.deb

Install on each machine with
::
    sudo dpkg -i cuda-repo-l4t-r19.2_6.0-42_armhf.deb   #install the repository meta data
    sudo apt-get install cuda-toolkit-6-0               #install cuda
    #add user to the video group
    sudo usermod -a -G video ubuntu                     #(change ubuntu to your username if you changed it)

