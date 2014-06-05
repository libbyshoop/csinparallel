#######################
Jetson TK1 Installation
#######################

Physical Set-Up
###############

- Unwrap the Jetson and attack the feet to the corners of the device
- Attach periferals, you will need an HDMI chord, an HDMI capable monitor (or an adapter), a USB mouse, a USB keyboard, and a USB hub if your mouse and keyboard aren't attached (the Jetson only has one USB port)
- To turn it on press the power button (it's the one closest of the center of the board)

Install the OS
##############

The Jetson comes with a modified version of Ubuntu 14.04 called Linux 4 Tegra or L4T.

- Log into the device

  * username: ubuntu
  * password: ubuntu

- Run the installer.sh script and reboot

.. code-block:: sh

    sudo ~/NVIDIA-INSTALLER/installer.sh
    sudo reboot

CUDA Set-up
###########

Create a NVIDIA developer account and register as a CUDA developer 
`here. <https://developer.nvidia.com/cuda-registered-developer-program>`_
Go to My Account by clicking the settings button at the top left
of NVIDIA's website and make sure that your registration 
was successful, and that your Basic Registered Developer Profile is completed.

The approval process may take a day or two. Once you're approved
download the ARMv7 L4T Linux version of the 
`CUDA ToolKit. <https://developer.nvidia.com/cuda-downloads>`_

Now run the following commands to install CUDA:

.. code-block:: sh

    cd ~/Downloads
    # install the repository meta-data
    sudo dpkg -i cuda-repo-l4t-r19.2_6.0-42_armhf.deb
    sudo apt-get update                     # update apt cache
    sudo apt-get install cuda-toolkit-6-0   # install cuda
    sudo usermod -a -G video ubuntu         # add user to the video group

Set up your environment variables by adding these lines to
~/.bashrc and running ``source ~/.bashrc``

.. code-block:: sh

    export PATH=/usr/local/cuda-6.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-6.0/lib:$LD_LIBRARY_PATH

Run ``nvcc -V`` if it prints out version information, everything was successfully installed.

Device Recovery
###############

If you need to completely reset the Jetson, connect it to 
another computer running linux (it might work on macs but it hasn't been tested) 
using the micro USB cable that comes with it.

- Download the latest
  `L4T release package <https://developer.nvidia.com/sites/default/files/akamai/mobile/files/L4T/Tegra124_Linux_R19.2.0_armhf.tbz2>`_ 
  and the 
  `sample file system <https://developer.nvidia.com/sites/default/files/akamai/mobile/files/L4T/Tegra_Linux_Sample-Root-Filesystem_R19.2.0_armhf.tbz2>`_

- Extract the files and assemble the Root File System

.. code-block:: sh

        sudo tar xpf Tegra124_Linux_R19.2.0_armhf.tbz2
        cd Linux_for_Tegra/rootfs/
        sudo tar xpf ../../Tegra_Linux_Sample-Root-Filesystem_R19.2.0_armhf.tbz2
        cd ../
        sudo ./apply_binaries.sh

- Flash the rootfs onto the system's internal eMMC

  a. Put your system into "reset recovery mode" by holding down
     the "RECOVERY" button and press "RESET" button once on the 
     main board.
  #. Ensure your Linux host system is connected to the target 
     device through the USB cable for flashing.
  #. Run this command, it will take a while:
       ``sudo ./flash.sh -S 8GiB 'jetson-tk1' mmcblk0p1``

- The board should reboot on it's own.

NVIDIA's instructions can be found
`here. <https://developer.nvidia.com/sites/default/files/akamai/mobile/files/L4T/l4t_quick_start_guide.txt>`_


