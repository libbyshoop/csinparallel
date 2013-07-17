.. comment
   Much of this text is plagiarized from the Intel page describing the architecture, which is Copyright 2012 Intel Corporation.

Intel Phi Architecture
======================

Basic Architecture
------------------

The Intel Xeon Phi coprocessor is primarily composed of processing cores connected by a very high bandwitch interconnect. Each core comes complete with a private L2 cache that is kept fully coherent by a global-distributed tag directory. Also, memory controllers and client logic for the PCI bus are also connected to the ring interconnect. This is shown below, with the memory controllers labelled "GDDR MC".


.. image:: diagram.jpg


Vector Processing Unit
----------------------

Each of the Xeon Phi coprocessor's cores has a vector processing unit (VPU). The VPU features a vector ALU that can execute 16 single precision or 8 double-precision operations per cycle, and also provides support for integers. The VPU also has an Extended Math Unit that can execute transcendental operations such as reciprocal, square root, and log, thereby allowing these operations to be executed in a vector fashion with high bandwidth. The EMU operates by calculating polynomial approximations of these functions.

The Interconnect
----------------

The interconnect which connects all the cores together is implemented as a bidirectional ring, one which can convey information in two directions. Each direction comprises three independent rings, as illustrated in the image below:

.. image:: interconnect.jpg

The first and largest independent ring is the data block ring, labelled "Data" in the image above. It is 64 bytes wide to support the high bandwidth required by the large number of cores. The next ring, labelled "Command and Address," is much smaller and is used to send read/write commands and memory addresses. Finally, the smallest and least expensive ring is the acknowledgement ring, which sends flow control and coherence messages.

Cluster on a Chip
-----------------

Each core tries to keep the memory it uses on its L2 cache. When it misses, however, an address request is sent on the address ring to the tag directories which, somewhat like the Hadoop namenode, determines whether the address is located in another core's L2 cache. In that case, a forwarding request is sent to that core's L2 over the address ring and the requested block is subsequently sent on the data block ring. If the requested data is not found in any caches, a memory address is sent from the tag directory to the memory controller.
