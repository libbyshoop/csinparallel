.. Hadoop LastFM analysis documentation master file, created by
   sphinx-quickstart on Wed Jul 30 15:34:11 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hadoop LastFM Analysis 
========================

This module demonstrates how hadoop and WMR can be used to
analize the lastFM million song dataset. It encorporates several
advanced hadoop techniques such as job chaining and multiple
input. Students should know how to use the WMR hadoop interface
before beginning this module. 

The dataset was obtained from Columbia University's 
`LabROSA <http://labrosa.ee.columbia.edu/millionsong/lastfm>`_.
However it has been converted into a format that is easier to work
with on WMR. The edited dataset is also much smaller since it doesn't
include the audio analysis information. If you would like the smaller
dataset for your own WMR cluster please contact JLyman@macalester.edu

.. toctree::
    :maxdepth: 1

    0-Introduction/Introduction
    1-Keys/Keys
    2-Challenges/Challenges

.. comment:
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
