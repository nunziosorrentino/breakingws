.. _overview:

Overview
========

The tools provided by BreakinGWs can generate simple images data
sets (circles, rectangles, ellipses and lines), in order to start using it
with the proposed CNN algorithms:

* ``genimages.py``: produce a set of images in *breakingws/datamange/data* folder;
* ``runcnn.py``: mean tool of the package, apply the CNN algorithm to a set of images with the relative labels, in order to predict the class of new images. 

BreakinGWs provides two models for the CNN. Each model contains a series of blocks (according to the desired complexity), with the convolutional layers that progressively 
increase their number of filters:

* ``breakingws/cnn/imma.py``: this script represents a non-complex model for training, validation and test of low resolution and not detailed imaged; 
* ``breakingws/cnn/glitcha.py``: this script represents a complex model for training, validation and test of glitches spectrograms provided by GravitySpy.

These models can be used in this way: 

.. code-block:: bash
   
   $ runcnn.py -m imma

or

.. code-block:: bash
   
   $ runcnn.py -m glitcha

The options of these tools are explained in the :ref:`how to use <howtouse>`
guide. The methods of the package are documented :ref:`here <api>`.

.. warning::
    These tools works well just if launched from *breakingws/bin* directory.
    This bug will be fixed in next versions!
