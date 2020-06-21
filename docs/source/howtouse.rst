.. _howtouse:

How to Use
==========

Get start with BreakinGWs. In *breakingws/datamanage/data/* there are 400
images (circles, rectagles, lines and ellipses in low resolution and 
equallt distributed) that can be used to train your first CNN with 
supervised learning. If you you want more images to start with (e.g. 1000
per class), use *genimages.py* tool:

.. code-block:: bash

    $ genimages.py -n 100
    
If an encouraging message rises up, your new data set has been modified.

Run your first CNN
------------------

.. image:: figures/imma_loss.png

.. image:: figures/imma_acc.png


