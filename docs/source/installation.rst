.. _installation:

Installation
============

Once cloned Breakingws on your local host, just follows these steps in 
order to properly use the package tools.

Prerequisites
-------------

BreakinGWs package is based on the Python3.6_ scripting language and Keras_
machine learning tools. In order to correcly use its capabilities, make 
sure your Python version has been properly installed. Anyway you can find 
your prefered installation mode (e.g., the Python installation provided by 
your GNU/Linux distribution) and then check out your Python version through:

.. _Python3.6: https://www.python.org/downloads/release/python-360/

.. _Keras: https://keras.io/api/

.. code-block:: bash

    $ python3 --version

Then verifies that it is Python3.6!


Create and Set the Environment
------------------------------
Once cloned the BreakinGWs repository from GitHub, follows these steps in
order to set the correct environment for the tools usage.

First of all, let's move to the package location:

.. code-block:: bash

    $ cd breakingws/

In order to give you the best from this package, BreakinGWs must be run on 
the same Python environment with which has been developed. 
In order to ensure this, a bash file voted to this  purpose is the one that
reach the right prerequisites. Just type:

.. code-block:: bash

    $ ./create-env.sh


If an encouraging message of success comes up, you can go the the next 
step.

Now you can setup the environment:

.. code-block:: bash

    $ source setup.sh

**This step must be done every time you refresh your terminal.** 

If everything has gone well, you should see something like this in your 
command line:

.. code-block:: bash

    (venv-breakingws-py3) <user>@<host>:

