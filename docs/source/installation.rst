.. _installation:

Installation
============

Prerequisites
-------------

BreakinGWs package is based on the Python3.6_ scripting language. In order to correcly use its capabilities, make sure your Python version has been
properly installed. Anyway you can find your prefered installation mode (e.g., the Python installation 
provided by your GNU/Linux distribution), then check out your Python version through:

.. _Python3.6: https://www.python.org/downloads/release/python-360/

.. code-block:: bash

    python3 --version

Then verifies that it is Python3.6!


Create and set the environment
------------------------------

In order to give you the best from this package, BreakinGWs must be run on the same Python environment with which has been developed. In order to ensure this, a bash file voted to this  purpose is the one that reach the right prerequisites. Just type:

```bash
$ ./create-env.sh
```
If an encouraging message of success comes up, you can go the the next step.

Now you can setup the environment:
```bash
$ source setup.sh
```
**This step must be done every time you refresh your terminal.** 

If everything has gone well, you should see something like this in your command line:
```bash
(venv-breakingws-py3) <user>@<host>:
```
