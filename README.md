# BreakinGWs
project for the *cmepda* course of the University of Pisa, voted to develop an efficient convolutuonal neural network for the study of the rapid noise artifacts (i.e. glitches) in gravitational waves detectors. This file shows the most important things to do before using the package. If you have problems or doubts after reading this README, please visit the documentation by typing below on the status flag. 

**Package build status**
[![Build Status](https://travis-ci.org/nunziosorrentino/breakingws.svg?branch=master)](https://travis-ci.com/nunziosorrentino/breakingws)

**Documentation status**
[![Documentation Status](https://readthedocs.org/projects/breakingws/badge/?version=latest)](https://breakingws.readthedocs.io/en/latest/?badge=latest)

## Installation

BreakinGWs provides an user-friendly configuration of the environment needed for its usage. You should be able to execute what follows:

1. Clone BreakinGWs from its GitHub repository: 

   You can use the HTTPS
   ```bash
   $ git clone https://github.com/nunziosorrentino/breakingws.git
   ```
   Or, if you have any public SSH key, you can clone it with:
   ```bash
   $ git clone git@github.com:nunziosorrentino/breakingws.git
   ```
   First moving to the next part, be sure to stay in the first *breakingws/* directory. If you have   already cloned the package just types:
   ```bash
   $ cd breakingws
   ```
2. Create a Python3 virtual environment:

   In order to give you the best from this package, BreakinGWs must be run on the same Python environment with which has been developed. In order to ensure this, a bash file voted to this  purpose
is the one that reach the right prerequisites. Just type:
   ```bash
   $ ./create-env.sh
   ```
   If an encouraging message of success comes up, you can go the the next step.

3. Setup the environment:
   ```bash
   $ source setup.sh
   ```
   **This step must be done every time you refresh your terminal.** 


If everything has gone well, you should see something like this in your command line:
```bash
(venv-breakingws-py3) <user>@<host>:
```
Now you can use BreakinGWs package. Enjoy!

