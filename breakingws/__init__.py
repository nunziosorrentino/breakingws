import os
from breakingws.version import __version__

# Basic folder structure of the package

BREAKINGWS_ROOT = os.path.abspath(os.path.dirname(__file__))
BREAKINGWS_BASE = os.path.abspath(os.path.join(BREAKINGWS_ROOT, os.pardir))
BREAKINGWS_DAMA = os.path.abspath(os.path.join(BREAKINGWS_BASE, 
                                               'datamanage'))
BREAKINGWS_DATA = os.path.abspath(os.path.join(BREAKINGWS_DAMA, 'data'))
