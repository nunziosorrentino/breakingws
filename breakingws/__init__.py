import os
from breakingws.version import __version__

# Basic folder structure of the package

BREAKINGWS_ROOT = os.path.abspath(os.path.dirname(__file__))
BREAKINGWS_BASE = os.path.abspath(os.path.join(BREAKINGWS_ROOT, os.pardir))
BREAKINGWS_DAMA = os.path.abspath(os.path.join(BREAKINGWS_BASE, 
                                               'datamanage'))
BREAKINGWS_DAMA_R = os.path.relpath(BREAKINGWS_DAMA).replace('..',
                                    './breakingws')
BREAKINGWS_DAMA_RR = os.path.relpath(BREAKINGWS_DAMA).replace('..',
                                    '../breakingws')                                               
BREAKINGWS_DATA = os.path.abspath(os.path.join(BREAKINGWS_DAMA, 'data'))
BREAKINGWS_DATA_R = os.path.relpath(BREAKINGWS_DATA).replace('..',
                                    './breakingws')
BREAKINGWS_DATA_RR = os.path.relpath(BREAKINGWS_DATA).replace('..',
                                    '../breakingws')
BREAKINGWS_CNN = os.path.abspath(os.path.join(BREAKINGWS_BASE, 'cnn'))
