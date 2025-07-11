""" ADVACAM pixet1.8.3 library
"""

import os
import sys

# Import version info
from ._version_info import *
# and set version
__version__ = _version_info._get_version_string()

cur_file_dir = os.path.dirname(os.path.realpath(__file__))
# add current file directory so that pypixet.so is found by python
sys.path.append(cur_file_dir)
# set current file directory as working dir to find pypixet.so dependancies
os.chdir(cur_file_dir)

# load every symbols of pypixet into upper module
from pypixet import *
