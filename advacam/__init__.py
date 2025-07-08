""" ADVACAM pixet1.8.3 library
"""

import os
import sys

cur_file_dir = os.path.dirname(os.path.realpath(__file__))

# add current file directory so that pypixet.so is found by python
sys.path.append(cur_file_dir)

# set current file directory as working dir so that pypixet.so dependancies
# will be found by the linker (pypixet.so and its deps RPATH are set to $ORIGIN)
os.chdir(cur_file_dir)

# load every symbols of pypixet into upper module
from pypixet import *
