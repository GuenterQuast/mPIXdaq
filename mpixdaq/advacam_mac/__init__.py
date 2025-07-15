"""ADVACAM pixet1.8.3 library for x86_64 systems"""

import os
import sys

cur_file_dir = os.path.dirname(os.path.realpath(__file__))
# add current file directory so that pypixet.so is found by python
sys.path.append(cur_file_dir)
#  add to path so that pypixet.so finds its dependencies
os.chdir(cur_file_dir)

# load symbols of pypixet into upper module
from pypixet import *
