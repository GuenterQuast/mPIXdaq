"""ADVACAM pixet1.8.3 library for armhf

Note: arm version also depends on library zest.so, which
 is only found by linker if the current directory is in $LD_LIBRARY_PATH:
 export LD_LIBRARY_PATH='.' before starting the python script

"""

import os
import sys

cur_file_dir = os.path.dirname(os.path.realpath(__file__))
# add current file directory so that pypixet.so is found by python
sys.path.append(cur_file_dir)
# add to path so that pypixet.so finds its dependencies
os.chdir(cur_file_dir)

# load every symbols of pypixet into upper module
from pypixet import *
