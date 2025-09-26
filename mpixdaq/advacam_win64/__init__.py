"""ADVACAM pixet1.8.4 library for windows 64bit

Note: Windows version needs local directory in library path

"""

import os
import sys

cur_file_dir = os.path.dirname(os.path.realpath(__file__))
# add current file directory so that pypixet.so is found by python
sys.path.append(cur_file_dir)
# add to path so that pypixet.so finds its dependencies
os.chdir(cur_file_dir)
