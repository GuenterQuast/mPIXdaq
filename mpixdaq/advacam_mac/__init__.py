"""ADVACAM pixet1.8.4 library for mac"""

import os
import sys

cur_file_dir = os.path.dirname(os.path.realpath(__file__))
# add current file directory so that pypixet.so is found by python
sys.path.append(cur_file_dir)
os.chdir(cur_file_dir)
