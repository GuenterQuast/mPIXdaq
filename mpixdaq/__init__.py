"""
mPIXdaq: Minimalist Python Script to illustrate data acquisition and
data analysis for the miniPIX EDU device by ADVACAM
"""

import os
import sys

# Import version info
from ._version_info import _get_version_string 

# import of package modules (needed for Python 3.7 under Windows)
from .mpixdaq import *

# and set version
__version__ = _get_version_string()
