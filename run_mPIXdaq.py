#!/usr/bin/env -S LD_LIBRARY_PATH=. python3
# on some platforms, the currnt directory needs to be in LD_LIBRARY_PATH
# so that all neccesary C-libraries are found by the pypixet interface
#  the "addition -S LD_LIBRARY_PATH=." does the trick ... 

from mpixdaq import mpixdaq
mpixdaq.run()
