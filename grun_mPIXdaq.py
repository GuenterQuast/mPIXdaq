#!/usr/bin/env python
#
# script grun_mPIXdaq.py
#  GUI to run mpixdaq with data acquisition, on-line analysis and visualization
#  of pixel frames and histogramming

# for argument setting via
import argparse
from mpixdaq.argparse_tk_gui import ArgparseGUI
from mpixdaq.mpixdaq import runDAQ
import sys
import shlex
import subprocess


def main():
    parser = runDAQ.argparser()
    script_name = "run_mPIXdaq.py"
    title = "mPIXdaq starter"

    #    clbk = None
    def clbk(args):
        arg_lst = [script_name] + args
        cmd_str = " ".join(shlex.quote(a) for a in arg_lst)
        print("Starte: ", cmd_str)
        print(flush=True)
        proc = subprocess.run(arg_lst)

    gui = ArgparseGUI(
        parser,
        run_callback=clbk,
        script_path=None,
        title=title,
        capture_output=False,
        theme=None,
    )
    gui.mainloop()


if __name__ == "__main__":  # -----------------------------------------------
    main()
