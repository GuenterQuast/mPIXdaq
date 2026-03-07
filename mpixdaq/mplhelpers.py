"""matplotlib helper functions"""

import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


#   helper classes for animated histogramming
class bhist:
    """one-dimensional histogram for animation, based on bar graph
    supports multiple classes as stacked histogram

    Args:
        * data: tuple of arrays to be histogrammed
        * bindeges: array of bin edges
        * xlabel: label for x-axis
        * ylabel: label for y axis
        * xscale: "linear" or "log" x-scale
        * yscale: "linear" or "log" y-scale
        * labels: labels for classes
        * colors: colors corresponding to labels
    """

    def __init__(self, ax=None, data=None, binedges=None, xlabel="x", ylabel="freqeuency", yscale="log", xscale="linear", labels=None, colors=None):
        # ### own implementation of one-dimensional histogram (numpy + pyplot bar) ###

        if type(data) is not type((1,)):
            print("! bhist requires a tuple as input, not ", type(data))

        self.n_classes = len(data)

        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot()
        else:
            self.ax = ax

        if labels is None:
            if self.n_classes == 1:
                labels = [None]
            else:
                labels = ["class " + str(_ic) for _ic in range(self.n_classes)]
        if colors is None:
            colors = self.n_classes * [None]

        self.bheights = []
        self.bars = []
        # plot class 1
        _bc, self.be = np.histogram(data[0], binedges)  # histogram data
        self.bheights.append(_bc)
        self.bcnt = (self.be[:-1] + self.be[1:]) / 2.0
        self.w = 0.8 * (self.be[1:] - self.be[:-1])
        ec = 'gray'
        self.bars.append(plt.bar(self.bcnt, _bc, align="center", width=self.w, color=colors[0], label=labels[0], edgecolor=ec, alpha=0.75))
        sum = _bc

        # plot other classes
        for _ic in range(1, self.n_classes):
            _bc, _be = np.histogram(data[_ic], binedges)  # histogram data
            self.bheights.append(_bc)

            self.bars.append(
                plt.bar(self.bcnt, _bc, align="center", width=self.w, color=colors[_ic], label=labels[_ic], edgecolor=ec, alpha=0.75, bottom=sum)
            )
            sum = sum + _bc

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        _mx = max(sum)
        self.maxh = _mx if _mx > 0.0 else 1000
        self.ax.set_ylim(0.75, self.maxh)
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)
        if labels[0] is not None:
            self.ax.legend(loc="upper right")

    def set(self, data):
        """set new histogram data

        Args:
           * data: heights for each bar

        Action: update pyplot bar graph
        """

        sum = np.zeros(len(self.bheights[0]))
        for _i in range(self.n_classes):
            _ic = self.n_classes - 1 - _i
            _bc, _be = np.histogram(data[_ic], self.be)  # histogram data ...
            self.bheights[_ic] = _bc
            for _b, _h in zip(self.bars[_ic], self.bheights[_ic] + sum):
                _b.set_height(_h)
            sum = sum + self.bheights[_ic]

        _mx = max(sum)
        if _mx > self.maxh:
            self.maxh = 1.2 * _mx
            self.ax.set_ylim(0.9, self.maxh)

    def add(self, data):
        """update histogram data

        Args:
            * data: heights for each bar

        Action: update pyplot bar objects
        """

        sum = np.zeros(len(self.bheights[0]))
        for _i in range(self.n_classes):
            # plot bars in reverse order
            _ic = self.n_classes - 1 - _i
            _bc, _be = np.histogram(data[_ic], self.be)  # histogram data ...
            self.bheights[_ic] = self.bheights[_ic] + _bc  # and add to existing
            for _b, _h in zip(self.bars[_ic], self.bheights[_ic] + sum):
                _b.set_height(_h)
            sum = sum + self.bheights[_ic]

        _mx = max(sum)
        if _mx > self.maxh:
            self.maxh = 1.2 * _mx
            self.ax.set_ylim(0.9, self.maxh)


class scatterplot:
    """two-dimensional scatter plot for animation, based on numpy.histogram2d
    The code supports multiple classes of data and plots a '.' in a
    corresponding color in every non-zero bin of a 2d-histogram

    Args:
        * data: tuple of pairs of coordinates (([x], [y]), ([], []), ...)
          per class to be shown
        * binedges: 2 arrays of bin edges [[bex], [bey]]
        * xlabel: label for x-axis
        * ylabel: label for y axis
        * labels: labels for classes
        * colors: colors corresponding to labels
    """

    def __init__(self, ax=None, data=None, binedges=None, xlabel="x", ylabel="y", labels=None, colors=None):
        #  own implementation of 2d scatter plot (numpy + pyplot.plot() ###

        if type(data) is not type((1,)):
            print("! scatterplot requires a tuple as input, not", type(data))

        self.n_classes = len(data)
        # initialize bins
        self.binedges = binedges
        self.bex = binedges[0]
        self.bey = binedges[1]
        self.bcntx = (self.bex[:-1] + self.bex[1:]) / 2.0
        self.bcnty = (self.bey[:-1] + self.bey[1:]) / 2.0
        # bin widths
        self.bwx = self.bex[1] - self.bex[0]
        self.bwy = self.bey[1] - self.bey[0]
        # fraction of bin width as plot off-set for classes
        self.pofx = [(_i + 1) * self.bwx / (self.n_classes + 1) - self.bwx / 2.0 for _i in range(self.n_classes)]
        self.pofy = [(_i + 1) * self.bwy / (self.n_classes + 1) - self.bwy / 2.0 for _i in range(self.n_classes)]

        self.H2d = []
        for _ic in range(self.n_classes):
            # use numpy histogram2d to creade histogram arrays, one per class
            _H2d, _bex, _bey = np.histogram2d(data[_ic][0], data[_ic][1], self.binedges)
            self.H2d.append(_H2d)

        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot()
        else:
            self.ax = ax
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        # self.ax.set_facecolor('k')

        # create initial plot
        if labels is None:
            if self.n_classes == 1:
                labels = [None]
            else:
                labels = ["class " + str(_ic) for _ic in range(self.n_classes)]
        if colors is None:
            colors = self.n_classes * [None]

        self.gr = []
        for _ic in range(self.n_classes):
            # _xy_list = np.argwhere(self.H2d[_ic] > 0)
            # _x = self.bcntx[_xy_list[:, 0]]
            # _y = self.bcntx[_xy_list[:, 1]]
            _xidx, _yidx = np.nonzero(self.H2d[_ic])
            _x = self.bcntx[_xidx]
            _y = self.bcnty[_yidx]
            (_gr,) = ax.plot(_x, _y, label=labels[_ic], color=colors[_ic], marker='o', markersize=0.5, ls='', alpha=0.75)
            self.gr.append(_gr)
        self.ax.set_xlim(self.bex[0], self.bex[-1])
        self.ax.set_ylim(self.bey[0], self.bey[-1])
        if labels[0] is not None:
            self.ax.legend(loc="upper right")

    def set(self, data):
        for _ic in range(self.n_classes):
            _H2d, _bex, _bey = np.histogram2d(data[_ic][0], data[_ic][1], self.binedges)  # numpy 2d histogram function
            self.H2d[_ic] = _H2d
            _xidx, _yidx = np.nonzero(self.H2d[_ic])
            self.gr[_ic].set(data=(self.bcntx[_xidx], self.bcnty[_yidx]))

    def add(self, data):
        """update scatter-plot data

        Args:
            * data: new (xy)-paris to be added

        Action: update pyplot line objects
        """

        for _ic in range(self.n_classes):
            _H2d, _bex, _bey = np.histogram2d(data[_ic][0], data[_ic][1], self.binedges)  # numpy 2d histogram function
            self.H2d[_ic] = self.H2d[_ic] + _H2d
            _xidx, _yidx = np.nonzero(self.H2d[_ic])
            self.gr[_ic].set(data=(self.bcntx[_xidx] + self.pofx[_ic], self.bcnty[_yidx] + self.pofy[_ic]))


# helper classes for matplotlib GUI
def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = mpl.get_backend()
    if backend == "TkAgg":
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


class controlGUI:
    """graphical user interface to control apps via multiprocessing Queue

    Args:

      - cmdQ: a multiprocessing Queue to accept commands
      - appName: name of app to be controlled
      - statQ: mp Queue to show status data
      - confdict: a configuration dictionary for buttons, format {name: [position 0-5, command]}
    """

    def __init__(self, cmdQ, appName="TestApp", statQ=None, confdict=None):
        self.cmdQ = cmdQ
        self.statQ = statQ
        self.button_dict = {'x': [5, ' ']} if confdict is None else confdict
        self.button_names = list(self.button_dict.keys())
        self.button_values = list(self.button_dict.values())

        self.mpl_active = True
        self.interval = 100  # update for timer

        mpl.rcParams['toolbar'] = 'None'
        # create a figure
        self.f = plt.figure("control Gui", figsize=(5.0, 1.0))
        move_figure(self.f, 1200, 0)
        self.f.canvas.mpl_connect("close_event", self.on_mpl_window_closed)

        self.f.subplots_adjust(left=0.025, bottom=0.025, right=0.975, top=0.975, wspace=None, hspace=0.1)
        gs = self.f.add_gridspec(nrows=7, ncols=1)
        # 1st subplot for text
        self.ax0 = self.f.add_subplot(gs[:-2, :])
        # no axes or labels
        self.ax0.xaxis.set_tick_params(labelbottom=False)
        self.ax0.yaxis.set_tick_params(labelleft=False)
        self.ax0.set_xticks([])
        self.ax0.set_yticks([])
        self.ax0.text(0.01, 0.78, "Process control:", size=12)
        self.ax0.text(0.15, 0.45, f"{appName}", color="goldenrod", size=15)
        self.status_txt = self.ax0.text(0.05, 0.1, "", color="ivory")

    # call-back functions
    def on_mpl_window_closed(self, ax):
        # active when application window is closed
        self.mpl_active = False
        self.cmdQ.put("E")
        time.sleep(0.3)
        sys.exit(0)

    def on_button_clicked(self, event):
        # find numer of the clicked button
        b_idx = self.baxes.index(event.inaxes)
        # extract sommand
        cmd = self.button_values[b_idx][1]
        # send to controlled process
        self.cmdQ.put(cmd)

    def cb_end(self, event):
        # active when application window is closed
        self.mpl_active = False
        self.cmdQ.put("E")
        time.sleep(0.3)
        sys.exit(0)

    def update(self, ax):
        """called by timer"""
        if self.statQ and not self.statQ.empty():
            self.status_txt.set_text(self.statQ.get())
        ax.figure.canvas.draw()

    def run(self):
        """run the GUI"""
        # create commad buttons
        self.baxes = []
        self.buttons = []
        for i, key in enumerate(self.button_names):
            self.baxes.append(self.f.add_axes([self.button_values[i][0] * 0.15 + 0.04, 0.05, 0.12, 0.20]))
            self.buttons.append(Button(self.baxes[-1], key, color="0.25", hovercolor="0.5"))
            self.buttons[-1].on_clicked(self.on_button_clicked)

        # timer waking up 10 times/s
        timer = self.f.canvas.new_timer(interval=self.interval)
        timer.add_callback(self.update, self.ax0)
        self.t_start = time.time()
        timer.start()

        print("*==* GUI for process control started")
        plt.show()


def run_controlGUI(*args, **kwargs):
    gui = controlGUI(*args, **kwargs)
    gui.run()
