"""mPIXdaq: Minimalist Python Script to illustrate data acquisition and data analysis
   for the miniPIX EDU device by ADVACAM

Code for reading data from device taken from examples provided by the manufacturer,
see https://wiki.advacam.cz/wiki/Python_API

This code uses standard libraries

  - numpy
  - matplotlib,
  - scipy.cluster.DBSCAN
  - numpy.cov
  - numpy.linalg.eig

to display the pixel energy map, to cluster pixels and to determine
the cluster shapes energies.

This example is meant as a starting point for use of the miniPIX in physics lab courses,
where transparent insights concerning the input data and subsequent analysis steps are
key learning objectives.

"""

import argparse
import sys
import os
import pathlib
import gzip
import time
import numpy as np
from queue import Queue
from threading import Thread

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("dark_background")
from matplotlib.colors import LogNorm
from sklearn.cluster import DBSCAN


# function for conditional import of ADVACAM libraries
def import_pixet():
    global pypixet
    import platform

    mach = platform.machine()  # machine type
    arch = platform.architecture()  # architecture and  linker format
    if mach == 'x86_64':
        from .advacam_x86_64 import pypixet
    elif mach == 'aarch64' and arch[0] == "32bit":
        from .advacam_armhf import pypixet
    elif mach == 'aarch64' and arch[0] == "64bit":
        from .advacam_arm64 import pypixet
    # elif: ### MAC to be done
    else:
        exit(" !!! pypixet not available for architecture " + mach + arch[0])


# function for conditional import from npy_append_array !!!
def import_npy_append_array():
    global NpyAppendArray
    from npy_append_array import NpyAppendArray


#
#  main functions and classes - - - - -
#

# - handling the miniPIX EDU device


class miniPIXdaq:
    """Initialize, readout miniPIX EDU device and store data

    After initialization, data from the device is stored in a
    ring buffer and the current buffer index is sent to the
    calling process via a Queue in an infinite loop, which
    ends when data is entered in a command Queue.

    Args:

      - ac_count: number of frames to overlay
      - ac_time: acquisition time

    Queues for communication and synchronization

      - dataQ:  Queue to transfer data
      - cmsQ: command Queue

    Data structure:

       - fBuffer: ring buffer with recent frame data

    """

    def __init__(self, ac_count=10, ac_time=0.1):
        """initialize miniPIX device and set up data acquisition"""

        # no device yet
        self.dev = None

        # import python interface to ADVACAM libraries
        try:
            import_pixet()
        except Exception as e:
            print("!!! failed to import pypixet library ", str(e))
            return

        # start miniPIX software
        rc = pypixet.start()
        if rc != 0:
            print("!!! return code from pypixet.start():", rc)
            return
        if not pypixet.isrunning():
            print("!!! pipixet did not start!")
            return

        self.pixet = pypixet.pixet
        devs = self.pixet.devicesByType(self.pixet.PX_DEVTYPE_MPX2)  # miniPIX EDU uses the mediPIX 2 chip
        if len(devs) == 0:
            print("!!! no miniPIX device found")
            return
        # retrieve device parameters
        self.id = 0
        self.dev = devs[self.id]
        print("*==* found device " + self.dev.parameters().get("DeviceName").getString())
        self.npx = self.dev.width()
        # options for data acquisition
        # OPMs = ["PX_TPXMODE_MEDIPIX", "PX_TPXMODE_TOT", "PX_TPXMODE_1HIT", "PX_TPXMODE_TIMEPIX"]
        # device initialization
        pixcfg = self.dev.pixCfg()  # Create the pixels configuration object
        # enable output of pixel energies
        pixcfg.setModeAll(self.pixet.PX_TPXMODE_TOT)
        if self.dev.useCalibration(1):  # pixel values in keV
            print("!!! Could not enable device calibration")
        else:
            print("*==* running in ToT mode converted to keV")
        # parameters controlling data acquisition
        #  -  ac_count, ac_time, fileType, fileName
        #     if ac_count>1: frame data is available only from last frame
        self.ac_count = ac_count
        self.ac_time = ac_time

        # ring buffer for data collection
        self.Nbuf = 8
        self.fBuffer = np.zeros((self.Nbuf, self.npx * self.npx), dtype=np.float32)
        self.widx = 0

        # Queues for communication and synchronization
        self.dataQ = Queue(self.Nbuf)
        self.cmdQ = Queue(1)

    def device_info(self):
        pars = self.dev.parameters()
        dn = pars.get("DeviceName").getString()
        fw = pars.get("Firmware").getString()
        temp = pars.get("Temperature").getDouble()
        bias = pars.get("BiasSense").getDouble()
        frq = self.dev.timepixClock()
        print("miniPIX device info:")
        print(f"   {dn}, Firmware: {fw}")
        print(f"   Temp: {temp:.1f}, Bias: {bias:.1f}, frequency: {frq:.2f} MHz")
        print(
            f"   sensor type: {self.dev.sensorType(self.id)}"
            + f"  pitch: {self.dev.sensorPitch(self.id)} µm"
            + f"  thickness: {self.dev.sensorThickness(self.id)} µm"
            + f"  width {self.npx}  height: {self.dev.height()}"
        )
        rc, n_good, n_bad, frame = self.dev.doDigitalTest()
        print(f"   good pixels {n_good},  bad pixels {n_bad}") if rc == 0 else print("  Digital test failed")
        print(f"   acquisition time min: {self.dev.acqTimeMin()} s    max: {self.dev.acqTimeMax()} s")
        hasCalibration = self.dev.hasCalibration()
        npx = self.npx
        if hasCalibration:
            a = [0] * npx * npx
            b = [0] * npx * npx
            c = [0] * npx * npx
            t = [0] * npx * npx
            self.dev.calibrationDataAbct(self.id, a, b, c, t)
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            t = np.array(t)
            print(
                "   calibration parameters:"
                + f"  a: {a.mean():.3g} +/- {a.std():.2g}"
                + f"  b: {b.mean():.3g} +/- {b.std():.2g}"
                + f"  c: {c.mean():.3g} +/- {c.std():.2g}"
                + f"  t: {t.mean():.3g} +/- {t.std():.2g}"
            )

    def __call__(self):
        """Read *ac_count* frames with *ac_time* accumulation time each and add all up
        return data via Queue
        """
        while self.cmdQ.empty():
            rc = self.dev.doSimpleIntegralAcquisition(self.ac_count, self.ac_time, self.pixet.PX_FTYPE_AUTODETECT, "")
            if rc != 0:
                print("!!! miniPIX device readout error: ", self.dev.lastError())
                self.dataQ.put(None)
            # get frame and store in ring buffer
            self.fBuffer[self.widx, :] = self.dev.lastAcqFrameRefInc().data()
            self.dataQ.put(self.widx)
            self.widx = self.widx + 1 if self.widx < self.Nbuf - 1 else 0

    def __del__(self):
        pypixet.exit()


# - class and functions for data analysis
class frameAnalyzer:
    def __call__(self, f):
        """Analyze frame data
          - find clusters
          - compute cluster energies

        Args: a 2d-frame from the miniPIX

        Returns:
          - n_pixels: number of pixels with energy > 0
          - n_clusters: number of clusters
          - n_cpixels: number of pixels per cluster
          - circularity: circularity per cluster (0. for linear, 1. for circular)
          - cluster_energies: energy per cluster
        """

        self.total_Energy = f[f > 0].sum()  # raw pixel energy
        self.pixel_list = np.argwhere(f > 0)
        self.n_pixels = len(self.pixel_list)

        if self.n_pixels == 0:
            self.n_clusters = 0
            self.n_cpixels = np.array([])
            self.circularity = np.array([])
            self.cluster_energies = np.array([])
            self.Energy_in_clusters = 0.0
            self.E_unass = self.total_Energy
            self.np_unass = self.n_pixels
            return self.n_pixels, self.n_clusters, self.n_cpixels, self.circularity, self.cluster_energies

        # find clusters (lines,  blobs and unassigned)
        #   find clusters with points separated by an euclidian distance less than 1.5 and
        #     min. of 3 points (i.e. tow neighbours) for central points
        self.clabels = np.array(DBSCAN(eps=1.5, min_samples=2).fit(self.pixel_list).labels_)
        self.n_clusters = len(set(self.clabels)) - (1 if -1 in self.clabels else 0)

        # sum up cluster energies
        self.n_cpixels = np.zeros(self.n_clusters + 1, dtype=np.int32)
        self.cluster_energies = np.zeros(self.n_clusters + 1, dtype=np.float32)
        self.circularity = np.zeros(self.n_clusters + 1, dtype=np.float32)
        for _i, _l in enumerate(set(self.clabels)):
            pl = self.pixel_list[self.clabels == _l]
            # number of pixels in cluster
            _npix = len(pl)
            self.n_cpixels[_i] = _npix
            # check whether cluster is linear or circular (from eigenvalues of covariance matrix)
            if _npix > 2:
                # check if circular blob or a line
                evals, evecs = np.linalg.eig(np.cov(pl[:, 0], pl[:, 1]))
                self.circularity[_i] = min(evals) / max(evals)  # ratio of eigenvalues
            #                if min(evals > 0.):
            #                    area = _npix * _npix / (200 * evals[0] * evals[1])
            #                    self.circularity[_i] = 0.5 * (self.circularity[_i] + area)

            # total energy in cluster
            #        cluster_energies[_i] = f[*pixel_list[labels == _l].T].sum()  # 2d-list as index is tricky!
            self.cluster_energies[_i] = f[pl[:, 0], pl[:, 1]].sum()  # a more readable approach

        self.Energy_in_clusters = self.cluster_energies[: self.n_clusters].sum()
        # self.np_unass = self.n_cpixels[self.n_clusters]
        # self.E_unass = self.cluster_energies[self.n_clusters]

        return self.n_pixels, self.n_clusters, self.n_cpixels, self.circularity, self.cluster_energies

    def check(self):
        """check for consistency

        total Energy and Energy in clusters must match
        """
        if self.n_clusters > 0:
            # cross check: total energy in frame
            E_from_clusters = cluster_energies.sum()
            if E_from_clusters != self.total_Energy:
                print(f"!!! warning: Energy {E_from_clusters} ne.  energy from pixels {self.total_Energy}")


class miniPIXana:
    """Analysis of miniPIX frames for low-rate scenarios,
    where on-line analysis is possible and animated graphs are meaningful

    Animated graph of (overlayed) pixel images and cluster properties
    """

    def on_mpl_close(self, event):
        """call-back for matplotlib 'close_event'"""
        self.mpl_active = False

    def __init__(self, title, npx, n_overlay, circularity_cut, txt_overlay, unit):
        """initialize figure with pixel image, two histograms and a scatter plot"""
        self.title = title
        self.npx = npx
        self.n_overlay = n_overlay
        self.circularity_cut = circularity_cut
        self.txt_overlay = txt_overlay
        self.unit = unit

        # - data structure to store miniPIX frames and analysis results per frame
        self.framebuf = np.zeros((self.n_overlay, self.npx, self.npx), dtype=np.float32)
        self.i_buf = 0
        self.i_frame = 0
        # cumulative image
        self.cimage = np.zeros((self.npx, self.npx), dtype=np.float32)
        # frame summary statistics
        self.n_clusters = 0
        self.Energy = 0.0
        self.np_unass = 0
        self.E_unass = 0.0

        # set-up frame analyzer
        self.frameAna = frameAnalyzer()

        # - prepare a figure with subplots
        self.fig = plt.figure('PIX data', figsize=(11.5, 8.5), facecolor="#1f1f1f")
        self.fig.suptitle("miniPiX EDU Data Acquisition", size="xx-large", color="cornsilk")
        self.fig.canvas.mpl_connect('close_event', self.on_mpl_close)
        self.mpl_active = True
        self.fig.subplots_adjust(left=0.05, bottom=0.03, right=0.97, top=0.99, wspace=0.0, hspace=0.1)
        plt.tight_layout()
        gs = self.fig.add_gridspec(nrows=16, ncols=16)

        # - - 2d-display for pixel map
        self.axim = self.fig.add_subplot(gs[:, :-4])
        self.axim.set_title(self.title, y=0.97, size="x-large")
        self.axim.set_xlabel("# x        ", loc="right")
        self.axim.set_ylabel("# y             ", loc="top")
        # no default frame around graph
        self.axim.set_frame_on(False)
        _rect = mpl.patches.Rectangle((0, 0), 255, 255, linewidth=1, edgecolor='grey', facecolor='none')
        self.axim.add_patch(_rect)
        self.vmin = 0.5
        vmax = 500
        self.img = self.axim.imshow(np.zeros((self.npx, self.npx)), origin="lower", cmap='hot', norm=LogNorm(vmin=self.vmin, vmax=vmax))
        cbar = self.fig.colorbar(self.img, shrink=0.6, aspect=40, pad=-0.045)
        self.img.set_clim(vmin=self.vmin, vmax=vmax)
        # cbar.set_label("Energy " + unit, loc="top", labelpad=-5 )
        self.axim.arrow(146, 261.0, 110.0, 0, length_includes_head=True, width=1.5, color="b")
        self.axim.arrow(110, 261.0, -110.0, 0, length_includes_head=True, width=1.5, color="b")
        self.axim.text(115.0, 259, "14 mm")
        self.axim.text(0.05, -0.055, self.txt_overlay, transform=self.axim.transAxes, color="royalblue")
        self.im_text = self.axim.text(0.075, -0.08, "#", transform=self.axim.transAxes, color="r", alpha=0.75)

        #  - histogram of pixel energies
        self.axh1 = self.fig.add_subplot(gs[1:5, -4:])
        nbins1 = 65
        max1 = 1300
        be1 = np.linspace(0, max1, nbins1 + 1, endpoint=True)
        self.bhist1 = bhist(
            ax=self.axh1, data=([],), binedges=be1, xlabel="pixel energies " + self.unit, ylabel="", yscale="log", labels=None, colors=('r',)
        )

        # - histogram of cluster energies
        self.axh2 = self.fig.add_subplot(gs[6:10, -4:])
        nbins2 = 100
        max2 = 10000
        be2 = np.linspace(0, 10000, nbins2 + 1, endpoint=True)
        self.bhist2 = bhist(
            ax=self.axh2,
            data=([], []),
            binedges=be2,
            xlabel="cluster energies " + self.unit,
            ylabel="",
            yscale="log",
            labels=("linear", "circular"),
            colors=('yellow', 'cyan'),
        )

        # - scatter plot: cluster energies & sizes
        self.ax3 = self.fig.add_subplot(gs[11:15, -4:])
        mxx = 10000
        bex = np.linspace(0.0, mxx, 250, endpoint=True)
        mxy = 50
        bey = np.linspace(0.0, mxy, 50, endpoint=True)
        # initialize for 3 classes of ([x],[y]) pairs
        self.scpl = scatterplot(
            ax=self.ax3,
            data=(([], []), ([], []), ([], [])),
            binedges=(bex, bey),
            xlabel="cluster energies (keV)",
            ylabel="pixels per cluster",
            labels=("linear", "circular", "unassigned"),
            colors=('yellow', 'cyan', 'r'),
        )

        # show plots in interactive mode
        plt.ion()
        plt.show()

        self.dt_last_plot = 0.0
        self.t_start = time.time()

    def anaviz(self, frame2d):
        """analyze frame data and update image and plots"""
        n_pixels, n_clusters, n_cpixels, circularity, cluster_energies = self.frameAna(frame2d)

        if n_clusters > 0:
            self.Energy = cluster_energies.sum()
            self.Energy_in_clusters = cluster_energies[:n_clusters].sum()
            self.np_unass = n_cpixels[n_clusters]  # last entry is for unassigned
            self.E_unass = cluster_energies[n_clusters]
        else:
            self.Energy = frame2d[frame2d > 0].sum()  # raw pixel energy
            self.Energy_in_clusters = 0.0
            self.np_unass = n_pixels
            self.E_unass = self.Energy
        self.n_clusters = n_clusters

        # boolean indices for linear and circular objects
        is_lin = circularity[:n_clusters] <= self.circularity_cut
        is_cir = circularity[:n_clusters] > self.circularity_cut

        # update histogram 1 with pixel energies
        self.bhist1.add((frame2d[frame2d > 0],))

        # update histogram 2 with cluster energies
        if n_clusters > 0:
            self.bhist2.add((cluster_energies[:n_clusters][is_lin], cluster_energies[:n_clusters][is_cir]))

        # update scatter plot
        xlin = cluster_energies[:n_clusters][is_lin]
        ylin = n_cpixels[:n_clusters][is_lin]
        xcir = cluster_energies[:n_clusters][is_cir]
        ycir = n_cpixels[:n_clusters][is_cir]
        self.scpl.add([(xlin, ylin), (xcir, ycir), ([self.E_unass], [self.np_unass])])

    def __call__(self, frame2d, dt_alive):
        """update cumulative pixel image, analyze data and
        update histograms, scatter plot and status text
        """
        self.dt_alive = dt_alive
        self.i_frame += 1
        # subtract oldest frame ...
        self.cimage -= self.framebuf[self.i_buf]
        # ... and store new one in ring-buffer
        self.framebuf[self.i_buf, :, :] = frame2d[:, :]
        # and add actual data to cumulated values
        self.cimage += frame2d
        if self.i_buf < self.n_overlay - 1:
            self.i_buf = self.i_buf + 1
        else:
            # buffer filled, analyze data
            self.anaviz(self.cimage)
            # reset buffer index
            self.i_buf = 0

        dt_active = time.time() - self.t_start
        # update, redraw and show all subplots in figure
        if dt_active - self.dt_last_plot > 0.15:  # limit number of graphics updates
            dead_time_fraction = 1.0 - dt_alive / dt_active
            status = (
                f"#{self.i_frame}   active {dt_active:.0f}s   alive {dt_alive:.0f}s "
                + f"  clusters = {self.n_clusters:.0f} / {self.Energy:.0f}keV "
                + f"  unassigned: {self.np_unass:.0f} / {self.E_unass:.0f}keV"
                + 10 * " "
            )
            self.img.set(data=self.cimage)
            self.im_text.set_text(status)
            self.fig.canvas.start_event_loop(0.001)  # better than plt.pause(), which would steal the focus
            self.dt_last_plot = dt_active


# helper classes and functions  - - - - -
class bhist:
    """one-dimensional histogram for animation, based on bar graph
    supports multiple classes as stacked histogram

    Args:
        * data: tuple of arrays to be histogrammed
        * bindeges: array of bin edges
        * xlabel: label for x-axis
        * ylabel: label for y axis
        * yscale: "lin" or "log" scale
        * labels: labels for classes
        * colors: colors corresponding to labels
    """

    def __init__(self, ax=None, data=None, binedges=None, xlabel="x", ylabel="freqeuency", yscale="log", labels=None, colors=None):
        # ### own implementation of one-dimensional histogram (numpy + pyplot bar) ###

        if type(data) != type((1,)):
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
        self.w = 0.8 * (self.be[1] - self.be[0])
        self.bars.append(plt.bar(self.bcnt, _bc, align="center", width=self.w, facecolor=colors[0], label=labels[0], edgecolor="grey", alpha=0.75))
        sum = _bc

        # plot other classes
        for _ic in range(1, self.n_classes):
            _bc, _be = np.histogram(data[_ic], binedges)  # histogram data
            self.bheights.append(_bc)
            self.bars.append(
                plt.bar(
                    self.bcnt, _bc, align="center", width=self.w, facecolor=colors[_ic], label=labels[_ic], edgecolor="grey", alpha=0.75, bottom=sum
                )
            )
            sum = sum + _bc

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        _mx = max(sum)
        self.maxh = _mx if _mx > 0.0 else 1000
        self.ax.set_ylim(0.9, self.maxh)
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
    supports multiple classes of data, plots a '.' in the corresponding color
    in every non-zero bin of a 2d-histogram

    Args:
        * data: tuple of pairs of cordinates (([x], [y]), ([], []), ...)
          per class to be shown
        * binedges: 2 arrays of bin edges [[bex], [bey]]
        * xlabel: label for x-axis
        * ylabel: label for y axis
        * labels: labels for classes
        * colors: colors corresponding to labels
    """

    def __init__(self, ax=None, data=None, binedges=None, xlabel="x", ylabel="y", labels=None, colors=None):
        #  own implementation of 2d scatter plot (numpy + pyplot.plot() ###

        if type(data) != type((1,)):
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
            colors = n_classes * [None]

        self.gr = []
        for _ic in range(self.n_classes):
            # _xy_list = np.argwhere(self.H2d[_ic] > 0)
            # _x = self.bcntx[_xy_list[:, 0]]
            # _y = self.bcntx[_xy_list[:, 1]]
            _xidx, _yidx = np.nonzero(self.H2d[_ic])
            _x = self.bcntx[_xidx]
            _y = self.bcnty[_yidx]
            (_gr,) = ax.plot(_x, _y, label=labels[_ic], color=colors[_ic], marker='.', markersize=1, ls='', alpha=0.75)
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


class runDAQ:
    """run miniPIX data acquisition, analysis and real-time graphics

    class to handle:

        - command-line arguments
        - initialization of miniPIX device of input file
        - real-time analysis of data frames
        - animated figures to show a live view of incoming data
        - event loop controlling data acquisition, data output to file
          graphical display
    """

    def __init__(self, wd_path):
        """initialize:

        - options from command line arguments
        - miniPIX detector or optionally input from file
        - graphics display
        """

        # write to user HOME if no path given
        if wd_path is None:
            wd_path = os.getenv("HOME")
        os.chdir(wd_path)
        self.wd_path = wd_path

        # parse command line arguments
        parser = argparse.ArgumentParser(description="read, analyze and display data from miniPIX EDU device")
        parser.add_argument('-v', '--verbosity', type=int, default=1, help='verbosity level (1)')
        parser.add_argument('-o', '--overlay', type=int, default=10, help='number of frames to overlay in graph (10)')
        parser.add_argument('-a', '--acq_time', type=float, default=0.1, help='acquisition time/frame (0.1)')
        parser.add_argument('-c', '--acq_count', type=int, default=5, help='number of frames to add (5)')
        parser.add_argument('-f', '--file', type=str, default='', help='file to store frame data')
        parser.add_argument('-t', '--time', type=int, default=36000, help='run time in seconds')
        parser.add_argument('--circularity_cut', type=float, default=0.5, help='cicrularity cut')
        parser.add_argument('-r', '--readfile', type=str, default='', help='file to read frame data')
        args = parser.parse_args()
        timestamp = time.strftime('%y%m%d-%H%M', time.localtime())

        # set options
        self.verbosity = args.verbosity
        self.out_filename = args.file + '_' + timestamp + '.npy' if args.file != '' else None
        self.read_filename = args.readfile if args.readfile != '' else None
        self.acq_time = args.acq_time
        self.acq_count = args.acq_count
        self.n_overlay = args.overlay
        self.circularity_cut = args.circularity_cut
        self.run_time = args.time

        self.tot_acq_time = self.acq_count * self.acq_time
        self.integration_time = self.tot_acq_time * self.n_overlay

        print(f"\n*==* script {sys.argv[0]} executing in working directory {self.wd_path}")

        if self.out_filename is not None:
            # data recording with npy_append_array()
            import_npy_append_array()

        # try to load pypixet library and connect to miniPIX
        if self.read_filename is None:
            print(f"     * overlaying {self.n_overlay} frames with {self.tot_acq_time} s")
            print(f"     * readout {self.acq_count} x {self.acq_time} s")
            # initialize data acquisition object
            self.daq = miniPIXdaq(self.acq_count, self.acq_time)
            if self.daq.dev is None:
                _a = input("  Problem with miniPIX device - read data from file ? (y/n) > ")
                if _a in {'y', 'Y', 'j', 'J'}:
                    path = os.path.dirname(os.path.realpath(__file__)) + '/'
                    self.read_filename = path + "data/BlackForestStone.npy.gz"
                else:
                    exit("Exiting")
            else:  # library and device are ok
                if self.verbosity > 1:
                    self.show_DeviceInfo = True
                    self.daq.device_info()
                self.npx = self.daq.npx
                self.unit = "(keV)" if self.daq.dev.isUsingCalibration() else "ToT (µs)"
                self.title = "pixel energy map " + self.unit
        #  end device initialization ---

        if self.read_filename is not None:
            # read from file if requested
            print("data from file " + self.read_filename)
            suffix = pathlib.Path(self.read_filename).suffix
            if suffix == ".gz":
                f = gzip.GzipFile(self.read_filename)
                self.fdata = np.load(f)
            elif suffix == ".npy":
                self.fdata = np.load(self.read_filename, mmap_mode="r")
            else:
                exit(" Exit - unknown file extension " + suffix)
            # assume data is 256x256 pixels in keV per pixel
            shape = self.fdata.shape
            if len(shape) < 3 or shape[1] != 256:
                exit(f"unexpected shape {shape} of array, expected 256x256")
            elif shape[1] != 256:
                exit(f"unexpected shape {shape} of array, expected 256x256")
            self.n_frames_in_file = shape[0]
            self.npx = shape[1]
            self.unit = "(keV)"
            self.title = "pixel energy map from file (keV)"
            print(f" found {self.n_frames_in_file} pixel frames in file")

        # finally, initialize analyis and figures
        if self.read_filename is None:
            txt_overlay = f"integration time {int(self.integration_time)}s"
        else:
            txt_overlay = f"sum of {int(self.n_overlay)} frames"
        self.mpixana = miniPIXana(self.title, self.npx, self.n_overlay, self.circularity_cut, txt_overlay, self.unit)

    def __call__(self):
        """run daq loop"""

        # set up daq
        dt_alive = 0.0
        dt_active = 0.0
        frame2d = np.zeros((self.npx, self.npx), dtype=np.float32)
        i_frame = 0
        # start daq as a Thread
        if self.read_filename is None:
            Thread(target=self.daq, daemon=True).start()
        # start daq loop
        print("\n" + 15 * ' ' + "\033[36m type <cntrl C> to end" + "\033[31m", end='\r')
        try:
            while dt_active < self.run_time and self.mpixana.mpl_active:
                if self.read_filename is None:
                    #                    frame2d[:, :] = np.array(self.daq.dataQ.get()).reshape((self.npx, self.npx))
                    _idx = self.daq.dataQ.get()
                    # print(self.daq.dataQ.qsize())
                    # data as 2d pixel array
                    frame2d[:, :] = self.daq.fBuffer[_idx].reshape(self.npx, self.npx)
                    dt_alive += self.acq_count * self.acq_time
                    i_frame += 1
                else:  # from file
                    i_frame += 1
                    if i_frame > self.n_frames_in_file:
                        break
                    frame2d = self.fdata[i_frame - 1]
                    ##!time.sleep(1.0)
                    time.sleep(0.1)

                # write frame to file ?
                if self.out_filename is not None:
                    with NpyAppendArray(out_filename) as npa:
                        npa.append(np.array([frame2d]))

                # real-time analysis and animated visualization
                self.mpixana(frame2d, dt_alive)

                # heart-beat for console
                print(f"  #{i_frame}", end="\r")

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print("Excpetion in daq loop: ", str(e))

        finally:
            # end daq loop
            if self.read_filename is None:
                self.daq.cmdQ.put("e")
            if self.mpixana.mpl_active:
                _a = input("\033[36m\n" + 20 * ' ' + " type <ret> to close window --> ")
                print("\033[0m")
            else:
                print("\33[0m\n" + 20 * ' ' + " Window closed, ending \n")
            if self.read_filename is None:
                pypixet.exit()


if __name__ == "__main__":  # -  - - - - - - - - - -
    rD = runDAQ()
    rD.run()
