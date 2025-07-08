#!/usr/bin/env python3

# try   -S LD_LIBRARY_PATH=".:libs"
#    !!! the -S option includes the script directory in the search path for libraries

"""miniPIXdaq: Minimalist Python Script to illustrate data acquisition and data analysis
   for the miniPIX EDU device

Code for reading data from device taken from examples provided by the manufacturer,
see https://wiki.advacam.cz/wiki/Python_API

This example uses standard libraries

  - numpy
  - matplotlib,
  - scipy.cluster.DBSCAN
  - numpy.cov
  - numpy.linalg.eig

to display the pixel energy map, cluster pixels and determine the cluster energies.

This example is meant as a starting point for use of the miniPIX in physics lab courses,
where transparent insights concerning the input data and subsequent analysis steps are
key learning objectives.

"""

import pypixet  # the python API for miniPIX

import argparse
import sys
import pathlib
import gzip
import time
import numpy as np
from queue import Queue
from threading import Thread

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.cluster import DBSCAN


# function for conditional import from npy_append_array !!!
def import_npy_append_array():
    global NpyAppendArray
    from npy_append_array import NpyAppendArray


#
#  functions and classes - - - - -
#
# - handling the miniPIX EDU device


class mPIXdaq:
    """Initialize and readout miniPIX EDU device

    Args:
      - ac_count: number of frames to overlay
      - ac_time: acquisition time
      - dataQ:  Queue to receive data
      - cmsQ: command Queue
    """

    def __init__(self, ac_count=10, ac_time=0.1, dataQ=None, cmdQ=None):
        """initialize miniPIX device and set up data acquisition"""
        # start miniPIX software
        pypixet.start()
        self.pixet = pypixet.pixet
        devs = self.pixet.devicesByType(self.pixet.PX_DEVTYPE_MPX2)  # miniPIX EDU uses the mediPIX 2 chip
        if len(devs) == 0:
            self.dev = None
            return
        # retrieve device parameters
        self.id = 0
        self.dev = devs[0]
        pars = self.dev.parameters()
        dn = pars.get("DeviceName").getString()
        fw = pars.get("Firmware").getString()
        temp = pars.get("Temperature").getDouble()
        bias = pars.get("BiasSense").getDouble()
        frq = self.dev.timepixClock()
        print("miniPIX device found:")
        print(f"   {dn}, Firmware: {fw}\n   Temp: {temp:.1f}, Bias: {bias:.1f}, frequency: {frq:.2f} MHz")
        self.npx = self.dev.width()
        # options for data acquisition
        # OPMs = ["PX_TPXMODE_MEDIPIX", "PX_TPXMODE_TOT", "PX_TPXMODE_1HIT", "PX_TPXMODE_TIMEPIX"]
        # device initialization
        pixcfg = self.dev.pixCfg()  # Create the pixels configuration object
        pixcfg.setModeAll(pixet.PX_TPXMODE_TOT)
        self.dev.useCalibration(1)  # pixel values in keV

        # parameters controlling data acquisition
        #  -  ac_count, ac_time, fileType, fileName
        #     if ac_count>1: frame data is available only from last frame
        self.count = ac_count
        self.time = ac_time
        # Queues for communication
        self.dataQ = dataQ
        self.cmdQ = cmdQ

    def device_info(self):
        npx = self.npx
        print(" Detailed device info:")
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
                f"   calibration parameters:"
                + f"  a: {a.mean():.3g} +/- {a.std():.2g}"
                + f"  b: {b.mean():.3g} +/- {b.std():.2g}"
                + f"  c: {c.mean():.3g} +/- {c.std():.2g}"
                + f"  t: {t.mean():.3g} +/- {t.std():.2g}"
            )

    def __call__(self):
        """Read *count* frames with *ac_time* accumulation time each and add all up
        return data via Queue
        """
        while cmdQ.empty():
            rc = self.dev.doSimpleIntegralAcquisition(self.count, self.time, self.pixet.PX_FTYPE_AUTODETECT, "")
            self.dataQ.put(self.dev.lastAcqFrameRefInc().data() if rc == 0 else None)

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
            self.n_cpixels = None
            self.circularity = None
            self.cluster_energies = None
            self.Energy_in_clusters = 0.0
            self.E_unass = self.total_Energy
            self.np_unass = self.n_pixels
            return self.n_pixels, self.n_clusters, self.n_cpixels, self.circularity, self.cluster_energies

        # find clusters (lines,  blobs and unassigned)
        #   find clusters with points separated by an euclidian distance less than 1.5 and
        #     min. of 3 points (i.e. tow neighbours) for central points
        self.clabels = np.array(DBSCAN(eps=1.5, min_samples=3).fit(self.pixel_list).labels_)
        self.n_clusters = len(set(self.clabels)) - (1 if -1 in self.clabels else 0)

        # sum up cluster energies
        self.n_cpixels = np.zeros(self.n_clusters + 1, dtype=np.int32)
        self.cluster_energies = np.zeros(self.n_clusters + 1)
        self.circularity = np.zeros(self.n_clusters + 1)
        for _i, _l in enumerate(set(self.clabels)):
            pl = self.pixel_list[self.clabels == _l]
            # number of pixels in cluster
            self.n_cpixels[_i] = len(pl)
            # check whether cluster is linear or circular (from eigenvalues of covariance matrix)
            if len(pl) > 2:
                evals, evecs = np.linalg.eig(np.cov(pl[:, 0], pl[:, 1]))
                self.circularity[_i] = min(evals) / max(evals)
            # total energy in cluster
            #        cluster_energies[_i] = f[*pixel_list[labels == _l].T].sum()  # 2d-list as index is tricky!
            self.cluster_energies[_i] = f[pl[:, 0], pl[:, 1]].sum()  # a more readable approach

        self.Energy_in_clusters = self.cluster_energies[: self.n_clusters].sum()
        self.np_unass = self.n_cpixels[self.n_clusters]
        self.E_unass = self.cluster_energies[self.n_clusters]

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


# helper classes and functions  - - - - -
class bhist:
    """one-dimensional histogram for animation, based on bar graph

    Args:
        * data: array containing float values to be histogrammed
        * bins: array of bin edges
        * xlabel: label for x-axis
        * ylabel: label for y axis
        * yscale: "lin" or "log" scale
    """

    def __init__(self, ax=None, data=None, bins=50, xlabel="x", ylabel="freqeuency", yscale="log"):
        # ### own implementation of one-dimensional histogram (numpy + pyplot bar) ###

        self.bc, self.be = np.histogram(data if data is not None else [], bins)  # histogram data
        self.bheights = self.bc
        self.bcnt = (self.be[:-1] + self.be[1:]) / 2.0
        self.w = 0.8 * (self.be[1] - self.be[0])
        if ax is None:
            self.bars = plt.bar(
                self.bcnt, self.bc, align="center", width=self.w, facecolor="b", edgecolor="grey", alpha=0.5
            )
            self.ax = plt.gca()
        else:
            self.ax = ax
            self.bars = ax.bar(
                self.bcnt, self.bc, align="center", width=self.w, facecolor="b", edgecolor="grey", alpha=0.5
            )

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        _mx = max(self.bheights)
        self.maxh = _mx if _mx > 0.0 else 1000
        self.ax.set_ylim(0.9, self.maxh)
        self.ax.set_yscale(yscale)

    def set(self, data):
        """set new histogram data

        Args:
           * data: heights for each bar

        Action: update pyplot bar graph
        """
        bc, be = np.histogram(data, self.be)  # histogram data
        for _b, _h in zip(self.bars, bc):
            _b.set_height(_h)
        _mx = max(bc)
        if _mx > self.maxh:
            self.maxh = 1.2 * _mx
            self.ax.set_ylim(0.9, self.maxh)

    def add(self, data):
        """update histogram data

        Args:
            * data: heights for each bar

        Action: update pyplot bar graph
        """
        bc, be = np.histogram(data, self.be)  # histogram data ...
        self.bheights = self.bheights + bc  # and add to existing
        for _b, _h in zip(self.bars, self.bheights):
            _b.set_height(_h)

        _mx = max(self.bheights)
        if _mx > self.maxh:
            self.maxh = 1.2 * _mx
            self.ax.set_ylim(0.9, self.maxh)


#
# main - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

# parse command line arguments
# ------
parser = argparse.ArgumentParser(description="read, analyze and display data from miniPIX EDU device")
parser.add_argument('-v', '--verbosity', type=int, default=1, help='verbosity level (1)')
parser.add_argument('-o', '--overlay', type=int, default=25, help='number of frames to overlay in graph (25)')
parser.add_argument('-a', '--acq_time', type=float, default=0.2, help='acquisition time/frame (0.2)')
parser.add_argument('-c', '--acq_count', type=int, default=1, help='number of frames to add (1)')
parser.add_argument('-f', '--file', type=str, default='', help='file to store frame data')
parser.add_argument('-t', '--time', type=int, default=36000, help='run time in seconds')
parser.add_argument('--circularity_cut', type=float, default=0.5, help='cicrularity cut')
parser.add_argument('-r', '--readfile', type=str, default='', help='file to read frame data')
args = parser.parse_args()
timestamp = time.strftime('%y%m%d-%H%M', time.localtime())

verbosity = args.verbosity
filename = args.file + '_' + timestamp + '.npy' if args.file != '' else None
read_filename = args.readfile if args.readfile != '' else None
acq_time = args.acq_time
acq_count = args.acq_count
n_overlay = args.overlay
circularity_cut = args.circularity_cut
run_time = args.time

integration_time = acq_count * acq_time * n_overlay

print(f'\n*==* script {sys.argv[0]} executing')
print("       type <cntrl C> to end\n")

if filename is not None:
    # data recording with npy_append_array()
    import_npy_append_array()

if read_filename is None:
    # set up Queues for communication with daq process
    maxsize = 16
    dataQ = Queue(maxsize)
    cmdQ = Queue(1)

    # initialize data acquisition object
    daq = mPIXdaq(acq_count, acq_time, dataQ, cmdQ)
    if daq.dev == None:
        _a = input("No devices found - read data from file (y/n) > ")
        if _a in {'y', 'Y', 'j', 'J'}:
            read_filename = "BlackForestStone.npy.gz"
        else:
            exit("Exit - no devices found")
    else:
        if verbosity > 0:
            show_DeviceInfo = True
            daq.device_info()
        npx = daq.npx
        unit = "(keV)" if daq.dev.isUsingCalibration() else "ToT (µs)"
        title = "pixel energy map " + unit
#   end device initialization ---

if read_filename is not None:
    # read from file if requested
    print("data from file " + read_filename)
    suffix = pathlib.Path(read_filename).suffix
    if suffix == ".gz":
        f = gzip.GzipFile(read_filename)
        data = np.load(f)
    elif suffix == ".npy":
        data = np.load(read_filename, mmap_mode="r")
    else:
        exit(" Exit - unknown file extension " + suffix)
    # assume data is 256x256 pixels in keV per pixel
    shape = data.shape
    if shape[1] != 256 or shape[2] != 256:
        exit(f"unexpected shape {shape} of array, expected 256x256")
    npx = shape[1]
    unit = "(keV)"
    title = "pixel energy map from file (keV)"
    n_mx = shape[0]
    print(f" found {n_mx} pixel frames in file")


# - data structure to store miniPIX frames and analysis results per frame
framebuf = np.zeros((n_overlay, npx, npx))
n_clusters_buf = np.zeros(n_overlay)
energy_buf = np.zeros(n_overlay)
unassigned_buf = np.zeros(n_overlay)
o_n_clusters = 0
o_energy = 0.
o_unassigned = 0.
_bidx = 0

# set-up analysis
frameAna = frameAnalyzer()

# - prepare a figure with subplots
fig = plt.figure('PIX data', figsize=(11.5, 8.5))
fig.suptitle("miniPiX EDU Data Acquisition", size="xx-large", color="darkblue")
fig.subplots_adjust(left=0.05, bottom=0.03, right=0.97, top=0.99, wspace=0.0, hspace=0.1)
plt.tight_layout()
gs = fig.add_gridspec(nrows=16, ncols=16)

# - - 2d-display for pixel map
axim = fig.add_subplot(gs[:, :-4])
axim.set_title(title, y=0.97, size="x-large")
axim.set_xlabel("# x        ", loc="right")
axim.set_ylabel("# y        ", loc="top")
vmin = 0.5
vmax = 500
image = np.zeros((npx, npx))
img = axim.imshow(image, cmap='hot', norm=LogNorm(vmin=vmin, vmax=vmax))
img.set_clim(vmin=vmin, vmax=vmax)
cbar = fig.colorbar(img, shrink=0.6, aspect=40, pad=-0.03)
# cbar.set_label("Energy " + unit, loc="top", labelpad=-5 )
axim.arrow(146, -5.0, 110.0, 0, length_includes_head=True, width=1.5, color="darkblue")
axim.arrow(110, -5.0, -110.0, 0, length_includes_head=True, width=1.5, color="darkblue")
axim.text(115.0, -3, "14 mm")
axim.text(.05, -0.055, f"integration time {int(integration_time)}s", transform=axim.transAxes, color = "b")
im_text = axim.text(0.05, -0.09, "#", transform=axim.transAxes, color="darkred", alpha=0.7)
plt.box(False)

# plot analysis results
#  - histogram of pixel energies
axh1 = fig.add_subplot(gs[1:5, -4:])
nbins1 = 100
max1 = 1300
be1 = np.linspace(0, max1, nbins1 + 1, endpoint=True)
bhist1 = bhist(ax=axh1, bins=be1, xlabel="pixel energies" + unit, ylabel="", yscale="log")
# - histogram of cluster energies
axh2 = fig.add_subplot(gs[6:10, -4:])
nbins2 = 100
max2 = 10000
be2 = np.linspace(0, 10000, nbins2 + 1, endpoint=True)
bhist2 = bhist(ax=axh2, bins=be2, xlabel="cluster energies" + unit, ylabel="", yscale="log")
# - scatter plot: cluster size vs. cluster energies
ax3 = fig.add_subplot(gs[11:15, -4:])
ax3.set_xlabel("cluster energies")
ax3.set_ylabel("pixels per cluster")
x3_lin = []
y3_lin = []
(gr_lin,) = ax3.plot(x3_lin, y3_lin, label="linear", marker='.', markersize=1, color="b", ls='', alpha=0.5)
x3_circ = []
y3_circ = []
(gr_circ,) = ax3.plot(x3_circ, y3_circ, label="circular", marker='.', markersize=1, color="g", ls='', alpha=0.5)
x3_unass = []
y3_unass = []
(gr_unass,) = ax3.plot(x3_unass, y3_unass, label="unassigned", marker='.', markersize=1, color="r", ls='', alpha=0.8)
ax3.set_xlim(0, 10000)
ax3.set_ylim(0, 50)
ax3.legend(loc="upper right")

# show plots in interactive mode
plt.ion()
plt.show()

print()
# daq loop
dt_alive = 0.0
dt_active = 0.0
n_frame = 0

# start daq as a Thread
if read_filename is None:
    Thread(target=daq, daemon=True).start()
t_start = time.time()
try:
    while dt_active < run_time:
        if read_filename is None:
            frame2d = np.array(dataQ.get()).reshape((npx, npx))
            dt_alive += acq_count * acq_time
            n_frame += 1
        else:
            n_frame += 1
            if n_frame == n_mx:
                break
            frame2d = data[n_frame - 1]
            ##!time.sleep(1.0)
            time.sleep(0.1)

        # write frame to file ?
        if filename is not None:
            with NpyAppendArray(filename) as npa:
                npa.append(np.array([frame2d]))

        # analyze data
        n_pixels, n_clusters, n_cpixels, circularity, cluster_energies = frameAna(frame2d)
        if n_clusters > 0:
            Energy = cluster_energies.sum()
            Energy_in_clusters = cluster_energies[:n_clusters].sum()
            np_unass = n_cpixels[n_clusters]  # last entry is for unassigned
            E_unass = cluster_energies[n_clusters]
        else:
            Energy = frame2d[frame2d > 0].sum()  # raw pixel energy
            Energy_in_clusters = 0.0
            E_unass = Energy
            np_unass = n_pixels

        # add actual data do cumulated values
        image = image + frame2d
        o_n_clusters += n_clusters
        o_energy += Energy
        o_unassigned += E_unass
        # store in ring-buffers to subtract later
        framebuf[_bidx] = frame2d  
        n_clusters_buf[_bidx] = n_clusters
        energy_buf[_bidx] = Energy
        unassigned_buf[_bidx] = E_unass
        _bidx = _bidx + 1 if _bidx < n_overlay - 1 else 0
        # subtract oldest frame
        image = image - framebuf[_bidx]
        o_n_clusters -= n_clusters_buf[_bidx]
        o_energy -= energy_buf[_bidx]
        o_unassigned -= unassigned_buf[_bidx]
 
        # update histogram 1 with pixel energies
        bhist1.add(frame2d[frame2d > 0])

        # update histogram 2 with cluster energies
        if cluster_energies is not None:
            bhist2.add(cluster_energies[:n_clusters])

        # update scatter plot
        for _i in range(n_clusters):
            if circularity[_i] < circularity_cut:
                x3_lin.append(cluster_energies[_i])
                y3_lin.append(n_cpixels[_i])
            else:
                x3_circ.append(cluster_energies[_i])
                y3_circ.append(n_cpixels[_i])
        gr_lin.set_xdata(x3_lin)
        gr_lin.set_ydata(y3_lin)
        gr_circ.set_xdata(x3_circ)
        gr_circ.set_ydata(y3_circ)
        if np_unass > 0:
            x3_unass.append(E_unass)
            y3_unass.append(np_unass)
            gr_unass.set_xdata(x3_unass)
            gr_unass.set_ydata(y3_unass)

        # update image and status text
        img.set_data(vmin + image)
        dt_active = time.time() - t_start
        dead_time_fraction = 1.0 - dt_alive / dt_active
        status = (
            f"#{n_frame}   active {dt_active:.0f}s   alive {dt_alive:.0f}s "
            + f"  clusters = {o_n_clusters:.0f}  energy: {o_energy:.0f}keV  unassigned: {o_unassigned:.0f}keV"
            + 10 * " "
        )
        im_text.set_text(status)

        # update all subplots in fig
        fig.canvas.start_event_loop(0.001)  # better than plt.pause(), which would steal the focus
        # heart-beat for console
        print(f"    #{n_frame}", end="\r")

except KeyboardInterrupt:
    if read_filename is None:
        cmdQ.put("e")

finally:
    # end daq loop
    _a = input("\n       type <ret> to end --> ")
    pypixet.exit()
