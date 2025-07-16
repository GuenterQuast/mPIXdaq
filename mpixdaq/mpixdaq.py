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

import matplotlib.pyplot as plt
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
#  functions and classes - - - - -
#
# - handling the miniPIX EDU device


class miniPIXdaq:
    """Initialize and readout miniPIX EDU device

    Args:
      - ac_count: number of frames to overlay
      - ac_time: acquisition time
      - dataQ:  Queue to receive data
      - cmsQ: command Queue
    """

    def __init__(self, ac_count=10, ac_time=0.1, dataQ=None, cmdQ=None):
        """initialize miniPIX device and set up data acquisition"""

        # no device yet
        self.dev = None

        # import python interface to ADVACAM libraries
        try:
            import_pixet()
        except: 
            print("!!! failed to import pypixet library")
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
        pixcfg.setModeAll(self.pixet.PX_TPXMODE_TOT)
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
        while self.cmdQ.empty():
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
        self.clabels = np.array(DBSCAN(eps=1.5, min_samples=2).fit(self.pixel_list).labels_)
        self.n_clusters = len(set(self.clabels)) - (1 if -1 in self.clabels else 0)

        # sum up cluster energies
        self.n_cpixels = np.zeros(self.n_clusters + 1, dtype=np.int32)
        self.cluster_energies = np.zeros(self.n_clusters + 1, dtype=np.float32)
        self.circularity = np.zeros(self.n_clusters + 1, dtype=np.float32)
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
            self.bars = plt.bar(self.bcnt, self.bc, align="center", width=self.w, facecolor="b", edgecolor="grey", alpha=0.5)
            self.ax = plt.gca()
        else:
            self.ax = ax
            self.bars = ax.bar(self.bcnt, self.bc, align="center", width=self.w, facecolor="b", edgecolor="grey", alpha=0.5)

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


class runDAQ:
    """run miniPIX data acquition and analysis"""

    def __init__(self, wd_path):
        """initialize
        - options from command line arguments
        - miniPIX detector or
        - optionally input from file
        - graphics display
        """

        # write to user HOME if no path given
        if wd_path is None:
            wd_path = os.getenv("HOME")
        os.chdir(wd_path)
        self.wd_path = wd_path

        # not running yet
        self.mpl_active = False

        # parse command line arguments
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

        # set options
        self.verbosity = args.verbosity
        self.out_filename = args.file + '_' + timestamp + '.npy' if args.file != '' else None
        self.read_filename = args.readfile if args.readfile != '' else None
        self.acq_time = args.acq_time
        self.acq_count = args.acq_count
        self.n_overlay = args.overlay
        self.circularity_cut = args.circularity_cut
        self.run_time = args.time

        self.integration_time = self.acq_count * self.acq_time * self.n_overlay

        print(f"\n*==* script {sys.argv[0]} executing in working directory {self.wd_path}")

        if self.out_filename is not None:
            # data recording with npy_append_array()
            import_npy_append_array()

        # try to load pypixet library and connect to miniPIX
        if self.read_filename is None:
            maxsize = 16
            self.dataQ = Queue(maxsize)
            self.cmdQ = Queue(1)
            # initialize data acquisition object
            self.daq = miniPIXdaq(self.acq_count, self.acq_time, self.dataQ, self.cmdQ)
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

        # finally, initialize figures
        self.init_figs()

    def on_mpl_close(self, event):
        """call-back for matplotlib for 'close_event'"""
        self.mpl_active = False

    def init_figs(self):
        """initialize figure with pixel image, histograms and scatter plot"""
        # - prepare a figure with subplots
        fig = plt.figure('PIX data', figsize=(11.5, 8.5))
        fig.suptitle("miniPiX EDU Data Acquisition", size="xx-large", color="darkblue")
        fig.canvas.mpl_connect('close_event', self.on_mpl_close)
        self.mpl_active = True
        fig.subplots_adjust(left=0.05, bottom=0.03, right=0.97, top=0.99, wspace=0.0, hspace=0.1)
        plt.tight_layout()
        gs = fig.add_gridspec(nrows=16, ncols=16)
        self.fig = fig

        # - - 2d-display for pixel map
        axim = fig.add_subplot(gs[:, :-4])
        axim.set_title(self.title, y=0.97, size="x-large")
        axim.set_xlabel("# x        ", loc="right")
        axim.set_ylabel("# y        ", loc="top")
        self.vmin = 0.5
        vmax = 500
        self.img = axim.imshow(np.zeros((self.npx, self.npx)), cmap='hot', norm=LogNorm(vmin=self.vmin, vmax=vmax))
        cbar = fig.colorbar(self.img, shrink=0.6, aspect=40, pad=-0.03)
        self.img.set_clim(vmin=self.vmin, vmax=vmax)
        # cbar.set_label("Energy " + unit, loc="top", labelpad=-5 )
        axim.arrow(146, -5.0, 110.0, 0, length_includes_head=True, width=1.5, color="darkblue")
        axim.arrow(110, -5.0, -110.0, 0, length_includes_head=True, width=1.5, color="darkblue")
        axim.text(115.0, -3, "14 mm")
        axim.text(0.05, -0.055, f"integration time {int(self.integration_time)}s", transform=axim.transAxes, color="b")
        self.im_text = axim.text(0.05, -0.09, "#", transform=axim.transAxes, color="darkred", alpha=0.7)
        plt.box(False)

        # plot analysis results
        #  - histogram of pixel energies
        axh1 = fig.add_subplot(gs[1:5, -4:])
        nbins1 = 100
        max1 = 1300
        be1 = np.linspace(0, max1, nbins1 + 1, endpoint=True)
        self.bhist1 = bhist(ax=axh1, bins=be1, xlabel="pixel energies" + self.unit, ylabel="", yscale="log")
        # - histogram of cluster energies
        axh2 = fig.add_subplot(gs[6:10, -4:])
        nbins2 = 100
        max2 = 10000
        be2 = np.linspace(0, 10000, nbins2 + 1, endpoint=True)
        self.bhist2 = bhist(ax=axh2, bins=be2, xlabel="cluster energies" + self.unit, ylabel="", yscale="log")
        # - scatter plot: cluster size vs. cluster energies
        ax3 = fig.add_subplot(gs[11:15, -4:])
        ax3.set_xlabel("cluster energies")
        ax3.set_ylabel("pixels per cluster")
        (self.gr_lin,) = ax3.plot([], [], label="linear", marker='.', markersize=1, color="b", ls='', alpha=0.5)
        (self.gr_circ,) = ax3.plot([], [], label="circular", marker='.', markersize=1, color="g", ls='', alpha=0.5)
        (self.gr_unass,) = ax3.plot([], [], label="unassigned", marker='.', markersize=1, color="r", ls='', alpha=0.8)
        ax3.set_xlim(0, 10000)
        ax3.set_ylim(0, 50)
        ax3.legend(loc="upper right")

        # show plots in interactive mode
        plt.ion()
        plt.show()

    def __call__(self):
        """run daq loop"""
        # - data structure to store miniPIX frames and analysis results per frame
        frame2d = np.zeros((self.npx, self.npx), dtype=np.float32)
        framebuf = np.zeros((self.n_overlay, self.npx, self.npx), dtype=np.float32)
        # accumulative image
        image = np.zeros((self.npx, self.npx), dtype=np.float32)
        # arrays for cluster statistics
        n_clusters_buf = np.zeros(self.n_overlay, dtype=np.float32)
        np_unassigned_buf = np.zeros(self.n_overlay, dtype=np.float32)
        energy_buf = np.zeros(self.n_overlay, dtype=np.float32)
        unassigned_buf = np.zeros(self.n_overlay, dtype=np.float32)
        o_n_clusters = 0
        o_energy = 0.0
        o_np_unassigned = 0
        o_unassigned = 0.0
        # cordinates for linear and circular clusters and unassigned pixels
        x3_lin = []
        y3_lin = []
        x3_circ = []
        y3_circ = []
        x3_unass = []
        y3_unass = []
        i_buf = 0

        # set-up analysis
        frameAna = frameAnalyzer()

        # set up daq
        dt_alive = 0.0
        dt_active = 0.0
        i_frame = 0
        # start daq as a Thread
        if self.read_filename is None:
            Thread(target=self.daq, daemon=True).start()

        # start daq loop
        print("\n" + 15 * ' ' + "\033[37m type <cntrl C> to end" + "\033[31m", end='\r')
        t_start = time.time()
        try:
            while dt_active < self.run_time and self.mpl_active:
                if self.read_filename is None:
                    frame2d[:, :] = np.array(self.dataQ.get()).reshape((self.npx, self.npx))
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

                # add actual data to cumulated values
                image = image + frame2d
                o_n_clusters += n_clusters
                o_energy += Energy
                o_np_unassigned += np_unass
                o_unassigned += E_unass
                # store in ring-buffers to subtract later
                framebuf[i_buf] = frame2d
                n_clusters_buf[i_buf] = n_clusters
                energy_buf[i_buf] = Energy
                np_unassigned_buf[i_buf] = np_unass
                unassigned_buf[i_buf] = E_unass
                i_buf = i_buf + 1 if i_buf < self.n_overlay - 1 else 0
                # subtract oldest frame
                image = image - framebuf[i_buf]
                o_n_clusters -= n_clusters_buf[i_buf]
                o_energy -= energy_buf[i_buf]
                o_np_unassigned -= np_unassigned_buf[i_buf]
                o_unassigned -= unassigned_buf[i_buf]

                # update histogram 1 with pixel energies
                self.bhist1.add(frame2d[frame2d > 0])

                # update histogram 2 with cluster energies
                if cluster_energies is not None:
                    self.bhist2.add(cluster_energies[:n_clusters])

                # update scatter plot
                for _i in range(n_clusters):
                    if circularity[_i] < self.circularity_cut:
                        x3_lin.append(cluster_energies[_i])
                        y3_lin.append(n_cpixels[_i])
                    else:
                        x3_circ.append(cluster_energies[_i])
                        y3_circ.append(n_cpixels[_i])
                self.gr_lin.set_xdata(x3_lin)
                self.gr_lin.set_ydata(y3_lin)
                self.gr_circ.set_xdata(x3_circ)
                self.gr_circ.set_ydata(y3_circ)
                if np_unass > 0:
                    x3_unass.append(E_unass)
                    y3_unass.append(np_unass)
                    self.gr_unass.set_xdata(x3_unass)
                    self.gr_unass.set_ydata(y3_unass)

                # update image and status text
                self.img.set_data(self.vmin + image)
                dt_active = time.time() - t_start
                dead_time_fraction = 1.0 - dt_alive / dt_active
                status = (
                    f"#{i_frame}   active {dt_active:.0f}s   alive {dt_alive:.0f}s "
                    + f"  clusters = {o_n_clusters:.0f} / {o_energy:.0f}keV "
                    + f"  unassigned: {o_np_unassigned:.0f} / {o_unassigned:.0f}keV"
                    + 10 * " "
                )
                self.im_text.set_text(status)

                # redraw and show all subplots in fig
                self.fig.canvas.start_event_loop(0.001)  # better than plt.pause(), which would steal the focus
                # heart-beat for console
                print(f"  #{i_frame}", end="\r")

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print("Excpetion in daq loop: ", str(e))

        finally:
            # end daq loop
            if self.read_filename is None:
                self.cmdQ.put("e")
            if self.mpl_active:
                _a = input("\033[37m\n" + 20 * ' ' + " type <ret> to close window -->\033[0m")
            else:
                print("\33[0m\n" + 20 * ' ' + " Window closed, ending ")
            if self.read_filename is None:
                pypixet.exit()


if __name__ == "__main__":  # -  - - - - - - - - - -
    rD = runDAQ()
    rD.run()
