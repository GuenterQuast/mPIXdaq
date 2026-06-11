#!/usr/bin/env python
"""Read cluster data written with mPIXdaq and provide a summary of
data datking and statistics
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import sigmaclip
import pandas as pd
import yaml
import gzip
import sys


class clusterReader:
    """Class implementing data input, classification of clusters and plotting"""

    def __init__(self):
        """read command line arguments"""
        self.parse_args()

    def parse_args(self):
        # - parse command line arguments
        parser = argparse.ArgumentParser(description="read cluster data and show summary")
        parser.add_argument("filename")
        parser.add_argument('-v', '--verbosity', type=int, default=1, help='verbosity level (1)')
        parser.add_argument(
            '--circularity_cut', type=float, default=0.5, help='cut on circularity for alpha detection (0.5)'
        )
        parser.add_argument('--small_cut', type=float, default=4, help='max. number of pixels for small cluster (4)')
        parser.add_argument('--flatness_cut', type=float, default=0.4, help='cut on flatness for alpha detection (0.6)')
        parser.add_argument('--emean_cut', type=float, default=200, help='cut on mean pixel energy (keV)')
        parser.add_argument('--emx_cut', type=float, default=400, help='cut on maximum pixel energy (keV)')

        args = parser.parse_args()
        self.filename = args.filename

        # *==* selection cuts ...
        self.small_cut = args.small_cut  # small clusters
        self.circularity_cut = args.circularity_cut  #  round topology
        self.flatness_cut = args.flatness_cut  # flat energy distribution
        self.emean_cut = args.emean_cut  #  cut on high energy loss per pixel
        self.emx_cut = args.emx_cut  # cut on maximum pixel energy

    def read_data(self, fn):
        """load yaml from file and fill pandas data frame"""

        if '.csv' in fn:  # from csv file
            self.df = pd.read_csv(fn)
            self.meta_data = None
            self.input_dict = None
        elif '.yml' in fn:  # or from file in yaml format
            f = gzip.open(fn, 'rb') if fn.split('.')[-1] == 'gz' else open(fn, 'r')
            self.input_dict = yaml.load(f, Loader=yaml.CLoader)
            self.keys = self.input_dict['keys']
            self.meta_data = self.input_dict["meta_data"]
            #  get cluster data: 1st is list of cluster properties, 2nd is list of [pixel index, energy] pairs
            cluster_data = self.input_dict['cluster_data']
            cluster_properties = [cluster_data[i][0] for i in range(len(cluster_data))]
            pixel_lists = [cluster_data[i][1] for i in range(len(cluster_data))]
            self.df = pd.DataFrame(data=cluster_properties, columns=self.keys)
            self.df['pixels'] = pixel_lists
        else:
            print("!!! unknown file format")
            self.df = None

        # *==* determine number of frames in data set
        self.n_frames = len(set(self.df['time']))  # use time stamps to count number of frames
        if self.meta_data is not None:
            self.acq_time = self.meta_data['acq_time']  # exposure time per frame
            self.T_alive = self.n_frames * self.acq_time
        else:
            self.T_alive = self.df['time'].to_numpy()[-1]

        # *==* remove empty records (i.e. frames without any cluster)
        is_not_nan = self.df['x_mean'].notna()
        n_empty = (~is_not_nan).sum()
        if n_empty > 0:
            print(f"removing empty records: {n_empty} removed")
            self.df = self.df[is_not_nan]

        # *==* determine total number of clusters in data set
        self.n_clusters = len(self.df)
        # *==* add some (useful) derived features to quantify cluster properties
        #  - mean pixel energy
        self.df['Epix_mean'] = self.df['energy'] / self.df['n_pix']
        #  - circularity defined as the ratio of the smaller and the larger eigenvalue
        #    of the covariance matrix of the pixels in a cluster
        self.df['circularity'] = self.df['var_mn'] / np.maximum(self.df['var_mx'].to_numpy(), 0.001)
        #  - flatness (of energy distribution) as the ratio of maximum variances
        #    of pixel and energy distributions in clusters
        self.df['flatness'] = self.df['varE_mx'] / np.maximum(self.df['var_mx'].to_numpy(), 0.001)

        # *==* meta data
        print("\n*==* Contents of file", fn)
        if self.meta_data is not None:
            s_time = f"{self.meta_data['time']}  "
            s_frames = f"{self.input_dict['eor_data']['Nframes']} frames of {self.meta_data['acq_time']}s exposure time"
            s_device = f"{self.input_dict['deviceInfo']['dn']}"
            print("  Data written on " + s_time + "with device  " + s_device)
            print("  " + s_frames)
            print(f"  {self.n_clusters} clusters -> rate = {self.n_clusters / self.T_alive:.3g} Hz")
            print("  cluster features:\n  ", self.keys)
            print()
        else:
            print("   no meta-data available")

    def set_selection_masks(self):
        """classification of cluster types"""

        # *==* set boolean masks for selection
        is_small = self.df['n_pix'] <= self.small_cut  #
        is_circular = self.df['circularity'] >= self.circularity_cut  # circular shape
        is_flat = self.df['flatness'] > self.flatness_cut  # flat energy distribution
        if 'e_mx' in self.df.keys():
            is_high_dEdx = self.df['e_mx'] > self.emx_cut  # high energy loss per pixel
        else:
            is_high_dEdx = self.df['Epix_mean'] > self.emean_cut  # high energy loss per pixel

        # *==* definition of ɑ candidates
        # alphas have
        #   - a round cluster shape with
        #   - a peaking energy distribution and
        #   - a large value of the maximum pixel energy
        shape_is_alpha = is_circular & ~is_flat
        # a loose definition of an ɑ based only on dEdx
        is_cand_alpha = is_high_dEdx
        # a tight definition of an ɑ as the logical 'and' of criterea
        is_alpha = shape_is_alpha & is_high_dEdx
        # avoid non-linearity of response if max. pixel energy is too high
        if 'e_mx' in self.df.keys():
            is_saturating = self.df['e_mx'] > 1200
        else:
            is_saturating = self.df['Epix_mean'] > 210
        is_clean_alpha = is_alpha & ~is_saturating

        # *==* definition of β candidates (long non-alpha tracks)
        shape_is_beta = ~shape_is_alpha & ~is_small
        is_beta = shape_is_beta & ~is_high_dEdx  # and wit low energy deposits

        # *==* define γ candidates (low-multiplicity clusters with small dEdx)
        is_gamma = is_small & ~is_high_dEdx

        # export selection
        # self.sel_alpha = is_cand_alpha
        self.sel_alpha = is_alpha
        # self.sel_alpha = is_clean_alpha  # well-measured alpha
        self.sel_beta = is_beta
        self.sel_gamma = is_gamma

    def __call__(self):
        """read read data, initialize selection cuts"""
        # read data
        self.read_data(self.filename)
        # set selection criterea
        self.set_selection_masks()
        # collect and print
        self.get_statistics()

    def get_statistics(self):
        """Print results"""

        # *==* collect statistics
        _key = 'energy'
        # number of events per class (number of True values in masks)
        N_alpha = self.sel_alpha.sum()
        N_beta = self.sel_beta.sum()
        N_gamma = self.sel_gamma.sum()

        # create 2.5 sigma truncated sample (to avoid outliers)
        c_alpha, _low, _high = sigmaclip(self.df[self.sel_alpha][_key], 2.5, 2.5)
        c_beta, _low, _high = sigmaclip(self.df[self.sel_beta][_key], 2.5, 2.5)
        c_gamma, _low, _high = sigmaclip(self.df[self.sel_gamma][_key], 2.5, 2.5)

        # print in tabular form
        print("\n*==* ɑ, β, γ Statistics:")
        print("                    " + f"\t {'ɑ ':>10s} \t {'β ':>10s} \t {'γ ':>10s}")
        print("  events            " + f"\t {int(N_alpha):10d} \t {int(N_beta):10d} \t {int(N_gamma):10d}")
        _tl = self.T_alive
        print("  rate (Hz)         " + f"\t {N_alpha / _tl:10.3g} \t {N_beta / _tl:10.3g} \t {N_gamma / _tl:10.3g}")
        print("  mean energy (keV) " + f"\t {c_alpha.mean():10.3g} \t {c_beta.mean():10.3g} \t {c_gamma.mean():10.3g}")
        print("  sigma energy (keV) " + f"\t {c_alpha.std():10.3g} \t {c_beta.std():10.3g} \t {c_gamma.std():10.3g}")
        print()

        self.c_alpha = c_alpha
        self.c_beta = c_beta
        self.c_gamma = c_gamma
        self.N_alpha = N_alpha
        self.N_beta = N_beta
        self.N_gamma = N_gamma

    def plot(self):
        """Graphical output"""
        _key = 'energy'

        def stacked_hists(_key, _bins):
            """helper to produce stacked histograms for alpha, beta and gamma candidates"""
            _vals = (self.df[self.sel_alpha][_key], self.df[self.sel_beta][_key], self.df[self.sel_gamma][_key])
            _labels = ('ɑ', 'β', 'γ')
            _colors = ('r', 'b', 'y')
            return plt.hist(_vals, label=_labels, bins=_bins, color=_colors, alpha=0.75, rwidth=0.75, stacked=True)

        # *==* histogram energy distributions
        # mx = max(df[is_clean_alpha][_key])
        mx = 4500
        f = 10 ** int(np.log10(mx / 5))
        _bins = np.linspace(0, mx, int(mx / f) + 1)
        _rc_hist = stacked_hists(_key, _bins)
        plt.xlabel("Cluster energy (keV)")
        plt.ylabel(f"Counts / {f} keV")
        _tl = self.T_alive

        rate_txt = f"ɑ: {self.N_alpha / _tl:.2g} Hz, β: {self.N_beta / _tl:.2g} Hz, γ: {self.N_gamma / _tl:.2g} Hz"
        plt.text(0.25, 0.75, "rates " + rate_txt, transform=plt.gca().transAxes)
        energy_txt = (
            f"ɑ: {self.c_alpha.mean():.3g} keV, β: {self.c_beta.mean():.3g} keV, γ: {self.c_gamma.mean():.3g} keV"
        )
        plt.text(0.25, 0.70, "energies " + energy_txt, transform=plt.gca().transAxes)
        plt.title("Energy")
        plt.yscale('log')
        plt.legend()
        return plt.gcf()


if __name__ == "__main__":  # ------------------------------------------------------------------------
    # ->>  set input file
    if len(sys.argv) <= 1:
        print("!!! no input file name given !!!")
        sys.exit(1)

    print(" .... importing yaml, be patient ...")
    reader = clusterReader()
    reader()
    fig = reader.plot()

    plt.show()
