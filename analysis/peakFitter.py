"""peakFitter: find and fit peaks in a spectrum"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#
from PhyPraKit import meanFilter, hFit


class peakFitter:
    """peakFitter: find and fit peaks in a spectrum

    algorithm: perform a tow-stage determination of peak positions:

    1. scipi.signal.find_peaks() is run on a smoothened histogram of channel counts.

    2. the characteristics of identified peaks are used as starting points for histogram
       fits with PhyPraKit.phyFit() based on the binned negative log-likelihood method.

    Parameter constraints are adjusted for a Gaussian peak on top of a small, flat background.
    """

    # factor to convert Gaussian Sigma to full-width-half-maximum
    sig2fwhm = 2.3548

    # define fit function
    @staticmethod
    def gauss_plus_bkg(x, Ns=1000, mu=1000, sig=50.0, Nb=100.0, s=0.0, mn=0, mx=1.0):
        """Gaussian shape on linear background"""
        # Ns: number of signal events
        # mu: peak position
        # sig: peak width in sigma
        # Nb: number of background events in interval [mn, mx]
        # s: slope of base line
        # mn: lower bound of fit interval (as fixed parameter)
        # mx: upper bound of fit interval (as fixed parameter)

        # calculate integral of Gauss (not needed if Ns is total signal from -\inf to \inf)
        # I = norm.cdf(mx, mu, sig) - norm.cdf(mn, mu, sig)

        # Gaussian signal
        S = np.exp(-0.5 * ((x - mu) / sig) ** 2) / sig / np.sqrt(2 * np.pi)
        # linear background model
        B = (1 + (x - mu) * s) / (s / 2 * (mx**2 - mn**2) + (1 - s * mu) * (mx - mn))
        return Ns * S + Nb * B

    def __init__(
        self,
        min_prominence=15,
        min_width=10,
        rel_height=0.5,
        fit_range_factor=0.9,
        min_channel=10,
        smoothing_window=5,
    ):
        self.min_prominence = min_prominence  #  minimum peak hight over baseline
        self.min_width = min_width  #  minimum width
        self.rel_height = rel_height  #  width at half peak height (i.e. FWHM)
        # constants for fit range
        self.fit_range_factor = fit_range_factor  # fit range = fit_range_factor * fwhm
        self.min_channel = min_channel  # threshold for min. valid channel number
        self.smoothing_window = smoothing_window  # window for smoothing with sliding average

        # some options:
        self.plot = True
        self.verbose = True

    def set_options(self, verbose=True, plot=True):
        """set options"""
        self.verbose = verbose
        self.plot = plot

    def __call__(self, hst):
        # find maxima with scipy.signal.find_peaks()
        #  first, smoothen data to reduce statistical noise

        hlen = len(hst)
        bin_edges = np.linspace(0, hlen, hlen + 1, endpoint=True)

        if self.smoothing_window > 1:
            hst_s = meanFilter(hst, self.smoothing_window)
        else:
            hst_s = hst
        # search for peaks
        # peaks, peak_props = find_peaks(hst_s, prominence=min_prominence, width=min_width, rel_height=rel_height, wlen=350)
        peaks, peak_props = find_peaks(
            hst_s,
            prominence=self.min_prominence,
            width=self.min_width,
            rel_height=self.rel_height,
            wlen=3.0 * self.min_width,
        )
        if True:
            print(len(peaks), "peaks found: ", peaks)
            print("prominences: ", peak_props["prominences"])
            print("left_bases: ", peak_props["left_bases"])
            print("right_bases: ", peak_props["right_bases"])
            print("fwhms: ", peak_props["widths"])

        prominences = peak_props["prominences"]
        left_bases = peak_props["left_bases"]
        right_bases = peak_props["right_bases"]
        fwhms = peak_props["widths"]

        # fit for precise determination of peak properties and uncertainties

        # - initialize arrays for output
        fit_list = []
        plot_ranges = np.zeros([len(peaks) + 1, 2])
        plot_ranges2 = np.zeros([len(peaks) + 1, 2])
        print("\n" + "*==* fit results:")
        print("              mu   ±  d_mu  ( sig  sig/mu  FWHM/mu )")
        # run fits for all identified peaks
        for i, p in enumerate(peaks):
            wid = int(self.fit_range_factor * fwhms[i])
            base = (hst[left_bases[i]] + hst[right_bases[i]]) / 2.0
            mn = int(max(self.min_channel, p - wid))
            prom = prominences[i]
            mx = int(min(hlen, p + wid))
            # skip peaks below threshold
            if mn >= mx:
                fit_list = np.append(fit_list, None)
                continue
            _be = np.linspace(mn, mx + 1, mx - mn + 1)
            _bc = hst[mn:mx]

            rdict = hFit(
                self.gauss_plus_bkg,
                _bc,  # bin entries
                _be,  # bin edges
                p0=[
                    prom,
                    p,
                    wid,
                    base * (mx - mn),
                    -0.01,
                    mn,
                    mx,
                ],  #  initial parameter values mu, sig, Nb, s, mn, mx
                # constraints=[[]'name', val ,err],[]]  # constraints within errors
                limits=(
                    ["sig", wid / 2.0, 2.0 * wid],
                    ["Ns", prom / 2.0, None],
                    ["Nb", 0.0, None],
                    ["s", -0.01, 0.01],
                ),  # limits
                fixPars=["mn", "mx"],
                use_GaussApprox=False,  # Gaussian approximation
                fit_density=False,  # fit density
                plot=False,  # plot data and model
            )
            if self.verbose:
                print(rdict)  # optionally, report fit results
            # save result
            fit_list = np.append(fit_list, rdict)

            # show results
            pvals, perrs, cor, gof, pnams = rdict.values()
            perrs = (perrs[:, 1] - perrs[:, 0]) / 2.0  # symmetric errors
            print(
                f"{i + 1} peak@{p}: {pvals[1]:.2f} ± {perrs[1]:.2g}"
                + f"( {pvals[2]:.2f} {100 * pvals[2] / pvals[1]:.1f}%"
                + f"{100 * peakFitter.sig2fwhm * pvals[2] / pvals[1]:.1f}% )"
            )

            plot_ranges[i][0] = max(0, p - wid)
            plot_ranges[i][1] = min(hlen, p + wid + 1)
            plot_ranges2[i][0] = max(0, p - 2 * wid)
            plot_ranges2[i][1] = min(hlen, p + 2 * wid + 1)

        # plot result
        if self.plot:
            fig = plt.figure("Spectrum", figsize=(12, 10))
            fig.suptitle("Spectrum ")
            fig.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.95, wspace=None, hspace=0.1)
            ax0 = fig.add_subplot(211)
            ax1 = fig.add_subplot(212)
            ax0.set_ylabel("Entries per Channel")
            ax0.grid(linestyle="dotted", which="both")
            # ax0.set_xticklabels([])
            ax1.set_ylabel("Entries per Channel")
            ax1.set_xlabel("Channel #")
            ax1.grid(linestyle="dotted", which="both")

            # show spectrum and result of find_peaks
            xhst = np.linspace(0, hlen, hlen, endpoint=False) + 0.5
            ax0.plot(
                xhst,
                hst_s,
                "b-",
                linewidth=1,
                zorder=2,
                label="smoothed channel counts",
            )
            ax0.errorbar(
                xhst,
                hst,
                yerr=np.sqrt(hst),
                zorder=1,
                label="channel counts",
                fmt=".",
                color="grey",
                markersize=2,
                linewidth=2,
                alpha=0.5,
            )
            for _i in range(len(peaks)):
                ax0.text(
                    peaks[_i],
                    hst_s[peaks[_i]],
                    str(_i + 1),
                    color="red",
                    horizontalalignment="center",
                )
            #   ax0.plot(peaks, hst_s[peaks], 'x', color='red', markersize=10, zorder=3,
            #             label='result of find_peaks()')
            ax0.legend(loc="best")

            # show fitted peaks
            ax1.errorbar(
                xhst,
                hst,
                yerr=np.sqrt(hst),
                fmt=".",
                color="grey",
                alpha=0.25,
                zorder=1,
                label="channel counts",
            )
            # select colors for peaks
            pcolors = (
                "steelblue",
                "darkorange",
                "green",
                "orchid",
                "turquoise",
                "tomato",
                "green",
                "pink",
                "salmon",
                "yellowgreen",
            )
            for i, fit in enumerate(fit_list):
                if fit is None:
                    continue

                _pvals, _perrs, _cor, _gof, _pnams = fit.values()
                _perrs = (_perrs[:, 1] - _perrs[:, 0]) / 2.0  # symmetric errors
                # plot fitted peak in fit range
                colr = pcolors[i % 10]
                xplt = np.linspace(
                    plot_ranges[i][0],
                    plot_ranges[i][1],
                    10 * int((plot_ranges[i][1] - plot_ranges[i][0])),
                )
                ax1.plot(
                    xplt,
                    self.gauss_plus_bkg(xplt, *_pvals),
                    linestyle="solid",
                    linewidth=3,
                    color=colr,
                    zorder=2,
                    label="peak " + str(i + 1) + "@" + str(int(10 * _pvals[1]) / 10.0),
                )
                # plot fitted peak near fit region
                xplt2 = np.linspace(
                    plot_ranges2[i][0],
                    plot_ranges2[i][1],
                    10 * int((plot_ranges[i][1] - plot_ranges[i][0])),
                )
                ax1.plot(
                    xplt2,
                    self.gauss_plus_bkg(xplt2, *_pvals),
                    zorder=2,
                    linestyle="dotted",
                    linewidth=2,
                    color=colr,
                )
                # show fitted peak properties
                mu = _pvals[1]
                sig = _pvals[2]
                fwhm = peakFitter.sig2fwhm * sig
                mx = self.gauss_plus_bkg(mu, *_pvals)
                h = _pvals[0] / np.sqrt(2 * np.pi) / sig
                ax1.vlines(mu, mx - h, mx, linewidth=3, color="goldenrod")
                ax1.vlines(mu, 0, mx - h, linewidth=1, linestyle="dashed", color="goldenrod")
                ax1.hlines(
                    mx - h / 2,
                    mu - fwhm / 2,
                    mu + fwhm / 2,
                    linewidth=2,
                    color="goldenrod",
                )
                ax1.hlines(mx - h, mu - fwhm / 4, mu + fwhm / 4, linewidth=3, color="goldenrod")
                ax1.legend(loc="best")
            plt.show()

        return fit_list
