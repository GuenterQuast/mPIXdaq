"""Function for import to fit a Landau-pdf to histogram data"""

from PhyPraKit import histstat, hFit
from scipy.stats import landau


# define landau function
def nLandau(x, N=100, loc=20.0, scale=5.0):
    """pdf of landau distribution with normalization"""
    return N * landau.pdf(x, loc, scale)


def fit_Landau(bc, be, mean, sigma, pr=True):
    """fit a Landau distribution to histogram data

    Args:
        bc: bin contents
    """

    rdict = hFit(
        nLandau,
        bc,
        be,  # bin entries and bin edges
        p0=[bc.sum(), mean, sigma],  # initial for parameter values
        #  constraints=['name', val ,err ],   # constraints within errors
        # limits=("N", 0.0, None),  # limits
        use_GaussApprox=False,  # Gaussian approximation
        fit_density=False,  # fit density
        plot=False,  # plot data and model
        plot_band=True,  # plot model confidence-band
        plot_cor=False,  # plot profiles likelihood and contours
        quiet=True,  # suppress informative printout
        axis_labels=["x", "entries / keV"],
        data_legend="electron",
        model_legend="landau",
    )
    pvals, perrs, cor, gof, pnams = rdict.values()
    if pr:
        # Print results to illustrate how to use output
        print("\n*==* Results of Histgoram Fit:")
        print(" goodness-of-fit: {:.3g}".format(gof))
        print(" parameter names:       ", pnams)
        print(" parameter values:      ", pvals)
        print(" neg. parameter errors: ", perrs[:, 0])
        print(" pos. parameter errors: ", perrs[:, 1])
        print(" correlations : \n", cor)
    return (pvals, perrs, gof, pnams)
