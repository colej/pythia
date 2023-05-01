## Series of functions to generate effective models of eclipsing binary signals
import numpy as np

from pythia.utils.general import sort_on_x
from pythia.utils.resampling import run_binning
from pythia.utils.conversions import time_to_ph
from scipy.interpolate import PchipInterpolator


def get_interp_model(x, y, period, t0,
                     phase_knots=[[-0.5,-0.11], [-0.1,0.1], [0.11,0.5]],
                     nbins=[75,50,75]
                     ):

    ph = time_to_ph(x, period=period, t0=t0)
    xx, yy, _ = sort_on_x(ph, y, np.ones_like(y))

    xx_, yy_ = [], []
    for ii, knots in enumerate(phase_knots):

        xk, yk, _ = run_binning(xx,yy, yerr=None, phStart=knots[0],
                                phStop=knots[1], nbins=nbins[ii])

        xx_.append(xk)
        yy_.append(yk)

    x_ = np.hstack(xx_)
    y_ = np.hstack(yy_)

    ifunc = PchipInterpolator(x_, y_, extrapolate=False)
    model = ifunc(ph)

    # returns phase array (ph), binned phase (x_) and flux (y_) points used
    # to calculate the model, and the interpolation function (ifunc)
    return ph, x_, y_, ifunc
