import numpy as np

def run_binning(x,y,yerr=None,phStart=-0.5,phStop=0.5,nbins=100):
	# Binning function -- takes into account aliasing and error
	# propogation on errorbins
	bwidth = (abs(phStart - phStop))/nbins

	print 'Adopted bin-width: ',bwidth
	bins      = np.arange(phStart,phStop+bwidth,bwidth)
	bin_means = ( np.histogram(x,bins,weights=y)[0] / np.histogram(x,bins)[0] )
	if yerr is not None:
		bin_errs = ( np.histogram(x,bins,weights=yerr)[0] / np.histogram(x,bins)[0] )
	else:
		bin_errs = None

    bins = bins[:-1] + 0.5 * bwidth

    return bins,bin_means,bin_errs


def run_mean_smooth(x,window_len):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,num=1000)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    !! Credit to @BRAM BUYSSCHAERT for original code
    """

    if x.ndim != 1:
        raise ValueError("Only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        raise ValueError("Window must be at least 3 points.")

    window_len = int(window_len)

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    w=np.ones(window_len,'d')

    y=np.convolve(w/w.sum(),s,mode='valid')

    if window_len%2 == 0:
      return y[int(window_len/2-1):-int(window_len/2)] #NOTE
    else:
      return y[int(window_len/2):-int(window_len/2)]
