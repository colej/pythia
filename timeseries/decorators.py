"""
Various decorator functions for time series analysis
    - Parallel periodogram
    - Autocompletion of default arguments
"""
import functools
import numpy as np

# TODO: Parallelize

def defaults_periodogram(fctn):
    """
    Set default parameters common to all periodograms.
    """
    @functools.wraps(fctn)
    def globpar(*args,**kwargs):
        #-- this is the information we need the compute everything
        times = args[0]
        signal = args[1]
        T = times.ptp()

        #-- get information on frequency range. If it is not given, compute the
        #   start (0.1/T) and stop (Nyquist) frequency.
        #   Also compute the frequency step as 0.1/T
        nyq_stat = kwargs.pop('nyq_stat',np.min)
        nyquist = getNyquist(times,nyq_stat=nyq_stat)
        f0 = kwargs.get('f0',0.01/T)
        fn = kwargs.get('fn',nyquist)
        df = kwargs.get('df',0.1/T)
        if df==0: df = 0.1/T
        if f0==0: f0 = 0.01/T
        if fn>nyquist:
            fn = nyquist
        kwargs['f0'] = f0
        kwargs['df'] = df
        kwargs['fn'] = fn

        return fctn(times,signal,*args[2:],**kwargs)

    return globpar




def getNyquist(times,nyq_stat=np.inf):
    """
    Calculate Nyquist frequency.

    Typical use is minimum or median of time points differences.

    If C{nyq_stat} is not callable, it is assumed to be a number and that number
    will just be returned: this you can do to search for frequencies above the
    nyquist frequency

    @param times: sorted array containing time points
    @type times: numpy array
    @param nyq_stat: statistic to use or absolute value of the Nyquist frequency
    @type nyq_stat: callable or float
    @return: Nyquist frequency
    @rtype: float
    """
    if not hasattr(nyq_stat,'__call__'):
        nyquist = nyq_stat
    else:
        nyquist = 1/(2.*nyq_stat(np.diff(times)))
    return nyquist
