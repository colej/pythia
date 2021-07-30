import numpy as np
from astropy.timeseries import LombScargle


def normalise_distribution(s1, times):
    s1_var = np.var(s1)
    return s1/s1_var

def normalise_amplitude(s1, times):
    norm_factor = np.sqrt(4.0 / len(times))
    return np.sqrt(s1) * norm_factor

def normalise_power_density(s1, times):
    T = times.ptp()
    norm_factor = np.sqrt(4.0 / len(times))
    return s1 * norm_factor**2 * T

def normalise_power(s1, times):
    return s1


def LS_periodogram(times, signal, f0=None, fn=None,
                 oversample_factor=10.0, normalisation='amplitude'):

    """
    Calculates the amplitude spectrum of a given signal

    Parameters
    ----------
        t : `array`
            Time values
        y : `array`
            Flux or magnitude measurements
            This is not automatically mean-substracted!!!
        f0 : float (default None)
            Minimum frequency to calculate spectrum. Defaults to df
        fn : float
            Maximum frequency to calculate spectrum. Defaults to Nyquist.
        oversample_factor : float
            Amount by which to oversample the spectrum. Defaults to 10.
        normalisation : str (default amplitude)
            Normalise the periodogram. Options are 'distribution',
            'ampliude', 'power_density', 'power'
    """

    df = 1.0 / (times.max() - times.min())

    if f0 is None:
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times))  # *nyq_mult

    freq = np.arange(f0, fn, df / oversample_factor)
    model = LombScargle(times, signal)
    sc = model.power(freq, method="fast", normalization="standard")

    ## Normalised to return desired output units
    noramlise_str = 'normalise_{}'.format(normalisation)
    LS_out = eval(noramlise_str+'(sc,times)')

    return freq, LS_out
