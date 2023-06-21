import numpy as np

from astropy.timeseries import LombScargle

def normalise_distribution(s1, times):
    s1_var = np.var(s1)
    return s1/s1_var

def normalise_amplitude(s1, times):
    norm_factor = np.sqrt( 1. / len(times))
    return np.sqrt(s1) * 2. * norm_factor

def normalise_power_density(s1, times):
    T = times.ptp()
    norm_factor = np.sqrt(4.0 / len(times))
    return s1 * norm_factor**2 * T

def normalise_power(s1, times):
    return s1


def LS_periodogram(times, obs, sigma=None, min=None, max=None,
                   oversample_factor=10.0, normalisation='amplitude'):

    """
    Calculates the amplitude spectrum of a given observational signal

    Parameters
    ----------
        t : `array`
            Time values
        obs : `array`
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

    if min is None:
        min = df
    if max is None:
        max = 0.5 / np.median(np.diff(times))  # *nyq_mult

    freq  = np.arange(min, max, df / oversample_factor)
    ls_   = LombScargle(times, obs, dy=sigma)
    sc    = ls_.power(freq, method="fast", normalization="psd")

    if sigma is None:
        sigma = np.array([1.]) #np.ones_like(signal)
    w     = (sigma ** -2.0).sum()
    power = (1. * sc) / w

    ## Normalised to return desired output units
    noramlise_str = 'normalise_{}'.format(normalisation)
    LS_out = eval(noramlise_str+'(power,times)')

    return freq, LS_out
