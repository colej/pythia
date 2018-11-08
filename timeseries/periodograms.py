import numpy as np
from numpy import cos,sin,pi
from scipy.special import jn
from ivs.aux.decorators import make_parallel
from ivs.aux import loggers
from ivs.aux import termtools
from ivs.timeseries.decorators import parallel_pergram,defaults_pergram,getNyquist

@make_parallel
def scargle(times, signal, f0=None, fn=None, df=None, norm='amplitude',
            weights=None, single=False):
    """
    Scargle periodogram of Scargle (1982).

    Several options are available (possibly combined):
        1. weighted Scargle
        2. Amplitude spectrum
        3. Distribution power spectrum
        4. Traditional power spectrum
        5. Power density spectrum (see Kjeldsen, 2005 or Carrier, 2010)

    This definition makes use of a Fortran-routine written by Jan Cuypers, Conny
    Aerts and Peter De Cat. A slightly adapted version is used for the weighted
    version (adapted by Pieter Degroote).

    Through the option "norm", it's possible to norm the periodogram as to get a
    periodogram that has a known statistical distribution. Usually, this norm is
    the variance of the data (NOT of the noise or residuals, see Schwarzenberg-
    Czerny 1998!).

    Also, it is possible to retrieve the power density spectrum in units of
    [ampl**2/frequency]. In this routine, the normalisation constant is taken
    to be the total time span T. Kjeldsen (2005) chooses to multiply the power
    by the 'effective length of the observing run', which is calculated as the
    reciprocal of the area under spectral window (in power, and take 2*Nyquist
    as upper frequency value).

    REMARK: this routine does B{not} automatically remove the average. It is the
    user's responsibility to do this adequately: e.g. subtract a B{weighted}
    average if one computes the weighted periodogram!!

    @param times: time points
    @type times: numpy array
    @param signal: observations
    @type signal: numpy array
    @param weights: weights of the datapoints
    @type weights: numpy array
    @param norm: type of normalisation
    @type norm: str
    @param f0: start frequency
    @type f0: float
    @param fn: stop frequency
    @type fn: float
    @param df: step frequency
    @type df: float
    @return: frequencies, amplitude spectrum
    @rtype: array,array
    """
    if single: pyscargle_ = pyscargle_single
    else:
        pyscargle_ = pyscargle
    #-- initialize variables for use in Fortran routine
    sigma=0.;xgem=0.;xvar=0.;n=len(times)
    T = times.ptp()
    nf=int((fn-f0)/df+0.001)+1
    f1=np.zeros(nf,'d');s1=np.zeros(nf,'d')
    ss=np.zeros(nf,'d');sc=np.zeros(nf,'d');ss2=np.zeros(nf,'d');sc2=np.zeros(nf,'d')

    #-- run the Fortran routine
    if weights is None:
        f1,s1=pyscargle_.scar2(signal,times,f0,df,f1,s1,ss,sc,ss2,sc2)
    else:
        w=np.array(weights,'float')
        logger.debug('Weighed scargle')
        f1,s1=pyscargle_.scar3(signal,times,f0,df,f1,s1,ss,sc,ss2,sc2,w)

    #-- search for peaks/frequencies/amplitudes
    if not s1[0]: s1[0]=0. # it is possible that the first amplitude is a none-variable
    fact  = np.sqrt(4./n)
    if norm =='distribution': # statistical distribution
        s1 /= np.var(signal)
    elif norm == "amplitude": # amplitude spectrum
        s1 = fact * np.sqrt(s1)
    elif norm == "density": # power density
        s1 = fact**2 * s1 * T
    return f1, s1
