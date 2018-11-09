import numpy as np
from numpy import cos,sin,pi

from scargle_engine import scar2,scar3

def normalise_distribution(s1):
    s1_var = np.var(s1)
    yield s1/s1_var

def normalise_amplitude(s1,norm_factor):
    yield np.sqrt(s1) * norm_factor

def normalise_power_density(s1,norm_factor,T):
    yield s1 * norm_factor**2 * T



def scargle(times, signal, f0=None, fn=None, df=None, norm='amplitude',
            weights=None):
    """
    Scargle periodogram of Scargle (1982).

    # TODO: Update normalisation options
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

    As of 09/11/2018, this routine has been stripped and updated to work in python 2.7
    and python 3.6. Additionally, the routines are updated to function as generators.

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

    #-- initialize variables for use in Fortran routine
    sigma=0.
    xgem=0.
    xvar=0.
    N=len(times)
    T = times.ptp()
    nf=int((fn-f0)/df+0.001)+1
    f1=np.zeros(nf,'d')
    s1=np.zeros(nf,'d')
    ss=np.zeros(nf,'d')
    sc=np.zeros(nf,'d')
    ss2=np.zeros(nf,'d')
    sc2=np.zeros(nf,'d')

    #-- run the Fortran routine
    if weights is None:
        f1,s1=scar2(signal,times,f0,df,f1,s1,ss,sc,ss2,sc2)
    else:
        w=np.array(weights,'float')
        f1,s1 = scar3(signal,times,f0,df,f1,s1,ss,sc,ss2,sc2,w)

    #-- search for peaks/frequencies/amplitudes
    if not s1[0]: s1[0]=0. # it is possible that the first amplitude is a none-variable
    norm_factor  = np.sqrt(4./N)

    if norm =='distribution': # statistical distribution
        s1_normed = list(normalise_distribution(s1))
    elif norm == "amplitude": # amplitude spectrum
        s1_normed = list(normalise_amplitude(s1,norm_factor))
    elif norm == "density": # power density
        s1_normed = list(normalise_power_density(s1,norm_factor,T))

    yield f1
    yield s1_normed
