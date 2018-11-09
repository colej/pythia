import numpy as np
from numpy import cos,sin,pi

from engine import scar2,scar3,fasper
from pythia.timeseries.decorators import defaults_periodogram,getNyquist


def scargle_normalise_distribution(s1):
    s1_var = np.var(s1)
    yield s1/s1_var

def scargle_normalise_amplitude(s1,norm_factor):
    yield np.sqrt(s1) * norm_factor

def scargle_normalise_power_density(s1,norm_factor,T):
    yield s1 * norm_factor**2 * T


@defaults_periodogram
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
        s1_normed = list(scargle_normalise_distribution(s1))[0]
    elif norm == "amplitude": # amplitude spectrum
        s1_normed = list(scargle_normalise_amplitude(s1,norm_factor))[0]
    elif norm == "density": # power density
        s1_normed = list(scargle_normalise_power_density(s1,norm_factor,T))[0]

    return f1, s1_normed

@defaults_periodogram
def fasper(times,signal, f0=None, fn=None, df=None, single=True, norm='amplitude'):
    """
    Fasper periodogram from Numerical Recipes.

    Normalisation here is not correct!!

    @param times: time points
    @type times: numpy array
    @param signal: observations
    @type signal: numpy array
    @param f0: start frequency
    @type f0: float
    @param fn: stop frequency
    @type fn: float
    @param df: step frequency
    @type df: float
    @return: frequencies, amplitude spectrum
    @rtype: array,array
    """
    #-- average nyquist frequency and oversampling rate
    nyq = 1./(2*np.diff(times).mean())
    mynyq = 1./(2*np.diff(times).min())
    T = times.ptp()
    ofac = 1./(df*T)
    hifac = fn/mynyq*mynyq/nyq
    #-- prepare input for fasper
    n = len(times)
    nout = int(4*ofac*hifac*n*4.)
    wk1 = np.zeros(nout)
    wk2 = np.zeros(nout)
    jmax,prob = 0,0.

    wk1,wk2,nwk,nout,jmax,prob = pyfasper.fasper(times,signal,ofac,hifac,wk1,wk2,nout,jmax,prob)
    #wk1,wk2,nout,jmax,prob = fasper_py(times,signal,ofac,hifac)

    wk1,wk2 = wk1[:nout],wk2[:nout]*1.5
    fact  = np.sqrt(4./n)
    if norm =='distribution': # statistical distribution
        wk2 /= np.var(signal)
    elif norm == "amplitude": # amplitude spectrum
        wk2 = fact * np.sqrt(wk2)
    elif norm == "density": # power density
        wk2 = fact**2 * wk2 * T
    if f0 is not None:
        keep = f0<wk1
        wk1,wk2 = wk1[keep],wk2[keep]
    return wk1,wk2


def DFTpower(time, signal, f0=None, fn=None, df=None, full_output=False):

    """
    Computes the modulus square of the fourier transform.

    Unit: square of the unit of signal. Time points need not be equidistant.
    The normalisation is such that a signal A*sin(2*pi*nu_0*t)
    gives power A^2 at nu=nu_0

    @param time: time points [0..Ntime-1]
    @type time: ndarray
    @param signal: signal [0..Ntime-1]
    @type signal: ndarray
    @param f0: the power is computed for the frequencies freq = arange(f0,fn,df)
    @type f0: float
    @param fn: see f0
    @type fn: float
    @param df: see f0
    @type df: float
    @return: power spectrum of the signal
    @rtype: array
    """

    freqs = np.arange(f0,fn,df)
    Ntime = len(time)
    Nfreq = int(np.ceil((fn-f0)/df))

    A = np.exp(1j*2.*pi*f0*time) * signal
    B = np.exp(1j*2.*pi*df*time)
    ft = np.zeros(Nfreq, complex)
    ft[0] = A.sum()
    for k in range(1,Nfreq):
        A *= B
        ft[k] = np.sum(A)

    if full_output:
        return freqs,ft**2*4.0/Ntime**2
    else:
        return freqs,(ft.real**2 + ft.imag**2) * 4.0 / Ntime**2


def DFTpower2(time, signal, freqs):

    """
    Computes the power spectrum of a signal using a discrete Fourier transform.

    The main difference between DFTpower and DFTpower2, is that the latter allows for non-equidistant
    frequencies for which the power spectrum will be computed.

    @param time: time points, not necessarily equidistant
    @type time: ndarray
    @param signal: signal corresponding to the given time points
    @type signal: ndarray
    @param freqs: frequencies for which the power spectrum will be computed. Unit: inverse of 'time'.
    @type freqs: ndarray
    @return: power spectrum. Unit: square of unit of 'signal'
    @rtype: ndarray
    """

    powerSpectrum = np.zeros(len(freqs))

    for i, freq in enumerate(freqs):
        arg = 2.0 * np.pi * freq * time
        powerSpectrum[i] = np.sum(signal * np.cos(arg))**2 + np.sum(signal * np.sin(arg))**2

    powerSpectrum = powerSpectrum * 4.0 / len(time)**2
    return(powerSpectrum)



def DFTscargle(times, signal,f0,fn,df):

    """
    Compute Discrete Fourier Transform for unevenly spaced data ( Scargle, 1989).

    Doesn't work yet!

    It is recommended to start f0 at 0.
    It is recommended to stop fn  at the nyquist frequency

    This makes use of a FORTRAN algorithm written by Scargle (1989).

    @param times: observations times
    @type times: numpy array
    @param signal: observations
    @type signal: numpy array
    @param f0: start frequency
    @type f0: float
    @param fn: end frequency
    @type fn: float
    @param df: frequency step
    @type df: float
    @return: frequencies, dft, Re(dft), Im(dft)
    """
    f0 = 0.
    #fn *= 2*np.pi
    #df *= 2*np.pi
    #-- initialize
    nfreq = int((fn-f0)/(df))
    print "Nfreq=",nfreq
    tzero = times[0]
    si = 1.
    lfreq = 2*nfreq+1
    mm = 2*nfreq
    ftrx = np.zeros(mm)
    ftix = np.zeros(mm)
    om = np.zeros(mm)
    w = np.zeros(mm)
    wz = df #pi/(nn+dt)
    nn = len(times)

    #-- calculate DFT
    ftrx,ftix,om,w = pydft.ft(signal,times,wz,nfreq,si,lfreq,tzero,df,ftrx,ftix,om,w,nn,mm)

    if f0==0:
        ftrx[1:] *= np.sqrt(2)
        ftix[1:] *= np.sqrt(2)
        w[1:] *= np.sqrt(2)

    cut_off = len(w)
    for i in range(0,len(w))[::-1]:
        if w[i] != 0: cut_off = i+1;break

    om = om[:cut_off]
    ftrx = ftrx[:cut_off]
    ftix = ftix[:cut_off]
    w = w[:cut_off]

    # norm amplitudes for easy inversion
    T = times[-1]-times[0]
    N = len(times)

    w *= T/(2.*N)
    ftrx *= T/(2.*N)
    ftix *= T/(2.*N)

    return om*2*np.pi,w,ftrx,ftix
