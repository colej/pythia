import numpy as np
import array

cimport cython
cimport numpy as np

from cpython cimport array
from libc.stdio cimport printf
from libc.string cimport memset #faster than np.zeros
from libc.math cimport sqrt, fmod, log

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.intp
ctypedef np.intp_t ITYPE_t

# Wrapped helper function for squaring a value
cdef inline square(DTYPE_t x): return x*x

# Declare the structure _HistType
cdef struct _HistType:
  double HIST_SIZE;
  int sizehists;
  unsigned long *histN;
  double *histA
  double *histB
  double *histC
  double *histD
  double *histE;

# Assign _HistType structure to a ctype
ctypedef _HistType _HistType_t


"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""



# python function to call the AoV routine
def aov_periodogram(times, obs, sigma=None, min=None, max=None, frequency=True,
               oversample_factor=1., nbins=8):


  if sigma is None:
    sigma = np.ones(len(obs))*np.std(obs)

  if frequency:
    df = 1./(times.max() - times.min())
    if min is None:
      min = df * 2.

    if max is None:
      max = 0.5/(np.min(np.diff(times)))

    frequencies = np.arange(min, max, df/oversample_factor)

  else:
    dp = 0.001
    if min is None:
      min = dp

    if max is None:
      max = 25.

    frequencies = 1./np.arange(min, max, dp/oversample_factor)

  if times.ndim != 1:
    raise ValueError('times or obs should be 1-D')
  if frequencies.ndim != 1:
    raise ValueError('frequencies array should be 1-D')

  # Prepare for cython interaction
  obs         = np.asarray(obs, dtype=DTYPE, order='C')
  times       = np.asarray(times, dtype=DTYPE, order='C')
  sigmas      = np.asarray(sigma, dtype=DTYPE, order='C')
  frequencies = np.asarray(frequencies, dtype=DTYPE, order='C')


  periodogram = np.zeros(frequencies.shape, dtype=DTYPE, order='C')
  standardised_obs = np.zeros(times.shape, dtype=DTYPE, order='C')
  standardised_sigma = np.zeros(times.shape, dtype=DTYPE, order='C')

  _standardise_fused(obs, sigma, standardised_obs, standardised_sigma)
  _aov_periodogram_cython(times, standardised_obs, frequencies, periodogram, nbins)

  return frequencies.ravel(), periodogram.ravel()

"""--------------------------------------------------------------------------"""
#
# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef _standardise_new(const DTYPE_t[::1] obs, const DTYPE_t[::1] sigma,
#                   DTYPE_t[::1] standardised_obs,
#                   DTYPE_t[::1] standardised_sigma):
#
#   cdef ITYPE_t size = obs.shape[0]
#   cdef DTYPE_t size_d = obs.shape[0]
#   cdef DTYPE_t average = 0.0
#   cdef DTYPE_t stddev = 0.0
#
#   cdef ITYPE_t i = 0
#
#   for i in range(size):
#     average += obs[i]
#     stddev += square(obs[i] - average)
#   average /= size_d
#   stddev = sqrt(stddev / size_d)
#
#   for i in range(size):
#     standardised_obs[i] = (obs[i] - average) / stddev
#     standardised_sigma[i] = sigma[i] / stddev


"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _standardise_fused(const DTYPE_t[::1] obs, const DTYPE_t[::1] sigma,
                        DTYPE_t[::1] standardised_obs,
                        DTYPE_t[::1] standardised_sigma):

    cdef ITYPE_t size = obs.shape[0]
    cdef DTYPE_t size_d = obs.shape[0]
    cdef DTYPE_t average = 0.0
    cdef DTYPE_t stddev = 0.0

    cdef ITYPE_t i = 0

    # Calculate the sum of observations
    for i in range(size):
        average += obs[i]

    # Calculate the average
    average /= size_d

    # Calculate the sum of squares of differences from the average
    for i in range(size):
        stddev += square(obs[i] - average)

    # Calculate the standard deviation
    stddev = sqrt(stddev / size_d)

    # Standardize the observations and sigmas
    cdef DTYPE_t temp
    for i in range(size):
        temp = (obs[i] - average) / stddev
        standardised_obs[i] = temp
        standardised_sigma[i] = sigma[i] / stddev


"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _aov_periodogram_cython(const DTYPE_t[::1] times, const DTYPE_t[::1] obs,
                             const DTYPE_t[::1] frequencies, DTYPE_t[::1] periodogram,
                             const ITYPE_t nbins):

  ''' Compute the AOV Periodogram using J. Devor's implementation '''

  cdef ITYPE_t size = times.shape[0]
  cdef ITYPE_t Nfreqs = frequencies.shape[0]
  cdef ITYPE_t i=0, index=0, k=0, N=0
  cdef DTYPE_t sum=0., X=0., Y=0., s1=0., L2=0., period_=0.

  cdef np.ndarray[ITYPE_t, ndim=1] histN = np.zeros(nbins, dtype=ITYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] histA = np.zeros(nbins, dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] histB = np.zeros(nbins, dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] histC = np.zeros(nbins, dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] histD = np.zeros(nbins, dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=1] histE = np.zeros(nbins, dtype=DTYPE)

  for k in range(Nfreqs):

    sum, X, Y, s1, L2, = 0., 0., 0., 0., 0.
    period_ = 1./frequencies[k]

    memset(&histN[0], 0, sizeof(histN[0])*nbins)
    memset(&histA[0], 0, sizeof(histA[0])*nbins)
    memset(&histB[0], 0, sizeof(histB[0])*nbins)
    memset(&histC[0], 0, sizeof(histC[0])*nbins)
    memset(&histD[0], 0, sizeof(histD[0])*nbins)
    memset(&histE[0], 0, sizeof(histE[0])*nbins)

    for i in range(size):
      X = fmod(times[i],period_) / period_
      index = (<int> (nbins * X))
      Y = obs[i]
      # printf('%li\t%f\t%f\t%f\n', index, X * nbins, X, Y)

      histN[index] += 1
      histA[index] += X
      histB[index] += X*X
      histC[index] += X*Y
      histD[index] += Y
      histE[index] += Y*Y

    for i in range(nbins):
      N = histN[i]
      if N <= 1:
        periodogram[k] = 100000.0

      else:
        Y = histD[i]
        X = (Y*Y)/N

        sum += Y
        s1 += X
        L2 += (square((histC[i] * N) - (Y * histA[i])) / (N * (square(histA[i]) - (histB[i] * N)))) + histE[i] - X


    s1 -= sum * sum / size
    L2 /= size - nbins
    if ( (s1 <= 0.0) | (L2 <= 0.0) ):
      periodogram[k] = 100000.0
    else:
      periodogram[k] = s1/L2
      # periodogram[k] = log(L2/s1)


"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
