import numpy as np
import array

cimport cython
cimport numpy as np

from cpython cimport array
from libc.stdio cimport printf
from libc.string cimport memset #faster than np.zeros
from libc.math cimport sqrt, fmod, log

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.intp
ctypedef np.intp_t ITYPE_t

cdef index_sum(const DTYPE_t[::1] arr):
    cdef DTYPE_t result = 0.0
    cdef Py_ssize_t i, n = arr.shape[0]
    for i in range(n):
        result += arr[i]
    return result

def ce_periodogram(times, obs, sigma=None, min=None, max=None, frequency=True,
                   oversample_factor=1., phase_bins=10, mag_bins=5):


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
    raise ValueError('times must be 1-D')
  if obs.ndim != 1:
    raise ValueError('obs must be 1-D')
  if frequencies.ndim != 1:
    raise ValueError('frequencies array should be 1-D')
  if obs.shape != times.shape:
    raise Exception("times doesn't have the same shape as obs")


  # Prepare for cython interaction
  obs         = np.asarray(obs, dtype=DTYPE, order='C')
  times       = np.asarray(times, dtype=DTYPE, order='C')
  sigmas      = np.asarray(sigma, dtype=DTYPE, order='C')
  frequencies = np.asarray(frequencies, dtype=DTYPE, order='C')

  # Declare empty periodogram to be filled
  periodogram = np.zeros(frequencies.shape, dtype=DTYPE, order='C')

  ## Call periodogram function
  _conditional_entropy_periodogram_cython(times, obs, frequencies, periodogram,
                                          phase_bins, mag_bins)

  return frequencies.ravel(), periodogram.ravel()


"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _conditional_entropy_periodogram_cython(const DTYPE_t[::1] times,
                                             const DTYPE_t[::1] obs,
                                             const DTYPE_t[::1] frequencies,
                                             DTYPE_t[::1] periodogram,
                                             const ITYPE_t phase_bins,
                                             const ITYPE_t mag_bins):
    cdef int j = 0
    cdef int n_freqs = frequencies.shape[0]

    for j in range(n_freqs):
        periodogram[j] = conditional_entropy_freq(times, obs, frequencies[j],
                                                  phase_bins, mag_bins)



"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef conditional_entropy_freq(const DTYPE_t[::1] x,
                              const DTYPE_t[::1] y,
                              const DTYPE_t freq,
                              const ITYPE_t phase_bins,
                              const ITYPE_t mag_bins):

    cdef ITYPE_t ix=0, iy=0, iz=0, ii=0, ij=0, ik=0
    cdef ITYPE_t npositive_counts=0
    cdef DTYPE_t result = 0., total_counts = 0., countsPhi = 0., pPhi=0.
    cdef DTYPE_t ent = 0.
    cdef np.ndarray[DTYPE_t, ndim=2] h = np.zeros((phase_bins, mag_bins), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] counts
    cdef np.ndarray[DTYPE_t, ndim=1] positive_counts
    cdef np.ndarray[DTYPE_t, ndim=1] normalized_positive_counts

    cdef np.ndarray[DTYPE_t, ndim=1] edges1 = np.empty(1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] edges2 = np.empty(1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] edges  = np.empty((phase_bins+1, mag_bins+1), dtype=DTYPE)

    h, edges1, edges2 = _phase_diagram_histogram(x, y, freq, phase_bins, mag_bins)


    total_counts = np.sum(h)
    # index_sum(h, total_counts)

    for ix in range(phase_bins):
        # countsPhi = np.sum(h[ix,:])
        countsPhi = index_sum(h[ix,:])

        if countsPhi > 0.:
            pPhi = countsPhi/total_counts
            counts = h[ix,:]
            positive_counts = counts[counts > 0]

            if len(positive_counts) > 0:
                normalized_positive_counts = positive_counts/total_counts

                ent = np.sum(normalized_positive_counts * np.log(pPhi/normalized_positive_counts))
                result += ent


    return result

"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""

#
# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef _phase_diagram_normalized(const DTYPE_t[::1] x,
#                                const DTYPE_t[::1] y,
#                                const DTYPE_t period,
#                                DTYPE_t[::1] xout,
#                                DTYPE_t[::1] yout):
#     cdef ITYPE_t i
#     cdef DTYPE_t x_min, y_min, x_max, y_max
#
#     x_min = x[0]%period
#     y_min = y[0]
#
#     # populate xout and yout arrays and
#     # determine min and max values
#     for i in range(x.shape[0]):
#         xout[i] = x[i] % period
#         yout[i] = y[i]
#
#     for i in range(x.shape[0]):
#         if xout[i] < x_min:
#             x_min = xout[i]
#         if yout[i] < y_min:
#             y_min = yout[i]
#
#
#     for i in range(x.shape[0]):
#         xout[i] = xout[i] - x_min
#         yout[i] = yout[i] - y_min
#
#     x_max = xout[0]
#     y_max = yout[0]
#
#     for i in range(x.shape[0]):
#         if xout[i] > x_max:
#             x_max = xout[i]
#         if y[i] > y_max:
#             y_max = yout[i]
#
#     for i in range(x.shape[0]):
#         xout[i] = xout[i] / x_max
#         yout[i] = yout[i] / y_max



"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _phase_diagram_normalized(const DTYPE_t[::1] x,
                               const DTYPE_t[::1] y,
                               const DTYPE_t period,
                               DTYPE_t[::1] xout,
                               DTYPE_t[::1] yout):

    # cdef ITYPE_t i
    cdef Py_ssize_t i = 0 , n = x.shape[0]
    cdef DTYPE_t x_min, y_min, x_max, y_max

    cdef np.ndarray[DTYPE_t, ndim=1] xph = np.zeros(n, dtype=DTYPE)

    # Determine min and max values
    x_min = x[0] % period
    y_min = y[0]
    x_max = x[0] % period
    y_max = y[0]

    for i in range(1,n):
        xph[i] = x[i]%period
        if xph[i] < x_min:
            x_min = xph[i]
        if xph[i] > x_max:
            x_max = xph[i]
        if y[i] < y_min:
            y_min = y[i]
        if y[i] > y_max:
            y_max = y[i]

    # Normalize arrays
    for i in range(n):
        xout[i] = (xph[i] - x_min) / (x_max - x_min)
        yout[i] = (y[i] - y_min) / (y_max - y_min)


"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _phase_diagram_histogram(const DTYPE_t[::1] x,
                             const DTYPE_t[::1] y,
                             const DTYPE_t freq,
                             const ITYPE_t phase_bins,
                             const ITYPE_t mag_bins):

    cdef ITYPE_t ii = 0, ij = 0
    cdef ITYPE_t nobs = x.shape[0]
    cdef np.ndarray[ITYPE_t, ndim=1] h_shape = np.empty(2, dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] h #= np.empty((phase_bins, mag_bins), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] edges1 = np.empty(1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] edges2 = np.empty(1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] edges  = np.empty((phase_bins+1, mag_bins+1), dtype=DTYPE)
    cdef np.ndarray[ITYPE_t, ndim=2] hrange = np.zeros((2,2), dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] phi    = np.zeros(x.shape[0], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] ny     = np.zeros(x.shape[0], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] hist_input = np.empty((nobs,2), dtype=DTYPE)

    # populate phi and ny arrays
    _phase_diagram_normalized(x, y, 1./freq, phi, ny)

    # populate hist_input array with output from the
    # _phase_diagram_normalized routine
    for ii in range(x.shape[0]):
        hist_input[ii,0] = phi[ii]
        hist_input[ii,1] = ny[ii]


    # Establish the normalisation ranges for the histogram routine
    hrange[0,1] = 1
    hrange[1,1] = 1

    # Declare h_shape for bins
    h_shape[0] = phase_bins
    h_shape[1] = mag_bins

    h, (edges1, edges2) = np.histogramdd(hist_input, bins=h_shape, range=hrange, weights=None, density=False)

    return h, edges1, edges2

"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""
"""--------------------------------------------------------------------------"""

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef histdd(np.ndarray[np.double_t, ndim=2] x, np.ndarray[np.intp_t, ndim=1] bins,
#             np.ndarray[np.double_t, ndim=2] range=None, np.ndarray[np.double_t, ndim=2] weights=None):
#
#     cdef np.ndarray[np.double_t, ndim=2] edges = np.zeros((len(bins), 2), dtype=np.double)
#     cdef np.ndarray[np.double_t, ndim=2] counts = np.zeros([b for b in bins], dtype=np.double)
#     cdef np.ndarray[np.intp_t, ndim=1] indices = np.empty(len(bins), dtype=np.intp)
#     cdef np.ndarray[np.intp_t, ndim=1] strides = np.empty(len(bins), dtype=np.intp)
#     cdef int i, j, n, m, k
#
#     # Set default range to the data range
#     if range is None:
#         range = np.zeros((len(bins), 2), dtype=np.double)
#         for i in range(x.shape[1]):
#             range[i, 0] = x[:, i].min()
#             range[i, 1] = x[:, i].max()
#
#     # Create bins and edges for each dimension
#     for i in range(x.shape[1]):
#         if bins[i] < 1:
#             raise ValueError('`bins` must be positive integers')
#         edges[i] = np.linspace(range[i, 0], range[i, 1], bins[i]+1)
#
#     # Compute the bin indices for each data point
#     for n in range(x.shape[0]):
#         for i in range(x.shape[1]):
#             indices[i] = np.searchsorted(edges[i], x[n, i], side='right') - 1
#         if np.all(indices >= 0) and np.all(indices < bins):
#             if weights is None:
#                 counts[tuple(indices)] += 1
#             else:
#                 counts[tuple(indices)] += weights[n]
#
#     return counts, edges

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def histogramdd(double[:, :] data, double[:, :] bins, double[:, :] range):
#     cdef int n_dims = data.shape[1]
#     cdef int n_samples = data.shape[0]
#     cdef int i, j, k
#     cdef int[:, :] indices = <int[:, :]>np.zeros((n_samples, n_dims), dtype=np.intp)
#     cdef double[:, :] histogram = <double[:, :]>np.zeros([int(bins[i, j]) for i in range(n_dims)], dtype=np.float64)
#     cdef double step_size
#
#     for i in range(n_dims):
#         step_size = (range[i, 1] - range[i, 0]) / bins[i, 0]
#         for j in range(n_samples):
#             indices[j, i] = int((data[j, i] - range[i, 0]) // step_size)
#     for j in range(n_samples):
#         histogram[tuple(indices[j, :])] += 1.0
#     return histogram
