# pythia
# Authors: Cole Johnston
#          Dominic Bowman
# Scarlge, FASPER routines adapted from code originally
# written by L. Cuypers, C. Aerts, P. De Cat, P. Degroote
Modules and routines for analysis of data from the PLATO2.0 mission

timeseries/periodograms -- library containing functions for time-series analysis


To compile fortran modules, run the following: 
        python -m numpy.f2py -c -m engine engine.f95


TO DO:
  Update to python3+ support
  Include DFT, FASPER routines in f95/03/08
  Include Cython wrappers of scargle, dft, fasper, NFFT routines

LONG TERM:
  Include support for pre-whitening analysis
