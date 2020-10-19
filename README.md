# pythia
# Authors: Cole Johnston
#          Dominic Bowman
# Scarlge, FASPER routines adapted from code originally
# written by L. Cuypers, C. Aerts, P. De Cat, P. Degroote
Modules and routines for analysis of data from the PLATO2.0 mission

timeseries/periodograms -- library containing functions for time-series analysis


Before installing, you must modify the contents of the yml file!
At the bottom, the prefix variable currently says: /YOUR/PATH/TO/miniconda3...
This needs to be changed to reflect the location of miniconda3 (or anaconda3)
in your directory structure.

To install, use miniconda3 or anaconda3, and run:
        conda env create -f pythia.yml

To compile fortran modules, run the following:
        python -m numpy.f2py -c -m engine engine.f95


TO DO:
  Include DFT, FASPER, RL routines in f95/03/08
  Include Cython wrappers of scargle, dft, fasper, NFFT routines
