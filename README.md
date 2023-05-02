# pythia
# Author: Cole Johnston
# co-authors: Nora Eisner

## Contributors: Dominic Bowman


Modules and routines for time-series analysis of astronomical data sets.


timeseries/periodograms -- library containing functions for time-series analysis


First, clone pythia into a location where your PYTHONPATH points.

Before installing, you must modify the contents of the yml file!
At the bottom, the prefix variable currently says: /YOUR/PATH/TO/miniconda3...
This needs to be changed to reflect the location of miniconda3 (or anaconda3)
in your directory structure.

To install, use miniconda3 or anaconda3, and run:
        conda env create -f pythia.yml

        python setup.py build_ext --inplace
        python setup.py install

### TO DO:
  - Generalize handling of priors
  - Include GUI support
  - Include Gaussian Processes
  - Expand to multiple filters (long term)
