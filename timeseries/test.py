import numpy as np
import matplotlib.pyplot as plt

from pythia.timeseries.periodograms import scargle


if __name__=="__main__":

  times,fluxes = np.loadtxt("./test.data",unpack=True)
  nu,amp = scargle(times,fluxes-np.mean(fluxes),fn=25.,norm='amplitude')

  fig,ax = plt.subplots(1,1,figsize=(6.6957,6.6957))
  ax.plot(nu,amp,'k-')
  ax.set_xlabel(r'$\nu\,\,{\rm [d^{-1}]}$',fontsize=14)
  ax.set_ylabel(r'${\rm Amplitude\,\,[ppm]}$',fontsize=14)
  ax.set_title(r'${\rm Example}$',fontsize=14)
  plt.show()
