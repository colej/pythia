import numpy as np
import matplotlib.pyplot as plt

from pythia.timeseries.iterative_prewhitening import run_ipw
from pythia.timeseries.periodograms_native import LS_periodogram

if __name__=="__main__":

  times,fluxes = np.loadtxt("./test_kepler_pulsator.txt",unpack=True)
  nu, amp = LS_periodogram( times, fluxes-np.mean(fluxes), fn=6.5,
                            normalisation='amplitude')

  fig,ax = plt.subplots(1,1,figsize=(6.6957,6.6957))
  ax.plot(nu,amp,'k-')
  ax.set_xlabel(r'$\nu\,\,{\rm [d^{-1}]}$',fontsize=14)
  ax.set_ylabel(r'${\rm Amplitude\,\,[ppm]}$',fontsize=14)
  ax.set_title(r'${\rm Example}$',fontsize=14)
  plt.show()

  yerr = 0.005* np.ones_like(times)

  residuals, model, offsets, \
  frequencies, amplitudes, \
  phases, stop_criteria, \
  noise_curve = run_ipw( times,fluxes-np.mean(fluxes),
                         yerr, t0=4953.53931246, fn=6.2,
                         maxiter=3)



  np.savetxt('test_kepler_pulsator.out',np.array([offsets,frequencies,amplitudes,phases]).T)
  fig,ax = plt.subplots(1,1,figsize=(6.6957,6.6957))

  print(' C + A*sin( 2*pi*f*(t-t0)+phi )')
  nu_, amp_ = LS_periodogram( times, residuals, fn=6.5,
                            normalisation='amplitude')

  outstr = '{} -- C: {:.6f} -- A: {:.6f} -- f: {:.6f} -- phi: {:.6f} -- SNR: {:.6f}'
  for ii,freq in enumerate(frequencies):
      print(outstr.format(ii, offsets[ii], amplitudes[ii], freq, phases[ii]+0.5*np.pi,stop_criteria[ii]))
      ax.axvline(freq, linestyle='--',color='red',alpha=0.4)
      ax.axhline(amplitudes[ii], linestyle='--',color='red',alpha=0.4)

  ax.plot(nu,amp,'-',color='grey')
  ax.plot(nu_,amp_,'k-')
  ax.set_xlabel(r'$\nu\,\,{\rm [d^{-1}]}$',fontsize=14)
  ax.set_ylabel(r'${\rm Amplitude\,\,[ppm]}$',fontsize=14)
  ax.set_title(r'${\rm Example}$',fontsize=14)
  plt.show()
