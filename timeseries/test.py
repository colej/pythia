import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc

from pythia.timeseries.periodograms import scargle
from pythia.timeseries.iterative_prewhitening import run_ipw, run_ipw_v02


# rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12


if __name__=="__main__":

  # times,fluxes = np.loadtxt("./test.data",unpack=True)
  times,fluxes = np.loadtxt("./test_tess.dat",unpack=True)
  nu_w,amp_w = scargle(times,np.ones_like(times),f0=-5.,fn=5.,norm='amplitude')
  nu,amp = scargle(times,fluxes-np.mean(fluxes),fn=50.,norm='amplitude')


  figw,axw = plt.subplots(1,1,figsize=(6.6957,6.6957),num=2)
  axw.plot(nu_w,amp_w,'k-')
  axw.set_xlabel(r'$\nu\,\,{\rm [d^{-1}]}$',fontsize=14)
  axw.set_ylabel(r'${\rm Amplitude\,\,[ppm]}$',fontsize=14)
  axw.set_title(r'${\rm Window}$',fontsize=14)



  fig,ax = plt.subplots(1,1,figsize=(6.6957,6.6957),num=1)
  ax.plot(nu,amp,'k-')
  ax.set_xlabel(r'$\nu\,\,{\rm [d^{-1}]}$',fontsize=14)
  ax.set_ylabel(r'${\rm Amplitude\,\,[ppm]}$',fontsize=14)
  ax.set_title(r'${\rm Example}$',fontsize=14)
  plt.show()

  yerr = 0.0005* np.ones_like(times)
  residuals, offsets, \
  frequencies, amplitudes, \
  phases, stop_criteria = run_ipw(times,fluxes-np.mean(fluxes), yerr, maxiter=5, fn=30.)

  # yerr = 0.0005* np.ones_like(times)
  # residuals, outpars= run_ipw_v02(times,fluxes-np.mean(fluxes), yerr, maxiter=5, fn=30.)
  #
  # frequencies = outpars['frequency']
  # amplitudes = outpars['amplitude']
  # phases = outpars['phase']
  # offsets = np.zeros_like(phases)
  # offsets[0] += outpars['offset']
  # stop_criteria = outpars['snr']

  np.savetxt('test.out',np.array([offsets,frequencies,amplitudes,phases]).T)
  fig,ax = plt.subplots(1,1,figsize=(6.6957,6.6957))

  print(' C + A*sin( 2*pi*f*(t-t0)+phi )')
  nu_,amp_ = scargle(times, residuals, fn=6.5, norm='amplitude')

  print(offsets)
  print(frequencies)
  print(amplitudes)
  print(phases)
  print(stop_criteria)

  outstr = '{} -- C: {:.6f} -- A: {:.6f} -- f: {:.6f} -- phi: {:.6f} -- SNR: {:.6f}'
  for ii,freq in enumerate(frequencies):
      print(outstr.format(ii, offsets[ii], amplitudes[ii], freq, phases[ii],stop_criteria[ii]))
      ax.axvline(freq, linestyle='--',color='red',alpha=0.4)
      ax.axhline(amplitudes[ii], linestyle='--',color='red',alpha=0.4)

  ax.plot(nu,amp,'-',color='grey')
  ax.plot(nu_,amp_,'k-')
  ax.set_xlabel(r'$\nu\,\,{\rm [d^{-1}]}$',fontsize=14)
  ax.set_ylabel(r'${\rm Amplitude\,\,[ppm]}$',fontsize=14)
  ax.set_title(r'${\rm Example}$',fontsize=14)
  plt.show()
