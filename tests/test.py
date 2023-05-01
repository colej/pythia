import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc

from pythia.timeseries.periodograms import LS_periodogram
from pythia.timeseries.iterative_prewhitening import run_ipw


plt.rcParams.update({
    "text.usetex": True,
        "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rcParams.update({
    "pgf.rcfonts": False,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
         r"\usepackage{amsmath}",
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
    ]),
})

plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
if __name__=="__main__":

  # times,fluxes = np.loadtxt("./test.data",unpack=True)
  times,fluxes = np.loadtxt("./test_tess.dat",unpack=True)
  nu_w,amp_w = LS_periodogram(times,np.ones_like(times),f0=0.001,fn=5.)
  nu,amp = LS_periodogram(times,fluxes-np.mean(fluxes),fn=50.)


  figw,axw = plt.subplots(1,1,figsize=(6.6957,6.6957),num=2)
  axw.plot(nu_w,amp_w,'k-')
  axw.plot(-1.*nu_w,amp_w,'k-')
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
  residuals, model, offsets, \
  frequencies, amplitudes, phases,\
  stop_criteria, noise_level = run_ipw(times,fluxes-np.mean(fluxes), yerr, maxiter=5, fn=30.)

  np.savetxt('test.out',np.array([offsets,frequencies,amplitudes,phases]).T)
  fig,ax = plt.subplots(1,1,figsize=(6.6957,6.6957))

  print(' C + A*sin( 2*pi*f*(t-t0)+phi )')
  nu_,amp_ = LS_periodogram(times, residuals, fn=6.5, normalisation='amplitude')

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
