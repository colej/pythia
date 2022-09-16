import scipy as sp
import numpy as np
import pymc3 as pm
import exoplanet as xo
import pymc3_ext as pmx
import lightkurve as lk
import theano.tensor as tt
import matplotlib.pyplot as plt

from progress.bar import Bar
# from pythia.timeseries.periodograms import scargle
from pythia.timeseries.utils import mean_smooth
from pythia.timeseries.periodograms import LS_periodogram


def get_correlation_factor(residus):
    """
    Calculate the correlation facor rho (Schwarzenberg-Czerny, 2003).

    Under white noise assumption, the residus are expected to change sign every
    2 observations (rho=1). Longer distances, 2*rho, are a sign of correlation.

    The errors are then underestimated by a factor 1/sqrt(rho).

    @param residus: residus after the fit
    @type residus: numpy array
    @param full_output: if True, the groups of data with same sign will be returned
    @type full_output: bool
    @return: rho(,same sign groups)
    @rtype: float(,list)
    """
    same_sign_groups = [1]

    for i in range(1,len(residus)):
        if np.sign(residus[i])==np.sign(residus[i-1]):
            same_sign_groups[-1] += 1
        else:
            same_sign_groups.append(0)

    rho = np.average(same_sign_groups)

    return rho


def map_optimise( x, y, yerr, t0, nu_init, nu_err, amp_init, amp_err,
                  phase_init=None, phase_err=None, fit_offset=False,
                  return_model=False):

    dim = len(nu_init)
    x_ = np.linspace(x.min(),x.max(),1000)
    phi_guess = np.array([-0.5*np.pi for x in nu_init])

    with pm.Model() as model:

        # Gaussian priors based frequency extracted from periodogram
        nu = pm.Bound(pm.Normal, lower=0)(
            "nu",
            mu=np.array(nu_init),
            sd=np.array(nu_err),
            shape=dim,
            testval=np.array(nu_init),
            )

        # Wide log-normal prior for semi-amplitude
        amp = pm.Bound(pm.Normal, lower=0)(
            "amp",
            mu=np.array(amp_init),
            sd=np.array(amp_err),
            shape=dim,
            testval=np.array(amp_init)
            )

        # Phase -- uses periodic boundary sampled in (sin(theta), cos(theta))
        # to avoid discontinuity
        # phase = xo.distributions.Angle("phase", shape=dim,
        #         testval=np.random.uniform(-np.pi, np.pi, dim))
        # if phase_init is None:
        #     phase_init = np.random.uniform(-np.pi, np.pi, dim)
        # phase = pm.Uniform( "phase", lower = -np.pi, upper = np.pi,
        #                     shape=dim, testval=phase_init)
        # phase = xo.distributions.Angle("phase", shape=dim,
        #         testval=np.ones(dim)*np.pi*0.2)
        phase = pm.Uniform('phase', lower=-np.pi, upper=np.pi, shape=dim,
                           testval=np.ones(dim)*np.pi*0.2,
                           transform=pm.transforms.circular)


        if fit_offset:
            offset = pm.Bound(pm.Uniform,lower=-10,upper=10)(
                    "offset",
                    shape=1,
                    testval=np.array([0.])
                    )
        else:
            offset = 0.


        # And a function for computing the full RV model
        def get_sine_model(t, name=""):
            # Sine wave according to parameters

            # sine =  offset + amp*np.sin( 2.*np.pi*nu*(t-t0) + phase )
            t_ = tt.shape_padright(t) - t0

            sine =  tt.squeeze( amp*tt.sin( 2.*np.pi*nu*(t_) + phase ) )
            pm.Deterministic("sine" + name, sine)

            # Sum over planets and add the background to get the full model
            if dim ==  1:
                return pm.Deterministic("sine_model" + name, offset + sine )
            else:
                return pm.Deterministic("sine_model" + name, offset + tt.sum(sine, axis=-1) )

        # Define the model at the observed times
        sine_model = get_sine_model(x)

        # Also define the model on a fine grid as computed above (for plotting)
        sine_model_pred = get_sine_model(x_, name="_pred")

        # Finally add in the observation model. This next line adds a new contribution
        # to the log probability of the PyMC3 model
        if yerr is None:
            pm.Normal("obs", mu=sine_model, observed=y)
        else:
            err = tt.sqrt(yerr ** 2)
            pm.Normal("obs", mu=sine_model, sd=err, observed=y)

    # plt.errorbar(x, y, yerr=yerr, color='black',marker='.',linestyle='')

    with model:

        map_soln = pmx.optimize(start=model.test_point, vars=[phase],
                        tol=1e-10,options={'disp':False})
        map_soln, info = pmx.optimize(start=map_soln,return_info=True,
                        tol=1e-15,options={'disp':False})
        # map_soln, info = xo.optimize(start=model.test_point,return_info=True,
        #                 progress_bar=False,tol=1e-15,options={'disp':False})


    # plt.plot(x_, map_soln["sine_model_pred"],'-', color='darkorange')
    #
    # # plt.legend(fontsize=10)
    # plt.xlim(x.min(), x.max())
    # plt.xlabel("time [days]")
    # plt.ylabel("Magnitude")
    # plt.ylim(plt.ylim()[::-1])
    # plt.show()

    if fit_offset:
        opt_ofst = np.zeros(dim)
        opt_ofst[0] += np.array(map_soln['offset'])
        fit_offset = False
    else:
        opt_ofst = np.array(np.zeros(dim))

    opt_freq = np.array(map_soln['nu'])
    opt_ampl = np.array(map_soln['amp'])
    opt_phase = np.array(map_soln['phase'])

    if return_model:
        model = map_soln['sine_model']
    else:
        model = None
    return np.array(y - map_soln['sine_model']), model, opt_ofst, opt_freq, opt_ampl, opt_phase


def sine_func_fixed_freq(p0, x, y, yerr, freqs, amps, return_sum = True):

    sine_ = np.zeros_like(x)
    for ii,freq in enumerate(freqs):
        pcurr = p0[ii:(ii+1)]
        sine_ += amps[ii] * np.sin( 2.*np.pi*freq*(x) + pcurr[0] )

    if return_sum:
        return np.nansum( ((y-sine_)/yerr)**2 )

    else:
        return sine_


def sine_func_offset(x, offset, freq, ampl, phase):

    sine_ = sine_func(x,freq, ampl, phase)

    return offset + sine_



def sine_func(x, freq, ampl, phase):

    sine_ = ampl * np.sin(2.*np.pi*freq*x + phase)

    return sine_



def get_snr(nu, amp, use_snr_window=True, snr_window=1., snr_range=[23.,24.]):

    if use_snr_window:
        npoints = len( nu[ nu <= nu[0] + snr_window ])
        mean_ = mean_smooth(amp, npoints)
    else:
        idx = np.where( ((nu>=snr_range[0]) & (nu<=snr_range[1])) )
        mean_ = np.ones_like(nu) * np.median(amp[idx])

    return amp / mean_, mean_



def run_ipw(times, signal, yerr, maxiter=100, t0=None,
        f0=None, fn=None, df=None, snr_stop_criteria=4., order_by_snr=False,
        use_snr_window=True, snr_window=1., snr_range=[23.,24.]):

    residuals = signal[:]
    residuals_n_minus_1 = signal[:]
    residuals_n_minus_2 = signal[:]
    offsets, frequencies, amplitudes, phases = [], [], [], []
    stop_criteria = []
    masked = []
    indices = []

    N = len(times)
    T = times.max() - times.min()
    counter = 0
    rayleigh_freq = 1.5/(times[-1]-times[0])
    if t0 is None:
        t0 = 0.5*(times[0]+times[-1])

    stopcrit = False

    nu, amp = LS_periodogram( times, residuals, f0=f0, fn=fn,
                              normalisation='amplitude')

    pbar = Bar('Running...', max=maxiter)
    fit_offset = True
    while maxiter and not stopcrit:

        # Find frequency
        snr_curve, _ = get_snr(nu, amp, use_snr_window=use_snr_window,
                        snr_window=snr_window, snr_range=snr_range)

        # Identify peak either by maximum SNR
        if order_by_snr:
            idx =  np.argmax(snr_curve)
        # or by maximum amplitude
        else:
            idx =  np.argmax(amp)

        indices.append(idx)

        nu_max = nu[idx]
        amp_max = amp[idx]
        mask = np.where( ((nu>=nu_max-0.5*rayleigh_freq) &
                          (nu<=nu_max+0.5*rayleigh_freq)) )[0]

        # Uncorrelated Montgomery 1994 errors
        sigma = np.std(residuals)
        nu_err = np.sqrt(6./N) * sigma / (np.pi * amp_max * T)
        amp_err = np.sqrt(2./N) * sigma

        # Optimize frequencey, amplitude, and phase of peak
        residuals_, model_, c_, nu_, \
        amp_, phase_= map_optimise(times, residuals, yerr, t0,
                        nu_init=[nu_max], nu_err=[nu_err],
                        amp_init=[amp_max], amp_err=[amp_err],
                        fit_offset=fit_offset)

        offsets.append(c_)
        frequencies.append(nu_)
        amplitudes.append(amp_)
        phases.append(phase_)

        maxiter -= 1
        residuals = np.hstack(residuals_)

        nu_res, amp_res = LS_periodogram( times, residuals, f0=f0, fn=fn,
                                          normalisation='amplitude')

        snr_curve_res, noise_ = get_snr( nu_res, amp_res,
                                         use_snr_window=use_snr_window,
                                         snr_window=snr_window,
                                         snr_range=snr_range)

        noise_res = amp_res / snr_curve_res

        snr_at_nu = amp_ / noise_res[idx]

        stop_criteria.append(snr_at_nu)

        #
        # plt.plot(nu, amp, 'k-')
        # plt.plot(nu_res, amp_res, '-', color='grey')
        # plt.axhline(amp_,linestyle='--',color='red')
        # plt.axvline(nu[idx],linestyle='--',color='red')
        # plt.axvline(nu_res[idx],linestyle=':',color='blue')
        # plt.show()


        nu = nu_res
        amp = amp_res
        fit_offset = False

        pbar.next()
        if snr_at_nu < snr_stop_criteria:
            stopcrit = True

    pbar.finish()
    ## Do one final optimisation with all of the frequencies
    sigma = np.std(signal)
    nu_err = [ 2.*np.sqrt(6./N) * sigma / (np.pi * a * T)
               for a in np.hstack(amplitudes) ]
    amp_err = [ 2. * np.sqrt(2./N) * sigma for a in amplitudes ]
    frequencies = np.hstack(frequencies)
    amplitudes = np.hstack(amplitudes)
    phases = np.hstack(phases)

    print('Individually optimised sinusoids: ')
    for ii, freq_ in enumerate(frequencies):
        print('Sinusoid {}: Frequency: {} Amplitude: {} Phase: {}'.format(
               ii, freq_, amplitudes[ii], phases[ii] ))

    residuals_f, model, c_f, nu_f, \
    amp_f, phase_f = map_optimise(times, signal, yerr, t0,
                    nu_init= frequencies, nu_err=nu_err,
                    amp_init=amplitudes, amp_err=amp_err,
                    fit_offset=True, return_model=True)

    nu_final, amp_final = LS_periodogram( times, residuals_f, f0=f0, fn=fn,
                                          normalisation='amplitude')

    snr_curve_final, noise_final = get_snr( nu_final, amp_final,
                                            use_snr_window=use_snr_window,
                                            snr_window=snr_window,
                                            snr_range=[min(amp_f),max(amp_f)])
    noise_f = amp_final / snr_curve_final


    snr_final = []
    for ii,idx in enumerate(indices):
        # snr_final.append( ls_amplitudes[ii] / noise_f[idx])
        snr_final.append( amp_f[ii] / noise_f[idx])


    residuals_f = np.hstack(residuals_f)
    offsets = np.hstack(c_f)
    frequencies = np.hstack(nu_f)
    amplitudes = np.hstack(amp_f)
    phases = np.hstack(phase_f)
    stop_criteria = np.hstack(snr_final)

    return residuals_f, model, offsets, frequencies, amplitudes, phases, \
           stop_criteria, noise_final




def filter_nans(x,y):

    idx = np.isnan(y)

    return x[~idx],y[~idx]
