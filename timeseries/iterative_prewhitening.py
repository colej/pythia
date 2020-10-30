import scipy as sp
import numpy as np
import pymc3 as pm
import exoplanet as xo
import theano.tensor as tt
import matplotlib.pyplot as plt

from progress.bar import Bar
from pythia.timeseries.periodograms import scargle
from pythia.timeseries.smoothing import mean_smooth


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


def map_optimise(x, y, yerr, t0, nu_init, nu_err, amp_init, amp_err,
                    fit_offset=False, return_model=False):

    dim = len(nu_init)
    x_ = np.linspace(x[0],x[-1],1000)
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
        phase = xo.distributions.Angle("phase", shape=dim,
                testval=np.ones(dim)*np.pi*0.5)

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
        err = tt.sqrt(yerr ** 2)
        pm.Normal("obs", mu=sine_model, sd=err, observed=y)


    plt.errorbar(x, y, yerr=yerr, color='black',marker='.',linestyle='')

    with model:

        map_soln = xo.optimize(start=model.test_point, vars=[phase],
                        progress_bar=False,tol=1e-10,options={'disp':False})
        map_soln, info = xo.optimize(start=map_soln,return_info=True,
                        progress_bar=False,tol=1e-15,options={'disp':False})
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
    return np.array([y - map_soln['sine_model']]), model, opt_ofst, opt_freq, opt_ampl, opt_phase

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


def run_curve_fit(func, x, y, p0, yerr, method='trf', ftol=1e-14,
                    lower=-np.inf , upper=np.inf, max_nfev=5000):
    opt,cov = sp.optimize.curve_fit(func, xdata=x, ydata=y, p0=p0, sigma=yerr,
                                    ftol=ftol, method=method,
                                    bounds=[0,upper], max_nfev=max_nfev)
    # opt,cov = sp.optimize.curve_fit(func, xdata=x, ydata=y, p0=p0, sigma=yerr,
    #                                 ftol=ftol, method=method,
    #                                 maxfev=max_nfev)
    std = np.sqrt(np.diag(cov))
    return opt, std


def optimise_(x, y, yerr, indict, fit_freq=False, fit_offset=False):

    freqs = indict['frequency']
    efreqs = indict['e_frequency']
    ampls = indict['amplitude']
    eampls = indict['e_amplitude']
    phases = indict['phase']
    offset = indict['offset']
    nfreqs = len(freqs)

    outdict = {**indict}

    y_ = y[:]

    if fit_freq:
        ## Define bounds
        if fit_offset:

            for jj, freq in enumerate(freqs):
                x0 = np.hstack([offset, freq, ampls[jj], phases[jj]])
                lower = [-1., max(0.,freq-3.*efreqs[jj]),
                         max(0.,ampls[jj]-3.*eampls[jj]), 0.]
                upper = [1., freq+5.*efreqs[jj], 1., 2.*np.pi]
                opt, std = run_curve_fit(sine_func_offset, x, y_, x0, yerr,
                                         method='trf', ftol=1e-14, lower=lower,
                                         upper=upper )

                outdict['offset']        = opt[0]
                outdict['frequency'][jj] = opt[1]
                outdict['amplitude'][jj] = opt[2]
                outdict['phase'][jj]     = opt[3]

                outdict['e_offset']        = std[0]
                outdict['e_frequency'][jj] = std[1]
                outdict['e_amplitude'][jj] = std[2]
                outdict['e_phase'][jj]     = std[3]

                y_ -= sine_func_offset(x, *opt)


        else:
            for jj, freq in enumerate(freqs):
                x0 = np.hstack([freq, ampls[jj], phases[jj]])
                lower = [max(0.,freq-5.*efreqs[jj]),
                         max(0.,ampls[jj]-5.*eampls[jj]), 0. ]
                upper = [freq+5.*efreqs[jj], ampls[jj]+5.*eampls[jj], 2.*np.pi]
                opt, std = run_curve_fit(sine_func, x, y_, x0, yerr,
                                         method='trf', ftol=1e-14, lower=lower,
                                         upper=upper )

                outdict['frequency'][jj] = opt[0]
                outdict['amplitude'][jj] = opt[1]
                outdict['phase'][jj]     = opt[2]

                outdict['e_frequency'][jj] = std[0]
                outdict['e_amplitude'][jj] = std[1]
                outdict['e_phase'][jj]     = std[2]

                y_ -= sine_func(x, *opt)


    else:
        ## Here use powell
        x0 = np.hstack(phases)
        bounds = []
        for f in freqs:
            # bounds.append( tuple([-2.*np.pi, 2.*np.pi]) )
            bounds.append( tuple([0., 2.*np.pi]) )

        opt = sp.optimize.minimize(sine_func_fixed_freq, x0,
                    args=(x, y, yerr, freqs, ampls), method='SLSQP', tol=1e-14,
                    bounds=tuple(bounds))

        outdict['phase'] = np.hstack(opt.x)

        y_ -= sine_func_fixed_freq(opt.x, x, y_, yerr, freqs, ampls,
                                    return_sum=False)

    return y_, outdict




def get_snr(nu, amp, use_snr_window=True, snr_window=1., snr_range=[23.,24.]):

    if use_snr_window:
        npoints = len( nu[nu<=snr_window])
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
    offsets,frequencies, amplitudes, phases = [], [], [], []
    stop_criteria = []
    masked = []

    N = len(times)
    T = times[-1]-times[0]
    counter = 0
    rayleigh_freq = 1.5/(times[-1]-times[0])
    if t0 is None:
        t0 = 0.5*(times[0]+times[-1])

    stopcrit = False

    nu,amp = scargle(times, residuals, f0=f0, fn=fn, df=df, norm='amplitude')

    # pbar = Bar('Running...', max=maxiter)
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


        nu_max = nu[idx]
        amp_max = amp[idx]
        mask = np.where( ((nu>=nu_max-0.5*rayleigh_freq) &
                          (nu<=nu_max+0.5*rayleigh_freq)) )[0]

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

        nu_res, amp_res = scargle(times, residuals, f0=f0, fn=fn,df=df, norm='amplitude')
        snr_curve_res, noise_ = get_snr(nu_res, amp_res, use_snr_window=use_snr_window,
                        snr_window=snr_window, snr_range=snr_range)
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

        if snr_at_nu < snr_stop_criteria:
            stopcrit = True


    ## Do one final optimisation with all of the frequencies
    sigma = np.std(yerr)
    nu_err = np.array( [ np.sqrt(6./N) * sigma / (np.pi * a * T) for a in np.hstack(amplitudes) ] )
    amp_err = np.ones_like(amplitudes)*np.sqrt(2./N) * sigma

    residuals_f, model, c_f, nu_f, \
    amp_f, phase_f= map_optimise(times, signal, yerr, t0,
                    nu_init=np.hstack(frequencies), nu_err=nu_err,
                    amp_init=np.hstack(amplitudes), amp_err=amp_err,
                    fit_offset=True,return_model=True)

    offsets = np.hstack(c_f)
    frequencies = np.hstack(nu_f)
    amplitudes = np.hstack(amp_f)
    phases = np.hstack(phase_f)
    stop_criteria = np.hstack(stop_criteria)

    return residuals, model, offsets, frequencies, amplitudes, phases, stop_criteria


def updated_extracted(extracted, current):

    for key in current:
        if 'offset' in key:
            extracted[key] = current[key]
        else:
            extracted[key] = np.hstack( [extracted[key],current[key]] )

    return extracted


def run_ipw_v02(times, signal, yerr, maxiter=100, t0=None,
        f0=None, fn=None, df=None, snr_stop_criteria=4., order_by_snr=False,
        use_snr_window=True, snr_window=1., snr_range=[23.,24.]):

    residuals = np.copy(signal)
    residuals_n_minus_1 = np.copy(signal)
    residuals_n_minus_2 = np.copy(signal)
    offsets,frequencies, amplitudes, phases = [], [], [], []
    stop_criteria = []
    masked = []

    extracted = {'frequency':[], 'amplitude':[], 'phase':[], 'offset': 0.,
                 'e_frequency':[], 'e_amplitude':[], 'e_phase':[], 'e_offset': 0.,
                 'snr':[]}
    current = {**extracted}

    N = len(times)
    T = times[-1]-times[0]
    counter = 0
    rayleigh_freq = 1.5/(times[-1]-times[0])
    if t0 is None:
        t0 = 0.5*(times[0]+times[-1])

    stopcrit = False

    nu,amp = scargle(times, residuals, f0=f0, fn=fn, df=df, norm='amplitude')

    # pbar = Bar('Running...', max=maxiter)
    fit_offset = True
    while maxiter and not stopcrit:


        # Find frequency
        snr_curve = get_snr(nu, amp, use_snr_window=use_snr_window,
                        snr_window=snr_window, snr_range=snr_range)

        # Identify peak either by maximum SNR
        if order_by_snr:
            idx =  np.argmax(snr_curve)
        # or by maximum amplitude
        else:
            idx =  np.argmax(amp)


        nu_max = nu[idx]
        amp_max = amp[idx]
        mask = np.where( ((nu>=nu_max-0.5*rayleigh_freq) &
                          (nu<=nu_max+0.5*rayleigh_freq)) )[0]

        sigma = np.std(residuals)
        nu_err = np.sqrt(6./N) * sigma / (np.pi * amp_max * T)
        amp_err = np.sqrt(2./N) * sigma

        current['frequency'] = np.hstack([nu_max])
        current['e_frequency'] = np.hstack([nu_err])
        current['amplitude'] = np.hstack([amp_max])
        current['e_amplitude'] = np.hstack([amp_err])
        current['phase'] = np.hstack([4.])
        current['e_phase'] = np.hstack([0.1])

        # Optimize frequencey, amplitude, and phase of peak
        # _, current = optimise_(times - t0, residuals, yerr,
        #                         current, fit_freq = False, fit_offset = False)
        residuals_, current = optimise_(times - t0, residuals, yerr,
                                current, fit_freq = True, fit_offset = fit_offset)

        maxiter -= 1
        # residuals = np.hstack(residuals_)
        residuals = np.copy(residuals_)

        nu_res, amp_res = scargle(times, residuals, f0=f0, fn=fn,df=df, norm='amplitude')
        snr_curve_res = get_snr(nu_res, amp_res, use_snr_window=use_snr_window,
                        snr_window=snr_window, snr_range=snr_range)
        noise_res = amp_res / snr_curve_res


        snr_at_nu = current['amplitude'][-1] / noise_res[idx]
        current['snr'] = [snr_at_nu]


        plt.plot(nu, amp, 'k-')
        plt.plot(nu_res, amp_res, '-', color='grey')
        plt.axhline(current['amplitude'][-1],linestyle='--',color='red')
        plt.axvline(nu[idx],linestyle='--',color='red')
        plt.axvline(nu_res[idx],linestyle=':',color='blue')
        plt.show()


        nu = nu_res[:]
        amp = amp_res[:]
        fit_offset = False

        extracted = updated_extracted(extracted, current)

        if snr_at_nu < snr_stop_criteria:
            stopcrit = True



    final = {**extracted}
    residuals, final = optimise_(times-t0, signal, yerr, final, fit_freq=True, fit_offset=True)


    return residuals, final
