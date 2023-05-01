import scipy as sp
import numpy as np


def updated_extracted(extracted, current):

    for key in current:
        if 'offset' in key:
            extracted[key] = current[key]
        else:
            extracted[key] = np.hstack( [extracted[key],current[key]] )

    return extracted



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
