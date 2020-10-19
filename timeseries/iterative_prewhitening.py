import numpy as np
import pymc3 as pm
import exoplanet as xo
import theano.tensor as tt
import matplotlib.pyplot as plt

from progress.bar import Bar
from pythia.timeseries.periodograms import scargle
from pythia.timeseries.smoothing import mean_smooth



def map_optimise(x, y, yerr, t0, nu_init, nu_err, amp_init, amp_err,
    fit_offset=False):

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
                    shape=dim,
                    testval=np.zeros(dim)
                    )
        else:
            offset = 0.


        # And a function for computing the full RV model
        def get_sine_model(t, name=""):
            # Sine wave according to parameters

            sine =  offset + amp*np.sin( 2.*np.pi*nu*(t-t0) + phase )
            pm.Deterministic("sine" + name, sine)

            # Sum over planets and add the background to get the full model
            # return pm.Deterministic("sine_model" + name, tt.sum(sine, axis=-1) )
            return pm.Deterministic("sine_model" + name, sine )

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

        plt.plot(x_, xo.eval_in_model(model.sine_model_pred), '--',
            label="model", color='dodgerblue')
        map_soln = xo.optimize(start=model.test_point, vars=[amp,phase],
                        progress_bar=False,tol=1e-10,options={'disp':False})
        map_soln, info = xo.optimize(start=map_soln,return_info=True,
                        progress_bar=False,tol=1e-14,options={'disp':False})

    # print('freq: {} --> {}'.format(nu_init[0], map_soln['nu']))
    plt.plot(x_, map_soln["sine_model_pred"],'-', color='darkorange')

    plt.legend(fontsize=10)
    plt.xlim(x.min(), x.max())
    plt.xlabel("time [days]")
    plt.ylabel("Magnitude")
    plt.ylim(plt.ylim()[::-1])
    plt.show()

    if fit_offset:
        opt_ofst = np.array(map_soln['offset'])
        fit_offset = False
    else:
        opt_ofst = np.array(np.zeros(dim))
    opt_freq = np.array(map_soln['nu'])
    opt_ampl = np.array(map_soln['amp'])
    opt_phase = np.array(map_soln['phase'])

    return np.array([y - map_soln['sine_model']]), opt_ofst, opt_freq, opt_ampl, opt_phase



def get_snr(nu, amp, use_snr_window=True, snr_window=1., snr_range=[23.,24.]):

    if use_snr_window:
        npoints = len( nu[nu<=snr_window])
        mean_ = mean_smooth(amp, npoints)
    else:
        idx = np.where( ((nu>=snr_range[0]) & (nu<=snr_range[1])) )
        mean_ = np.ones_like(nu) * np.median(nu[idx])

    return amp / mean_


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

        # Optimize frequencey, amplitude, and phase of peak
        residuals_, c_, nu_, \
        amp_, phase_= map_optimise(times, residuals, yerr, t0,
                        nu_init=[nu_max], nu_err=[nu_err],
                        amp_init=[amp_max], amp_err=[amp_err],
                        fit_offset=True)
        offsets.append(c_)
        frequencies.append(nu_)
        amplitudes.append(amp_)
        phases.append(phase_)

        maxiter -= 1
        residuals = np.hstack(residuals_)

        nu_res, amp_res = scargle(times, residuals, f0=f0, fn=fn,df=df, norm='amplitude')
        snr_curve_res = get_snr(nu_res, amp_res, use_snr_window=use_snr_window,
                        snr_window=snr_window, snr_range=snr_range)
        noise_res = amp_res / snr_curve_res

        snr_at_nu = amp_ / noise_res[idx]

        stop_criteria.append(snr_at_nu)
        #
        plt.plot(nu, amp, 'k-')
        plt.plot(nu_res, amp_res, '-', color='grey')
        plt.axhline(amp_,linestyle='--',color='red')
        plt.axvline(nu[idx],linestyle='--',color='red')
        plt.axvline(nu_res[idx],linestyle=':',color='blue')
        plt.show()

        # pbar.next()

        nu = nu_res
        amp = amp_res

        if snr_at_nu < snr_stop_criteria:
            stopcrit = True

    # pbar.finish()
    offsets = np.hstack(offsets)
    frequencies = np.hstack(frequencies)
    amplitudes = np.hstack(amplitudes)
    phases = np.hstack(phases)
    stop_criteria = np.hstack(stop_criteria)

    return residuals, offsets, frequencies, amplitudes, phases, stop_criteria



def find_frequency(times,signal,method='scargle',model='sine',full_output=False,
            optimize=0,max_loops=20, scale_region=0.1, scale_df=0.20, model_kwargs=None,
            correlation_correction=True,prewhiteningorder_snr=False,
            prewhiteningorder_snr_window=1.,**kwargs):
    """
    Find one frequency, automatically going to maximum precision and return
    parameters & error estimates.

    This routine makes the frequency grid finer until it is finer than the
    estimated error on the frequency. After that, it will compute (harmonic)
    parameters and estimate errors.

    There is a possibility to escape this optimization by setting scale_df=0 or
    freqregscale=0.

    You can include a nonlinear least square update of the parameters, by
    setting the keyword C{optimize=1} (optimization outside loop) or
    C{optimize=2} (optimization after each iteration).

    The method with which to find the frequency can be set with the keyword
    C{method}, the model used to fit and optimize should be set with C{model}.
    Extra keywords for the model functions should go in C{model_kwargs}. If
    C{method} is a tuple, the first method will be used for the first frequency
    search only. This could be useful to take advantage of such methods as
    fasper which do not allow for iterative zoom-ins. By default, the function looks
    for the highest (or deepest in the case of the pdm method) peak, but instead it is
    possible to go for the peak having the highest SNR before prewhitening by setting
    C{prewhiteningorder_snr} to True. In this case, the noise spectrum is calculated
    using a convolution with a C{prewhiteningorder_snr_window} wide box.

    Possible extra keywords: see definition of the used periodogram function.

    B{Warning}: the timeseries must be B{sorted in time} and B{cannot contain
    the same timepoint twice}. Otherwise, a 'ValueError, concatenation problem'
    can occur.

    Example keywords:
        - 'correlation_correction', default=True
        - 'freqregscale', default=0.5: factor for zooming in on frequency
        - 'dfscale', default = 0.25: factor for optimizing frequency resolution

    Example usage: We generate a sine signal

    >>> times = np.linspace(0,100,1000)
    >>> signal = np.sin(2*np.pi*2.5*times) + np.random.normal(size=len(times))

    Compute the frequency

    >>> parameters, pgram, model = find_frequency(times,signal,full_output=True)

    Make a plot:

    >>> p = pl.figure()
    >>> p = pl.axes([0.1,0.3,0.85,0.65])
    >>> p = pl.plot(pgram[0],pgram[1],'k-')
    >>> p = pl.xlim(2.2,2.8)
    >>> p = pl.ylabel('Amplitude')
    >>> p = pl.axes([0.1,0.1,0.85,0.2])
    >>> p = pl.plot(pgram[0][:-1],np.diff(pgram[0]),'k-')
    >>> p = pl.xlim(2.2,2.8)
    >>> p,q = pl.xlabel('Frequency (c/d)'),pl.ylabel('Frequency resolution $\Delta F$')

    ]]include figure]]timeseries_freqanalyse_06.png]

    @rtype: record array(, 2x1Darray, 1Darray)
    @return: parameters and errors(, periodogram, model function)
    """
    if model_kwargs is None:
        model_kwargs = dict()
    #-- initial values
    e_f = 0
    freq_diff = np.inf
    prev_freq = -np.inf
    counter = 0

    f_max = np.inf
    f_min = 0.#-np.inf

    #-- calculate periodogram until frequency precision is
    #   under 1/10th of correlation corrected version of frequency error
    method_kwargs = kwargs.copy() # don't modify the dictionary the user gave

    while freq_diff>e_f/10.:
        #-- possibly, we might want to use different periodograms for the first
        #   calculation than for the zoom in, since some periodograms are faster
        #   than others but do not have the ability to 'zoom in' (e.g. the FFT)
        if freq_diff==np.inf and not isinstance(method,str):
            method_ = method[1]
            method = method[0]  # override method to be a string the next time
        #-- calculate periodogram
        freqs,ampls = getattr(pergrams,method)(times,signal,**method_kwargs)
        f0,fn,df = freqs[0],freqs[-1],freqs[1]-freqs[0]
        #-- now use the second method for the zoom-ins from now on
        if freq_diff==np.inf and not isinstance(method,str):
            method = method_
        #-- extract the frequency: this part should be generalized, but for now,
        #-- instead of going for the highest peak, let's get the most significant one
        if prewhiteningorder_snr:
            if counter == 0: #we calculate a noise spectrum with a convolution in a 1 d-1 window
                windowlength = float(prewhiteningorder_snr_window)/(freqs[1]-freqs[0])
                window = np.ones(int(windowlength))/float(windowlength)
                ampls_ = np.concatenate((ampls[::-1],ampls,ampls[::-1])) #we mirror the amplitude spectrum on both ends so the convolution will be better near the edges
                noises_ = np.convolve(ampls_, window, 'same')
                noises = np.split(noises_,3)[1] #and we recut the resulted convolution to match the original frequency range
                freqs_old = np.copy(freqs)
                noises_old = np.copy(noises)
            else:
                noises = np.interp(freqs,freqs_old,noises_old) #we use the original noise spectrum in this narrower windows too, which should save some time, and avoid the problem of having a wider window for the SNR calculation than the width of the zoom-in window
            frequency = freqs[np.argmax(ampls/noises)]
        else:
            frequency = freqs[np.argmax(ampls)]
        if full_output and counter==0:
            freqs_,ampls_ = freqs,ampls
        #-- estimate parameters and calculate a fit, errors and residuals
        params = getattr(fit,model)(times,signal,frequency,**model_kwargs)
        if hasattr(fit,'e_'+model):
            errors = getattr(fit,'e_'+model)(times,signal,params,correlation_correction=correlation_correction)
            e_f = errors['e_freq'][-1]
        #-- possibly there are not errors defined for this fitting functions
        else:
            errors = None
        #-- optimize inside loop if necessary and if we gained prediction
        #   value:
        if optimize==2:
            params_,errors_,gain = fit.optimize(times,signal,params,model)
            if gain>0:
                params = params_
                logger.info('Accepted optimization (gained %g%%)'%gain)

        #-- improve precision
        freq_diff = abs(frequency-prev_freq)
        prev_freq = frequency
        freq_region = fn-f0
        f0 = max(f_min,frequency-freq_region*scale_region/2.)
        fn = min(f_max,frequency+freq_region*scale_region/2.)
        df *= scale_df
        method_kwargs['f0'] = f0
        method_kwargs['fn'] = fn
        method_kwargs['df'] = df
        #-- possibilities to escape iterative zoom in
        #print '---> {counter}/{max_loops}: freq={frequency} ({f0}-->{fn}/{df}), e_f={e_f}, freq_diff={freq_diff}'.format(**locals()),max(ampls)
        if scale_region==0 or scale_df==0:
            break
        if counter >= max_loops:
            logger.error("Frequency precision not reached in %d steps, breaking loop"%(max_loops))
            break
        if (fn-f0)/df<5:
            logger.error("Frequency precision not reached with stepsize %e , breaking loop"%(df/scale_df))
            break
        counter += 1
    #-- optimize parameters outside of loop if necessary:
    if optimize==1:
        params_,errors_,gain = fit.optimize(times,signal,params,model)
        if gain>0:
            params = params_
            logger.info('Accepted optimization (gained %g%%)'%gain)
    #-- add the errors to the parameter array if possible
    if errors is not None:
        params = numpy_ext.recarr_join(params,errors)
    logger.info("%s model parameters via %s periodogram:\n"%(model,method)+pl.mlab.rec2txt(params,precision=8))
    #-- when full output is required, return parameters, periodogram and fitting
    #   function
    if full_output:
        mymodel = getattr(evaluate,model)(times,params)
        return params,(freqs_,ampls_),mymodel
    else:
        return params
