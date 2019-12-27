import import numpy as np
from numpy import cos,sin,pi

from pythia.timeseries.pergrams import scargle


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
