#######################################################################################
# Contact:  -   castillo.renato.90@gmail.com
#           -   castillo@biomomentum.com
######################################################################################
import numpy as np
import scipy as sp

def ResidualStandardError(x, xfit, predictors):
    """

    Residual Error from the fit to function.

    Parameters
    ----------
    x         : independent variable values (assumed to be error-free) 
    xfit      : signal fit of xfit                                     
    predictor : number of predictors of the function
    
    Returns
    -------
    ser       : standard residual error                               

    """
    N = len(x)
    ssr = np.sum((x - xfit)**2)
    ser = np.sqrt(ssr/(N - predictors))
    return ser
def leasqrfunc(Params, Time, data):
    """

    Sinus function to fit the data.

    Parameters
    ----------
    Params    : array of funtions parameters
    
    Returns
    -------
    Function values at Time                               

    """
    return (Params[0] + Params[1]*np.sin(2*np.pi*Params[2]*Time + Params[3])) - data
def funfminsearch(phase, Offset, Amplitude, f, time, x):
    """

    Function to minimize error.

    Parameters
    ----------
    phase     : sinus phase 
    Offset    : sinus offset                                     
    Amplitude : sinus amplitude
    f         : sinus frequency
    time      : time array (s)
    
    Returns
    -------
    ser       : standard residual error                              

    """
    x_fit = Offset + Amplitude*np.sin(2*np.pi*f*time + phase)
    ser = ResidualStandardError(x, x_fit, 4)
    return ser
def FitSinusoid(time, data, freq, Interval, method = "trf", lossType = "soft_l1", fscale = 0.001):
    """

    Sinusoid fit on data using levenberg-marquardt non-linear method.

    Parameters
    ----------
    time     : time vector (s) 
    data     : data vector (data units)                                     
    freq     : frequency guess (Hz)
    Interval : [start point for data, endpoint for data] (list of intergers)
    
    Returns
    -------
    Offset       : offset (data units)
    Amplitude : amplitude (data units)
    fn        : frequency (Hz)
    phase     : phase (rad)
    ser       : residual standard error (4 preditors) (data units)                              

    """
    # Reorganise time vector
    fs = 1/np.mean(np.diff(time)) # compute sampling frequency
    
    if len(Interval) == 0 or Interval[0] == 0 or Interval[1] == len(data):
        start = 0
        finish = len(time) - 1
        t_fit = time
    else:
        start = Interval[0] 
        finish = Interval[1]
        t1 = np.flip(-1*time[:start-1]) - 1/fs
        t2 = time[start:finish] - time[start]
        t3 = time[finish+1:]
        t3 = t3 - t3[0] + t2[-1] + 1/fs
        t_fit = np.hstack((t1, t2, t3))
    
    # Trim Data
    x = data[start:finish] 
    t = time[start:finish] - time[start] # time stats at 0 s
    
    # Compute FFT
    y = np.fft.fft(x)
    L = len(y)
    P2 = y/L
    P1 = P2[:int(L/2+1)]
    P1[1:-2] = 2*P1[1:-2] # Single Spectrum FFT
    
    to1 = 1e-6
    P1[abs(P1) < to1] = 0 
    theta = np.angle(P1) # phase angle
    
    # Parameters extraction from FFT
    idx_Amplitude_fft = np.argmax(abs(P1[1:]))
    idx_Amplitude_fft += 1
    phase_fft = np.pi*(0.5 + theta[idx_Amplitude_fft]/np.pi) # phase (rad): shift from pi/2 (cosine to sine)
    
    # Trust Region Algorithm (non-liner least square)
    # Initial Guess
    Offset = np.mean(x) # offset (data units)
    Amplitude = np.ptp(x)/2 # amplitude (data units)
    fn = freq # frequency (Hz)
    
    # Check fit for FFT phase
    x_fit_fft = Offset + Amplitude*np.sin(2*np.pi*fn*t + phase_fft)
    ser_fft = ResidualStandardError(x, x_fit_fft, 4) 
    
    # Minimize error to find phase initial guess
    phase_fminbnd, ser_fminbnd, _, _ = sp.optimize.fminbound(lambda phase : funfminsearch(phase, Offset, Amplitude, fn, t, x), -np.pi, np.pi, full_output=True)
    
    if ser_fft > ser_fminbnd: # check which phase has smaller ser 
        phase = phase_fminbnd
    else:
        phase = phase_fft
    
    Param0 = np.array([Offset, Amplitude, fn, phase]) # Initial Guess
    # Sinusoid function = Offset + Amplitude*sin(2*pi*fn*t_fit + phase)
    Param = sp.optimize.least_squares(leasqrfunc, Param0, loss=lossType, f_scale=fscale, args = (t, x), method=method)
    
    xfit = leasqrfunc(Param.x, t,0) 
    ser = ResidualStandardError(x, xfit, 4) # residual standard error (4 predictors)
    
    # Get Parameters of Sinusoid fit
    Offset, Amplitude, fn, phase = Param.x[0], Param.x[1], Param.x[2], Param.x[3] 
    
    # Compute fit for all time
    
    x_fit = leasqrfunc(Param.x, t_fit,0)
    t_fit -= t_fit[0]
    return Offset, Amplitude, fn, phase, x_fit, t_fit, ser
def trim_edges(data, fs, fq, criteria = None, distance = None):
    """

    Trim Data.

    Parameters
    ----------
    data : data array (data units)
    fs   : sampling frequency (Hz)
    fq   : signal frequency from mach-1 file (Hz)
    
    Returns
    -------
    Interval : [start point for data, endpoint for data] (list of intergers)                              

    """
    Amp = abs(data - np.mean(data))
    if criteria is not None:
        trim = []
        for idx in range(len(data)):
            if data[idx] - np.mean(data) <= criteria:
                break
        trim.append(idx)
        for idx in reversed(range(len(data))):
            if data[idx] - np.mean(data) <= criteria:
                break
        trim.append(idx)
        amplitude = np.ptp(data[trim[0]:trim[1]])/2
    else:
        amplitude = np.max(Amp)
    # Data normalization
    Amp_normalized = Amp/amplitude
    
    # Find peak algorithm only counts when 60% of amplitude is reached and distance fs/(2*fq)
    if distance is None:
        peaks, _ = sp.signal.find_peaks(Amp_normalized, height = (0.8,1.2), distance = int(fs/(2*fq)) - 5)
    else:
        peaks, _ = sp.signal.find_peaks(Amp_normalized, height = (0.8,1.2), distance = distance)
    
    if len(peaks) < 1:
        Interval = [0, len(data)]
    else:
        Interval = [peaks[0], peaks[-1]]
    
    return Interval, Amp_normalized, peaks

def butterworth_filter(data):
    """

    1D low pass butterworth filter.

    Parameters
    ----------
    data     : data array (data units)
    
    Returns
    -------
    filtered_data : data array filtered (data units)                             

    """
    fc = 20 # Cuttof frequency (Hz)
    fs = 100 # Sampling frequency (Hz)
    b, a = sp.signal.butter(1, fc/(fs/2)) # Low pass filter order 1
    filtered_data1 = sp.signal.filtfilt(b,a,data)
    filtered_data2 = np.flip(sp.signal.filtfilt(b,a,np.flip(filtered_data1)))
    filtered_data = np.hstack((data[:2],filtered_data2[2:-2], data[-2:]))
    return filtered_data

def linear_least_square(x,y):
    """

    1D low pass butterworth filter.

    Parameters
    ----------
    x     : data array - independent variable (data units)
    y     : data array - dependent variable (data units)
    
    Returns
    -------
    A : matrix of linear fit  
    curveFit : linear fit on data
    Rsq_adj : R-squared                         

    """
    X = np.zeros((len(x),2))
    Y = y
    X[:,0] = x
    X[:,1] = 1
    A = np.linalg.solve(np.dot(X.transpose(),X),np.dot(X.transpose(), Y))
    N = len(Y)
    curveFit = np.dot(X,A)
    poly_order = 1
    mse = np.sum((Y - curveFit)**2)/N
    Rsq = 1 - mse/np.var(Y)
    Rsq_adj = 1 - (1 - Rsq)*(N - 1)/(N - poly_order - 1)
    return A, curveFit, Rsq_adj