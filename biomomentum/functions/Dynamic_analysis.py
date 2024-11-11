import numpy as np
import scipy as sp

from scipy.interpolate import interp1d
from scipy.signal import periodogram, detrend

from .utils import ResidualStandardError

def leasqrfunc(Params, Time, data):
    """
    Sinus function to fit the data.
    Args:
        Params    : array of funtions parameters
        data      : array of data
    
    Returns:
    Difference between model and data                              
    """
    return (Params[0] + Params[1]*np.sin(2*np.pi*Params[2]*Time + Params[3])) - data

def leasqrfunc_sync(Params, Time, data_1, data_2):
    """
    Sinus function to fit the data1 and data2 with same frequency.

    Args:
        Params    : array of funtions parameters
        data_1    : array of data series 1
        data_2    : array of data series 2
    
    Returns:
        Difference between model and data series                              
    """
    model_1 = data_1 - (Params[0] + Params[1]*np.sin(2*np.pi*Params[2]*Time + Params[3]))    
    model_2 = data_2 - (Params[4] + Params[5]*np.sin(2*np.pi*Params[2]*Time + Params[6]))
    return np.concatenate([model_1, model_2])

def initialize_phase(time, data, amplitude, fcycle):
    """
    Initialize the phase of the sinus based on the fft of the signal

    Args:
        Time (np.array): time in s
        data (np.array): data to fit the sin function
        amplitude (float)    : amplitude initial guess
        fcycle (float): frequency initial guess
    
    Returns:
        phase (float): initial guess of the sin phase                             
    """
    y = np.fft.fft(data)
    N = len(y)
    P2 = y/N
    P1 = P2[:int(N/2 + 1)]
    P1[1:-2] = 2*P1[1:-2]
    theta = np.angle(P1)
    idx_Amplitude_fft = np.argmax(abs(P1[1:]))
    idx_Amplitude_fft += 1
    phase_fft = np.pi*(0.5 + theta[idx_Amplitude_fft]/np.pi)    
    Offset = np.mean(data) 
    x_fit_fft = Offset + amplitude*np.sin(2*np.pi*fcycle*time + phase_fft)
    ser_fft = ResidualStandardError(data, x_fit_fft, 4)
    phase_fminbnd, ser_fminbnd, _, _ = sp.optimize.fminbound(lambda phase : funfminsearch(phase, Offset, amplitude, fcycle, time, data), -np.pi, np.pi, full_output=True)
    if ser_fft > ser_fminbnd: 
        phase = phase_fminbnd
    else:
        phase = phase_fft
    return phase

def funfminsearch(phase, Offset, Amplitude, f, time, x):
    """
    Function to minimize error.

    Args:
        phase     : sinus phase 
        Offset    : sinus offset                                     
        Amplitude : sinus amplitude
        f         : sinus frequency
        time      : time array (s)
    
    Returns:
        ser       : standard residual error                              
    """
    x_fit = Offset + Amplitude*np.sin(2*np.pi*f*time + phase)
    ser = ResidualStandardError(x, x_fit, 4)
    return ser

def FitSinusoid(time,  freq, data_1, data_2 = None, method = "trf", lossType = "soft_l1", fscale = 0.001, trim_data = False):
    """
    Sinusoid fit on data using least-squares.

    Args:
        time     : time vector (s) 
        data_1   : data vector (data units)  
        data_2   : data vector (data units) to sync with data_1 (None by default)                                   
        freq     : frequency guess (Hz)
        method   : method for the least-squares (see scipy.optimize.least_squares)
        lossType : loss for the method applied (see scipy.optimize.least_squares)
        trim_data: bool to indicate whether to trim the data or not
    
    Returns:
        Params       : if data_2 is not given then -
                        Params[0] : Sin offset of data_1
                        Params[1] : Sin amplitude of data_1
                        Params[2] : Sin frequency of data_1
                        Params[3] : Sin phase of data_1
                    if data_2 is given
                        Params[2] : Sin frequency of data_1 and data_2
                        Params[4] : Sin offset of data_2
                        Params[5] : Sin amplitude of data_2
                        Params[6] : Sin phase of data_2
        data_sin  : Sin fit into data
        data_extra: Data trimmed if flag passed (for plotting)
        ser_data  : Residual standard error of the fit                              
    """
    dt = np.mean(np.diff(time))
    Nt = len(time)
    fs = 1/dt

    time_correct = np.round(np.arange(0, Nt*dt, dt), 2)
    if len(np.where(abs(time - time_correct) > 0)[0]) != 0:
        interp_P = interp1d(time, data_1, kind = 'cubic')
        data_1 = interp_P(time_correct)
        if data_2 is not None:
            interp_F = interp1d(time, data_2, kind = 'cubic') 
            data_2 = interp_F(time_correct)
    if trim_data:
        interval = trim_edges(data_1)
    else:
        interval = np.array([0, len(data_1)])
    data_1 = data_1[interval[0]:interval[1]]
    N_cycles = round(fs/freq)
    Nt = int(np.floor(len(data_1)/N_cycles)*N_cycles)
    t = np.arange(0, Nt*dt, dt)
    data_1_trimmed = data_1[:Nt]
    nfft = 2**int(np.ceil(np.log2(Nt)))
    f, Pxx = periodogram(detrend(data_1_trimmed, type = 'constant'), fs = fs, nfft = nfft)
    idx_fdrive = np.argmax(Pxx)
    f_cycle = f[idx_fdrive]
    fs = freq*fs/f_cycle
    dt = 1/fs
    N_cycle = round(fs/f_cycle)
    t_new = np.arange(0, Nt*dt, dt)
    interp_P = interp1d(t, data_1_trimmed, kind = 'linear', fill_value = 'extrapolate')
    data_1_extra = interp_P(t_new)
    Nt_extra = int(np.floor(len(data_1_extra)/N_cycle)*N_cycle)
    t_extra = t_new[:Nt_extra]
    data_1_extra = data_1_extra[:Nt_extra]
    data_1_array = np.reshape(data_1_extra, (N_cycle, -1))
    data_1_array = detrend(data_1_array, axis = 0)
    data_1_Amp = np.mean(np.max(data_1_array, axis = 0))
    sign_data_1 = np.sign(data_1_array[0,0])
    data_1_phase_guess = initialize_phase(t_extra, data_1_extra, data_1_Amp, f_cycle)
    if sign_data_1 <= 0:
        data_1_phase_guess = -data_1_phase_guess
    if data_2 is None:
        Params_data_1_initial = [np.mean(data_1_extra), data_1_Amp, f_cycle, data_1_phase_guess]
        lb = [-np.inf, 0, 0, data_1_phase_guess - np.pi/2]
        ub = [np.inf, np.inf, np.inf, data_1_phase_guess + np.pi/2] 
        Params_data_1 = sp.optimize.least_squares(leasqrfunc, Params_data_1_initial, loss=lossType, f_scale=fscale, bounds = (lb, ub), args = (t_extra, data_1_extra), method=method)
        data_1_sin = Params_data_1.x[0] + Params_data_1.x[1] * np.sin(2 * np.pi * Params_data_1.x[2] * t_extra + Params_data_1.x[3])
        ser = ResidualStandardError(data_1_extra, data_1_sin, 4) 
        return Params_data_1.x, data_1_sin, t_extra, data_1_extra, ser 
    else:
        data_2 = data_2[interval[0]:interval[1]]
        data_2_trimmed = data_2[:Nt]
        interp_F = interp1d(t, data_2_trimmed, kind = 'linear', fill_value = 'extrapolate')
        data_2_extra = interp_F(t_new)
        data_2_extra = data_2_extra[:Nt_extra]
        data_2_array = np.reshape(data_2_extra, (N_cycle, -1))
        data_2_array = detrend(data_2_array, axis = 0)
        data_2_Amp = np.mean(np.max(data_2_array, axis = 0))
        sign_data_2 = np.sign(data_2_array[0,0])
        data_2_phase_guess = initialize_phase(t_extra, data_2_extra, data_2_Amp, f_cycle)
        if sign_data_2 <= 0:
            data_2_phase_guess = -data_2_phase_guess
        Params_data_1_2_initial = [np.mean(data_1_extra), data_1_Amp, f_cycle, data_1_phase_guess, np.mean(data_2_extra), data_2_Amp, data_2_phase_guess]
        lb = [-np.inf, 0, 0, data_1_phase_guess - np.pi/2, -np.inf, 0, data_2_phase_guess - np.pi/2]
        ub = [np.inf, np.inf, np.inf, data_1_phase_guess + np.pi/2, np.inf, np.inf, data_2_phase_guess + np.pi/2] 
        Params_data_1_2 = sp.optimize.least_squares(leasqrfunc_sync, Params_data_1_2_initial, loss=lossType, f_scale=fscale, bounds=(lb, ub), args=(t_extra, data_1_extra, data_2_extra))
        data_1_sin = Params_data_1_2.x[0] + Params_data_1_2.x[1] * np.sin(2 * np.pi * Params_data_1_2.x[2] * t_extra + Params_data_1_2.x[3])
        data_2_sin = Params_data_1_2.x[4] + Params_data_1_2.x[5] * np.sin(2 * np.pi * Params_data_1_2.x[2] * t_extra + Params_data_1_2.x[6])
        ser_data_1 = ResidualStandardError(data_1_extra, data_1_sin, 4)
        ser_data_2 = ResidualStandardError(data_2_extra, data_2_sin, 4)
        return Params_data_1_2.x, data_1_sin, data_2_sin, t_extra, data_1_extra, data_2_extra, ser_data_1, ser_data_2
        
def trim_edges(data, criteria = 0.8):
    """
    Trim Data.

    Args:
        data : data array (data units)
        fs   : sampling frequency (Hz)
    
    Returns:
        Interval : [start point for data, endpoint for data]                              
    """
    data_Amp = abs(data - np.mean(data))
    data_Amp -= np.min(data_Amp)
    data_Amp_norm = data_Amp / np.max(data_Amp)
    peaks, _ = sp.signal.find_peaks(data_Amp_norm, height = criteria)
    Interval = np.array([np.where(data_Amp_norm[:peaks[0]] >= criteria*data_Amp_norm[peaks[0]])[0][0],
                    np.where(data_Amp_norm[peaks[-1]:] >= criteria*data_Amp_norm[peaks[-1]])[0][-1] + peaks[-1]])
    return Interval

def butterworth_filter(data):
    """
    1D low pass butterworth filter.

    Args:
        data     : data array (data units)
    
    Returns:
        filtered_data : data array filtered (data units)                             
    """
    fc = 20 # Cuttof frequency (Hz)
    fs = 100 # Sampling frequency (Hz)
    b, a = sp.signal.butter(1, fc/(fs/2)) # Low pass filter order 1
    filtered_data1 = sp.signal.filtfilt(b,a,data)
    filtered_data2 = np.flip(sp.signal.filtfilt(b,a,np.flip(filtered_data1)))
    filtered_data = np.hstack((data[:2],filtered_data2[2:-2], data[-2:]))
    return filtered_data
