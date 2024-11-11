#%%
####################################### RESET WORKSPACE VARIABLES #######################################
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
        del globals()[var]

clear_all()
#%%
####################################### Import useful librairies into the workspace #######################################
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.signal import periodogram, detrend

from matplotlib.patches import Polygon
from HayesElasticModel_PYTH import *
from Sinus_Analysis_PYTH import *
from read_mach_1_file_PYTH import read_mach_1_file, read_mach_1_files, select_mach_1_file
from Stress_Relaxation_Fit_PYTH import *
from Analysis_Map_file_PYTH import select_files_dir, get_subSurfaces, interpolateMAP
#%%
####################################### FUNCTIONS #######################################
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: get_super
% Description: Insert super script char on string
% Inputs:   x - Character to super script
% Outputs:  char in super script format
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def get_super(x): 
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s)) 
    return x.translate(res)

def FitSinusoid(time, data, freq, Interval, method = "trf", lossType = "soft_l1", fscale = 0.001):
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
#%%
# Testing Dynamic - Sinusoid Analysis
mach_1_data = read_mach_1_file(select_mach_1_file())
#%%
flag_correct_fdrive = True
metrics = {} 
Amps_total = []
average = np.zeros(10)
fdrive_set = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.05, 0.1, 0.2, 0.5, 1, 2])
Moduli = np.zeros((24, 4))
#%%
function = 'Sinusoid-5'
units_gf = "Single" in mach_1_data[function]["<INFO>"]["Load Cell Type:"]
Time = mach_1_data[function]["<DATA>"]["Time"]
posZ = mach_1_data[function]["<DATA>"]["Position (z)"]
Fz = mach_1_data[function]["<DATA>"]["Fz"]
dt = np.mean(np.diff(Time)) 
Nt = len(Time) 
fs = 1/dt 
frequency = float(mach_1_data[function]["<Sinusoid>"]["Frequency, Hz:"])
Time_correct = np.arange(0, Nt*dt, dt)
Time_correct = np.round(Time_correct,2) 
interp_F = interp1d(Time, Fz, kind = 'cubic') 
interp_P = interp1d(Time, posZ, kind = 'cubic')

F_correct = interp_F(Time_correct)
P_correct = interp_P(Time_correct)
P_c = P_correct

P_mean = np.mean(P_correct)
P_amp = np.abs(P_correct-P_mean)
P_amp -=  np.min(P_amp)
P_amp_norm = P_amp / np.max(P_amp)
F_mean = np.mean(F_correct)
F_amp = np.abs(F_correct-F_mean)
F_amp -= np.min(F_amp)

Amp_pct = 0.8 

peaks, _ = sp.signal.find_peaks(P_amp_norm, height = Amp_pct)
idx_Int = np.array([np.where(P_amp_norm[:peaks[0]] >= 0.8*P_amp_norm[peaks[0]])[0][0],
                    np.where(P_amp_norm[peaks[-1]:] >= 0.8*P_amp_norm[peaks[-1]])[0][-1] + peaks[-1]])

P_correct = P_correct[idx_Int[0]:idx_Int[1]]
F_correct = F_correct[idx_Int[0]:idx_Int[1]]
Nt = len(P_correct)
Ncycle = round(fs/frequency)
Nc = 1
while Nt/(Nc*Ncycle) != round(Nt/(Nc*Ncycle)):
    Nt -= 1           
t = np.arange(0, Nt*dt,dt)
P = P_correct[:Nt]
F = F_correct[:Nt]
#%%
nfft = 2**np.ceil(np.log2(Nt)).astype(int)
def nextpow2(a):
    return 2**int(np.ceil(np.log2(a)))
nfft = nextpow2(Nt)

f, Pxx = periodogram(detrend(P, type ='constant'), fs = fs, nfft = nfft)
d_P = detrend(P, type = 'constant')
y = np.fft.fft(d_P)
L = len(y)
P2 = y/L
P1 = P2[:int(L/2+1)]
P1[1:-2] = 2*P1[1:-2]  

to1 = 1e-6



 
idx_fdrive = np.argmax(Pxx)
s = frequency*fs/f[idx_fdrive]
dt = 1/fs
fcycle = f[idx_fdrive]
Ncycle = round(fs/fcycle)
tcycle = np.arange(0, Nc*Ncycle*dt, dt)
t_new = np.arange(0, Nt * dt, dt)

interp_F = interp1d(t, F, kind='linear', fill_value='extrapolate')
interp_P = interp1d(t,P, kind='linear',fill_value='extrapolate')
F = interp_F(t_new)
P = interp_P(t_new)
t = t_new
#%%
fig = plt.figure(figsize=(20,15))
axes = fig.add_subplot(111)
#axes.semilogy(f, Pxx)
axes.plot(Pxx)
axes.plot(np.abs(P1))
axes.set_ylabel("Position Z (mm)",size=20)
axes.set_xlabel("Time (s)",size=20)
axes.tick_params(axis = 'both', which = 'major', labelsize=15)
plt.grid()
plt.show()
#%%
#Adjust length of singals
Nt = len(P)
while Nt % (Nc * Ncycle) != 0:
    Nt-=1
t = t[:Nt]
P = P[:Nt]
F = F[:Nt]
              
P_array = np.reshape(P, (Nc*Ncycle,-1))
F_array = np.reshape(F, (Nc*Ncycle,-1))
    #Detrend
P_array = detrend(P_array, axis=0) #Remove trend along each column
F_array = detrend(F_array, axis=0)
    #Average of the maximum values
P_avg = np.mean(np.max(P_array, axis = 0))
F_avg = np.mean(np.max(F_array, axis = 0))
    #Estimate initial guess parameters
sgn_P = np.sign(P_array[0, 0])
sgn_F = np.sign(F_array[0, 0])
if sgn_P > 0:
    x0_P_phase = np.pi / 2
else:
    x0_P_phase = -np.pi / 2
if sgn_F > 0:
    x0_F_phase = np.pi / 2
else:
    x0_F_phase = -np.pi / 2
#Fit function
def fun(x,t):
    return np.column_stack((
        x[0] + x[1] * np.sin(2* np.pi * x[2] * t + x[3]),
        x[4] + x[5] * np.sin(2*np.pi * x[2] * t + x[6])
        ))
            #Error function
def fun_error (x,t,P,F):
    P_F_est = fun(x,t)
    return np.concatenate([P - P_F_est[:,0], F - P_F_est[:,1]])       
#Initial guess and bounds (upper and lower)
x0 = [np.mean(P), P_avg, fcycle, x0_P_phase, np.mean(F), F_avg, x0_F_phase]
lb = [-np.inf, 0, 0, x0_P_phase - np.pi/2, -np.inf, 0, x0_F_phase - np.pi/2]
ub = [np.inf, np.inf, np.inf, x0_P_phase + np.pi/2, np.inf, np.inf, x0_P_phase + np.pi/2]           
# Perform the fit using least_squares
res = least_squares(fun_error, x0, bounds=(lb, ub), args=(t, P, F))
# Extract fit parameters. An array of parameter values that minimize the error function. 
fit_params = res.x
#%%
fig = plt.figure(figsize=(20,15))
axes = fig.add_subplot(111)
axes.plot(Time_correct, P_c, "-b")
axes.plot(Time_correct[idx_Int[0]:idx_Int[1]], P_correct, "-r")
axes.plot(t, P, "-g")
axes.set_ylabel("Position Z (mm)",size=20)
axes.set_xlabel("Time (s)",size=20)
axes.tick_params(axis = 'both', which = 'major', labelsize=15)
plt.grid()
plt.show()
#%%
PeakToPeak = 1
function = "Sinusoid-5"
units_gf = "Single" in mach_1_data[function]["<INFO>"]["Load Cell Type:"]
Time = mach_1_data[function]["<DATA>"]["Time"]
posZ = mach_1_data[function]["<DATA>"]["Position (z)"]
Fz = mach_1_data[function]["<DATA>"]["Fz"]
fs = 1/np.mean(np.diff(Time))
if units_gf:
    Fz = Fz*0.00980665
frequency = float(mach_1_data[function]["<Sinusoid>"]["Frequency, Hz:"])
cycles = float(mach_1_data[function]["<Sinusoid>"]["Number of Cycles:"])
thickness = float(mach_1_data[function]["<Sinusoid>"]["Amplitude, mm:"])/(PeakToPeak/200) 
Interval, Amp_normalized, peaks = trim_edges(posZ, fs, frequency, distance = int(fs/(4*frequency)) - 5)
Fz_filtered = butterworth_filter(Fz)
posZeq, posZamp, fn1, phi1, posZ_fit, t_fit, ser_posZ = FitSinusoid(Time, posZ, frequency, Interval, method="trf", lossType="soft_l1", fscale=0.0001)
Fzeq,Fzamp,fn2,phi2,Fz_fit, _, ser_Fz = FitSinusoid(Time, Fz, fn1, Interval, method="trf", lossType="soft_l1", fscale=0.0001)
fn = fn1
delta = abs(phi2 - phi1)
delta_deg = delta*180/np.pi
if delta_deg >= 180:
    delta_deg = 360 - delta_deg
k = Fzamp/posZamp
E1 = k*np.cos(delta)
E2 = k*np.sin(delta)
E_dyn = np.linalg.norm(np.array([E1, E2]))      

# Plotting Sinus Fit unto data
fig = plt.figure(figsize=(20,15))
axes = fig.add_subplot(111)
axes.plot(Time, posZ, "+")
axes.plot(t_fit, posZ_fit, "--r")
axes.set_ylabel("Position Z (mm)",size=20)
axes.set_xlabel("Time (s)",size=20)
axes.tick_params(axis = 'both', which = 'major', labelsize=15)
plt.grid()
plt.show()

fig = plt.figure(figsize=(20,15))
axes = fig.add_subplot(111)
axes.plot(Time, Fz, "o")
axes.plot(t_fit, Fz_fit, "--r")
axes.set_ylabel("Force (gf)",size=20)
axes.set_xlabel("Time (s)",size=20)
axes.tick_params(axis = 'both', which = 'major', labelsize=15)
plt.grid()
plt.show()

fig = plt.figure(figsize=(20,15))
axes = fig.add_subplot(111)
axes.plot(posZ, Fz, "o")
axes.plot(posZ_fit, Fz_fit, "--r")
axes.set_ylabel("Force (gf)",size=20)
axes.set_xlabel("Position Z (mm)",size=20)
axes.tick_params(axis = 'both', which = 'major', labelsize=15)
plt.grid()
plt.show()

print("=======================================")
print(function + " :")
print("Frequency (Hz): ", np.round(fn1,6))
print("Position-z Amplitude (mm): ", np.round(posZamp,6))
print("Fz Amplitude (N): ", np.round(Fzamp,6))
print("E*: ", np.round(E_dyn,6))
print("Phase lag (deg): ", np.round(delta_deg,6))
print("Residual Standard Error Position-z (mm): ", np.round(ser_posZ,6))
print("Residual Standard Error Force-z (N): ", np.round(ser_Fz,6))
print("===============================================")










