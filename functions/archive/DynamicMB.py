import os
import re
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.signal import periodogram, detrend
from read_mach_1_file_PYTH import read_mach_1_file, read_mach_1_files, select_mach_1_file
#%%

#Setting plot style in 
plt.rc('text', usetex = True)
#plt.rc('front',family = 'serif')
plt.rc('axes',labelsize=16, linewidth = 0.5)
plt.rc('legend', fontsize = 12)
plt.rc('lines', linewidth = 2, markersize = 4)
#%%
#Load MACH1 files 
data_headers_selection = None # Select what data of mach-1 files to load - None if all data has to be loaded
sampleName = "23" # Enter the name of sample or None to load all files in selected directory
all_data_Mach_1, mach_1_dir = read_mach_1_files(1, data_headers_selection, sampleName, False) #Select folder with the data. "all_data_Mach1" is an array with samples' IDs. 
Samples = list(all_data_Mach_1.keys()) #Array with Sample's IDs. 
#%%
flag_correct_fdrive = True
metrics = {} #Where data will be stored. A dictionary is equivalent to a structure in Matlab.
Amps_total = []
average = np.zeros(10)
fdrive_set = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.05, 0.1, 0.2, 0.5, 1, 2])
Moduli = np.zeros((24, 4))
print("Working on DATA file " + Samples[0] + ": ", )
IDName = Samples[0]
for function in all_data_Mach_1[Samples[0]]:
    if "Sinusoid" in function:
        print("Data for: " + function)
        Time = all_data_Mach_1[IDName][function]["<DATA>"]["Time"]
        posZ = all_data_Mach_1[IDName][function]["<DATA>"]["Position (z)"]
        Fz = all_data_Mach_1[IDName][function]["<DATA>"]["Fz"]
        if "object" in str(Time.dtype):
            empty_indexes = [i for i, x in enumerate(Fz) if x == '']
            Time = np.array(np.delete(Time, empty_indexes), dtype = float)
            posZ = np.array(np.delete(posZ, empty_indexes), dtype = float)
            Fz = np.array(np.delete(Fz, empty_indexes), dtype = float)
        dt = np.mean(np.diff(Time)) #Calculate the mean difference between consecutive elements of Time
        Nt = len(Time) #length of the Time vector
        fs = 1/dt #sampling frequency
        Time_correct = np.arange(0, Nt*dt, dt)
        Time_correct = np.round(Time_correct,2) #New time vector for interpolation with two decimals
        interp_F = interp1d(Time, Fz, kind = 'cubic') #interp1d is used for cubic spline interpolation (equivalent to MATLAB's 'spline')
        interp_P = interp1d(Time, posZ, kind = 'cubic')
        F_correct = interp_F(Time_correct)#perform the interpolation
        P_correct = interp_P(Time_correct)
        #TRIM OF EDGES: use position vector since it is the controlled variable. Normalize and detrend the data
        P_mean = np.mean(P_correct)
        P_amp = np.abs(P_correct-P_mean)
        P_amp -=  np.min(P_amp)
        P_amp_norm = P_amp / np.max(P_amp)
        F_mean = np.mean(F_correct)
        F_amp = np.abs(F_correct-F_mean)
        F_amp -= np.min(F_amp)
        Amp_pct = 0.8 #percentage of amplitude has been reached. Cutoff criteria for when to start looking for first the peak
        dir_start = [0, Nt -1]
        dir_cond = [1, -1]
        dir_end = [Nt -1 , 0]
        idx_Int = np.zeros(2, dtype=int)
        for i in range(2): #i values are 0 and 1
            idx = np.zeros_like(P_amp)
            cond = -1
            for ii in range(dir_start[i], dir_end[i] + dir_cond[i], dir_cond[i]):
                if P_amp_norm[ii] > Amp_pct: #greater than the criteria
                    idx[ii] = P_amp_norm[ii]
                    cond = 1
                elif cond == 1 and P_amp_norm[ii] < Amp_pct:#stop when less than criteria
                    break 
            idx_nonzero = np.nonzero(idx)[0] #Find indices where idx is not zero
            if i ==0: #start point
                idx_Int[i] = idx_nonzero[0]
            elif i == 1:  #end point
                idx_Int[i] = idx_nonzero[-1]
        #Trim data based on the indeces found
        P_correct = P_correct[idx_Int[0]:idx_Int[1]]
        F_correct = F_correct[idx_Int[0]:idx_Int[1]]
        Nt = len(P_correct)
        #TRUNCATE SIGNAL
        #fcycle = fdrive_set[n-4]
        frequency = float(all_data_Mach_1[IDName][function]["<Sinusoid>"]["Frequency, Hz:"])
        cycles = float(all_data_Mach_1[IDName][function]["<Sinusoid>"]["Number of Cycles:"])
        Ncycle = round(fs/fcycle)
        Nc = 1
#%%
#Samples = ['IVT188-16','IVT188-17', 'IVT188-18', 'IVT188-19', 'IVT188-20','IVT188-21', 'IVT188-22', 'IVT188-23', 'IVT188-24']
flag_correct_fdrive = True
metrics = {} #Where data will be stored. A dictionary is equivalent to a structure in Matlab.

for sample, IDName in enumerate(Samples): #Loop on both the index (sample) and the value (IDName) of the elements in Samples
    Amps_total = []
    averages = np.zeros(10)
    #Cycle_start = np.array([5, 6, 5, 6, 5, 5, 8, 5, 5]) 
    #Cycle_start = np.array([4, 5, 4, 5, 4, 5, 4, 6, 4]) 
    # Define Repetitions based on sample index
    #if sample == 7:
        #Repetitions = np.concatenate([np.arange(5, 11),np.arange(11, 23),np.arange(23, 29)])
    #elif sample ==6:
        #Repetitions = np.arange(Cycle_start[sample], Cycle_start[sample] + 25)
    #elif sample == 1 or 3 or 5: 
        #Repetitions = np.concatenate([np.arange(5, 23),np.arange(23, 28)])
    #else:
        #Repetitions = np.arange(Cycle_start[sample], Cycle_start[sample] + 22)    
    #Define f drive set
    fdrive_set = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.05, 0.1, 0.2, 0.5, 1, 2])
    #Initialize Moduli matrix
    Moduli = np.zeros((24,4))
    #Time = []
    #posX = []
    #Fz = []
    print("Working on DATA file " + IDName + ": ", )
    functions = list(all_data_Mach_1[IDName].keys())
    for n, function in enumerate(functions):        
        pattern = 'Sinusoid-\d+'        
        if re.fullmatch(pattern,function):
            #Nsine = Repetitions[n-4]
                     
            print("Data for: " + function )
            Time = all_data_Mach_1[IDName][function]["<DATA>"]["Time"]
            posX = all_data_Mach_1[IDName][function]["<DATA>"]["Position (z)"]
            Fz = all_data_Mach_1[IDName][function]["<DATA>"]["Fz"]
            #Time_Raw = np.append(Time_Raw, Time)
            #posX_Raw = np.append(posX_Raw, posX)
            #Fz_Raw = np.append(Fz_Raw, Fz)
            #zero_pos = np.where(Time_Raw == 0) #zero_pos contains the INDEX where the value of Time is O. The start of each test
            #Define the vectors selecting the positions of interest:                  
            #zero_pos = np.where(Time == 0)[0]
            #CORRECT SIGNAL PARAMETERS
            dt = np.mean(np.diff(Time)) #Calculate the mean difference between consecutive elements of Time
            Nt = len(Time) #length of the Time vector
            fs = 1/dt #sampling frequency
            Time_correct = np.arange(0, Nt*dt, dt)
            Time_correct = np.round(Time_correct,2) #New time vector for interpolation with two decimals
            interp_F = interp1d(Time, Fz, kind = 'cubic') #interp1d is used for cubic spline interpolation (equivalent to MATLAB's 'spline')
            interp_P = interp1d(Time, posX, kind = 'cubic')
            F_correct = interp_F(Time_correct)#perform the interpolation
            P_correct = interp_P(Time_correct)
            #TRIM OF EDGES: use position vector since it is the controlled variable. Normalize and detrend the data
            P_mean = np.mean(P_correct)
            P_amp = np.abs(P_correct-P_mean)
            P_amp -=  np.min(P_amp)
            P_amp_norm = P_amp / np.max(P_amp)
            F_mean = np.mean(F_correct)
            F_amp = np.abs(F_correct-F_mean)
            F_amp -= np.min(F_amp)
            Amp_pct = 0.8 #percentage of amplitude has been reached. Cutoff criteria for when to start looking for first the peak
            #Find the first and last max/min of data series
            #Nt = len(P_correct)
            dir_start = [0, Nt -1]
            dir_cond = [1, -1]
            dir_end = [Nt -1 , 0]
            idx_Int = np.zeros(2, dtype=int)
            for i in range(2): #i values are 0 and 1
                idx = np.zeros_like(P_amp)
                cond = -1
                for ii in range(dir_start[i], dir_end[i] + dir_cond[i], dir_cond[i]):
                    if P_amp_norm[ii] > Amp_pct: #greater than the criteria
                        idx[ii] = P_amp_norm[ii]
                        cond = 1
                    elif cond == 1 and P_amp_norm[ii] < Amp_pct:#stop when less than criteria
                        break 
                idx_nonzero = np.nonzero(idx)[0] #Find indices where idx is not zero
                if i ==0: #start point
                    idx_Int[i] = idx_nonzero[0]
                elif i == 1:  #end point
                    idx_Int[i] = idx_nonzero[-1]
            #Trim data based on the indeces found
            P_correct = P_correct[idx_Int[0]:idx_Int[1]]
            F_correct = F_correct[idx_Int[0]:idx_Int[1]]
            Nt = len(P_correct)
            #TRUNCATE SIGNAL
            #fcycle = fdrive_set[n-4]
            frequency = float(all_data_Mach_1[IDName][function]["<Sinusoid>"]["Frequency, Hz:"])
            cycles = float(all_data_Mach_1[IDName][function]["<Sinusoid>"]["Number of Cycles:"])
            Ncycle = round(fs/fcycle)
            Nc = 1
            while Nt/(Nc*Ncycle) != round(Nt/(Nc*Ncycle)):
                Nt -= 1
            #Create a new time vector
            t = np.arange(0, Nt*dt,dt)
            #Truncate Position and Force arrays
            P = P_correct[:Nt]
            F = F_correct[:Nt]
            #PSD
            nfft = 2**np.ceil(np.log2(Nt)).astype(int)
            def nextpow2(a):#Define a function to get the next power of 2
                return 2**int(np.ceil(np.log2(a)))
            nfft = nextpow2(Nt)
            f, Pxx = periodogram(detrend(P, type ='constant'), fs = fs, nfft = nfft)
            idx_fdrive = np.argmax(Pxx)
            fs = fcycle*fs/f[idx_fdrive]
            dt = 1/fs
            fcycle = f[idx_fdrive]
            Ncycle = round(fs/fcycle)
            tcycle = np.arange(0, Nc*Ncycle*dt, dt)
            t_new = np.arange(0, Nt * dt, dt)
            #Interpolate F and P
            interp_F = interp1d(t, F, kind='linear', fill_value='extrapolate')
            interp_P = interp1d(t,P, kind='linear',fill_value='extrapolate')
            F = interp_F(t_new)
            P = interp_P(t_new)
            t = t_new
            #Adjust length of singals
            Nt = len(P)
            while Nt % (Nc * Ncycle) != 0:
                Nt-=1
            t = t[:Nt]
            P = P[:Nt]
            F = F[:Nt]
            #Reshape 
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
                return np.concatenate([
                    x[0] + x[1] * np.sin(2* np.pi * x[2] * t + x[3]), #P fit
                    x[4] + x[5] * np.sin(2*np.pi * x[2] * t + x[6])  #F fit
                ])
            #Error function
            def fun_error (x,t,P,F):
                P_F_est = fun(x,t)
                return np.concatenate([P - P_F_est[:,0], F - P_F_est[:,1]])       
            #Initial guess and bounds (upper and lower)
            x0 = [np.mean(P), P_avg, fcycle, x0_P_phase, np.mean(F), F_avg, x0_F_phase]
            lb = [-np.inf, 0, 0, x0_P_phase - np.pi/2, -np.inf, 0, x0_F_phase - np.pi/2]
            ub = [np.inf, np.inf, np.inf, x0_P_phase + np.pi/2, np.inf, np.inf, x0_P_phase + np.pi/2]           
            # Perform the fit using least_squares
            res = least_squares(fun_error, x0, bounds=(lb, ub), args=(t, P, F),method='trf', verbose=2)
            # Extract fit parameters. An array of parameter values that minimize the error function. 
            fit_params = res.x
            #Fit parameters
            Off_P = fit_params[0]
            Amp_P = fit_params[1]
            fcycle = fit_params[2]
            phase_P = fit_params[3]
            Off_F = fit_params[4]
            Amp_F = fit_params[5]
            phase_F = fit_params[6]
            #PLOT
            if 10 <= n-4 <= 19:
                pass
            else:
                # Create a new figure
                #fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # equivalent to tiled layout with two horizontal plots
                # First subplot: Position vs. Time
                #axes[0].plot(t, P, '-b', linewidth=2, label='Position Data')
                #axes[0].plot(t, Off_P + Amp_P * np.sin(2 * np.pi * fcycle * t + phase_P), '--k', linewidth=2, label='Fit')
                #axes[0].set_xlabel('time, [s]')
                #axes[0].set_ylabel('Position, [mm]')
                #axes[0].legend()    
                # Second subplot: Force vs. Time
                #axes[1].plot(t, F, '-r', linewidth=2, label='Force Data')
                #axes[1].plot(t, Off_F + Amp_F * np.sin(2 * np.pi * fcycle * t + phase_F), '--k', linewidth=2, label='Fit')
                #axes[1].set_xlabel('time, [s]')
                #axes[1].set_ylabel('Force, [gf]')
                #axes[1].legend()    
                # Adjust layout to avoid overlap
                #plt.tight_layout()
                # Save the figure
                #filename = f'IVT188_TP0_Dynamic_{n}_{Samples[sample]}.png'
                #plt.savefig(filename, format='png')
                # Show the figure
                #plt.show()
                # Calculate phase offset in degrees
                phase_offset = np.degrees(abs(phase_P - phase_F))
                # Calculate G' (storage modulus), G'' (loss modulus), and |G| (complex modulus)
                G_prime = Amp_F / Amp_P * np.cos(np.radians(phase_offset))  # G'
                G_dprime = Amp_F / Amp_P * np.sin(np.radians(phase_offset))  # G''
                G_mag = np.sqrt(G_prime**2 + G_dprime**2)  # |G|
                # Store the results in the Moduli array
                Moduli[n-4, 0] = phase_offset  # Phase offset
                Moduli[n-4, 1] = G_prime       # G'
                Moduli[n-4, 2] = G_dprime      # G''
                Moduli[n-4, 3] = G_mag         # |G|
                # Close any open plots
                #plt.close('all')
            # Check for specific range of n
            if 10 <= n-4 <= 19:
                # Reshape and detrend F array
                amps = np.reshape(detrend(F), (Ncycle, -1))
                #Calculate average amplitude
                avg_amp = np.mean((np.max(amps, axis=0) - np.min(amps, axis=0)) / 2)
                # Store the average amplitude for current n
                averages[(n-4) - 10] = avg_amp
                #Append maximum amplitudes to Amps_total
                Amps_total=np.concatenate((Amps_total, np.max(amps,axis=0)))
    #fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    #fig.subplots_adjust(wspace=0.3)
    # First plot for Amps_total
    #axs[0].plot(Amps_total, 'b')
    #axs[0].set_xlabel('Cycles')
    #axs[0].set_ylabel('Amplitude, [gf]')
    #axs[0].set_title('Amplitude over Cycles')
    # Second plot for averages
    #axs[1].plot(averages, 'bo-')
    #axs[1].set_xlabel('Repetition')
    #axs[1].set_title('Average Amplitude per Repetition')
    # Save the figure as a PNG file
    #plt.savefig(f'IVT188_TP0_Dynamic_Amplitude_{Samples[sample]}.png', format='png')
    # Move the window to the center of the screen (not applicable in Matplotlib like in MATLAB)
    # This step is generally handled by the OS or UI, so we can skip it.
    # Store results in a dictionary
    metrics['Sample'] = {IDName}
    metrics['phaseoffset'] = Moduli[:, 0]
    metrics['Gprime'] = Moduli[:, 1]
    metrics['Gdprime'] = Moduli[:, 2]
    metrics['Gmag'] = Moduli[:, 3]
    metrics['AvgAmp'] = averages
    # Close the plot
    #plt.close()

# Save the 'metrics' dictionary to a .mat file
sio.savemat('IVT188_TP0_Dynamic.mat', {'metrics': metrics})

# Convert the 'metrics' dictionary into a Pandas DataFrame
metrics_df = pd.DataFrame.from_dict(metrics)

# Save the DataFrame to a CSV file with quotes around all strings
metrics_df.to_csv('IVT188_TP0_Dynamic.csv', sep=',', quotechar='"', quoting=1, index=False)





        

















   




            