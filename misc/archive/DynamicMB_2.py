os.system('cls' if os.name == 'nt' else 'clear')  # Clear console

import os
import re
import openpyxl
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.signal import periodogram, detrend
from read_mach_1_file_PYTH import read_mach_1_file, read_mach_1_files, select_mach_1_file

#Setting plot style in 
plt.rc('text', usetex = True)
#plt.rc('front',family = 'serif')
plt.rc('axes',labelsize=16, linewidth = 0.5)
plt.rc('legend', fontsize = 12)
plt.rc('lines', linewidth = 2, markersize = 4)

#Load MACH1 files 
data_headers_selection = None # Select what data of mach-1 files to load - None if all data has to be loaded
sampleName = None # Enter the name of sample or None to load all files in selected directory
all_data_Mach_1, mach_1_dir = read_mach_1_files(1, data_headers_selection, sampleName, True) #Select folder with the data. "all_data_Mach1" is an array with samples' IDs. 
Samples = list(all_data_Mach_1.keys()) #Array with Sample's IDs. 

flag_correct_fdrive = True
PeakToPeak = 2 # Percent strain peak-to-peak amplitude for dynamic testing
metrics = {} #Where data will be stored. A dictionary is equivalent to a structure in Matlab.
Moduli = np.array([]) #An empty list to store the values of each Sample (IVT16, IVT17... IVT24) 
phase_offset_row = []
G_prime_row = []
G_dprime_row = []
G_mag_row = []
averages_row = []

for sample, IDName in enumerate(Samples): #Loop on both the index (sample) and the value (IDName) of the elements in Samples
    Amps_total = []    
    print("Working on DATA file " + IDName + ": ", )
    functions = list(all_data_Mach_1[IDName].keys())
    for n, function in enumerate(functions): 
    
        if sample == 1 and n == 4:
            continue
        if sample == 3 and n == 4:
            continue
        if sample == 5 and n == 22:
            continue
        if sample == 7 and n == 14:
            continue
        if sample == 7 and n == 15:
            continue       
        
        pattern = 'Sinusoid-\d+'               
        if re.fullmatch(pattern,function):                                 
            print("Data for: " + function)
            print(f"N-index: {n}")
            print(f"Sample-index {sample}")

            Time = all_data_Mach_1[IDName][function]["<DATA>"]["Time"]
            posX = all_data_Mach_1[IDName][function]["<DATA>"]["Position (z)"]
            Fz = all_data_Mach_1[IDName][function]["<DATA>"]["Fz"]
            
            #CORRECT SIGNAL PARAMETERS
            dt = np.mean(np.diff(Time)) #Calculate the mean difference between consecutive elements of Time
            Nt = len(Time) #length of the Time vector
            fs = 1/dt #sampling frequency
            frequency = float(all_data_Mach_1[IDName][function]["<Sinusoid>"]["Frequency, Hz:"])
            
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
            Ncycle = round(fs/frequency)
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
            fs = frequency*fs/f[idx_fdrive]
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
                return np.column_stack((
                x[0] + x[1] * np.sin(2* np.pi * x[2] * t + x[3]), #P fit
                x[4] + x[5] * np.sin(2*np.pi * x[2] * t + x[6])  #F fit
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
            #Fit parameters
            Off_P = fit_params[0]
            Amp_P = fit_params[1]
            fcycle = fit_params[2]
            phase_P = fit_params[3]
            Off_F = fit_params[4]
            Amp_F = fit_params[5]
            phase_F = fit_params[6]
            #PLOT
            # Create a new figure
            if sample == 1:
                if 10 < n < 21:                  
                    pass
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # equivalent to tiled layout with two horizontal plots
                    plt.rcParams['text.usetex'] = False
                    # First subplot: Position vs. Time
                    axes[0].plot(t, P, '-b', linewidth=2, label='Position Data')
                    axes[0].plot(t, Off_P + Amp_P * np.sin(2 * np.pi * fcycle * t + phase_P), '--k', linewidth=2, label='Fit')
                    axes[0].set_xlabel('time, [s]')
                    axes[0].set_ylabel('Position, [mm]')
                    axes[0].legend()    
                    # Second subplot: Force vs. Time
                    plt.rcParams['text.usetex'] = False
                    axes[1].plot(t, F, '-r', linewidth=2, label='Force Data')
                    axes[1].plot(t, Off_F + Amp_F * np.sin(2 * np.pi * fcycle * t + phase_F), '--k', linewidth=2, label='Fit')
                    axes[1].set_xlabel('time, [s]')
                    axes[1].set_ylabel('Force, [gf]')
                    axes[1].legend()    
                    # Adjust layout to avoid overlap
                    plt.tight_layout()
                    # Save the figure
                    #filename = f'IVT188_TP0_Dynamic_{n}_{Samples[sample]}.png'
                    #plt.savefig(filename, format='png')
                    # Show the figure
                    plt.show()
                    # Calculate phase offset in degrees
                    phase_offset = np.degrees(abs(phase_P - phase_F))
                    # Calculate G' (storage modulus), G'' (loss modulus), and |G| (complex modulus)
                    G_prime = Amp_F / Amp_P * np.cos(np.radians(phase_offset))  # G'
                    G_dprime = Amp_F / Amp_P * np.sin(np.radians(phase_offset))  # G''
                    G_mag = np.sqrt(G_prime**2 + G_dprime**2)  # |G|
                    # Store the results in the Moduli array
                    phase_offset_row.append(phase_offset)
                    G_prime_row.append(G_prime)
                    G_dprime_row.append(G_dprime)
                    G_mag_row.append(G_mag)
                    # Close any open plots
                    plt.close('all') 
            elif sample == 3:
                if 10 < n < 21:
                    pass
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # equivalent to tiled layout with two horizontal plots
                    plt.rcParams['text.usetex'] = False
                    # First subplot: Position vs. Time
                    axes[0].plot(t, P, '-b', linewidth=2, label='Position Data')
                    axes[0].plot(t, Off_P + Amp_P * np.sin(2 * np.pi * fcycle * t + phase_P), '--k', linewidth=2, label='Fit')
                    axes[0].set_xlabel('time, [s]')
                    axes[0].set_ylabel('Position, [mm]')
                    axes[0].legend()    
                    # Second subplot: Force vs. Time
                    plt.rcParams['text.usetex'] = False
                    axes[1].plot(t, F, '-r', linewidth=2, label='Force Data')
                    axes[1].plot(t, Off_F + Amp_F * np.sin(2 * np.pi * fcycle * t + phase_F), '--k', linewidth=2, label='Fit')
                    axes[1].set_xlabel('time, [s]')
                    axes[1].set_ylabel('Force, [gf]')
                    axes[1].legend()    
                    # Adjust layout to avoid overlap
                    plt.tight_layout()
                    # Save the figure
                    #filename = f'IVT188_TP0_Dynamic_{n}_{Samples[sample]}.png'
                    #plt.savefig(filename, format='png')
                    # Show the figure
                    plt.show()
                    # Calculate phase offset in degrees
                    phase_offset = np.degrees(abs(phase_P - phase_F))
                    # Calculate G' (storage modulus), G'' (loss modulus), and |G| (complex modulus)
                    G_prime = Amp_F / Amp_P * np.cos(np.radians(phase_offset))  # G'
                    G_dprime = Amp_F / Amp_P * np.sin(np.radians(phase_offset))  # G''
                    G_mag = np.sqrt(G_prime**2 + G_dprime**2)  # |G|
                    # Store the results in the Moduli array
                    phase_offset_row.append(phase_offset)
                    G_prime_row.append(G_prime)
                    G_dprime_row.append(G_dprime)
                    G_mag_row.append(G_mag)
                    # Close any open plots
                    plt.close('all')
            elif sample == 6:
                if 12 < n < 23:
                    pass
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # equivalent to tiled layout with two horizontal plots
                    plt.rcParams['text.usetex'] = False
                    # First subplot: Position vs. Time
                    axes[0].plot(t, P, '-b', linewidth=2, label='Position Data')
                    axes[0].plot(t, Off_P + Amp_P * np.sin(2 * np.pi * fcycle * t + phase_P), '--k', linewidth=2, label='Fit')
                    axes[0].set_xlabel('time, [s]')
                    axes[0].set_ylabel('Position, [mm]')
                    axes[0].legend()    
                    # Second subplot: Force vs. Time
                    plt.rcParams['text.usetex'] = False
                    axes[1].plot(t, F, '-r', linewidth=2, label='Force Data')
                    axes[1].plot(t, Off_F + Amp_F * np.sin(2 * np.pi * fcycle * t + phase_F), '--k', linewidth=2, label='Fit')
                    axes[1].set_xlabel('time, [s]')
                    axes[1].set_ylabel('Force, [gf]')
                    axes[1].legend()    
                    # Adjust layout to avoid overlap
                    plt.tight_layout()
                    # Save the figure
                    #filename = f'IVT188_TP0_Dynamic_{n}_{Samples[sample]}.png'
                    #plt.savefig(filename, format='png')
                    # Show the figure
                    plt.show()
                    # Calculate phase offset in degrees
                    phase_offset = np.degrees(abs(phase_P - phase_F))
                    # Calculate G' (storage modulus), G'' (loss modulus), and |G| (complex modulus)
                    G_prime = Amp_F / Amp_P * np.cos(np.radians(phase_offset))  # G'
                    G_dprime = Amp_F / Amp_P * np.sin(np.radians(phase_offset))  # G''
                    G_mag = np.sqrt(G_prime**2 + G_dprime**2)  # |G|
                    # Store the results in the Moduli array
                    phase_offset_row.append(phase_offset)
                    G_prime_row.append(G_prime)
                    G_dprime_row.append(G_dprime)
                    G_mag_row.append(G_mag)
                    # Close any open plots
                    plt.close('all')
            elif sample == 7:
                if 9 < n < 22:
                    pass
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # equivalent to tiled layout with two horizontal plots
                    plt.rcParams['text.usetex'] = False
                    # First subplot: Position vs. Time
                    axes[0].plot(t, P, '-b', linewidth=2, label='Position Data')
                    axes[0].plot(t, Off_P + Amp_P * np.sin(2 * np.pi * fcycle * t + phase_P), '--k', linewidth=2, label='Fit')
                    axes[0].set_xlabel('time, [s]')
                    axes[0].set_ylabel('Position, [mm]')
                    axes[0].legend()    
                    # Second subplot: Force vs. Time
                    plt.rcParams['text.usetex'] = False
                    axes[1].plot(t, F, '-r', linewidth=2, label='Force Data')
                    axes[1].plot(t, Off_F + Amp_F * np.sin(2 * np.pi * fcycle * t + phase_F), '--k', linewidth=2, label='Fit')
                    axes[1].set_xlabel('time, [s]')
                    axes[1].set_ylabel('Force, [gf]')
                    axes[1].legend()    
                    # Adjust layout to avoid overlap
                    plt.tight_layout()
                    # Save the figure
                    #filename = f'IVT188_TP0_Dynamic_{n}_{Samples[sample]}.png'
                    #plt.savefig(filename, format='png')
                    # Show the figure
                    plt.show()
                    # Calculate phase offset in degrees
                    phase_offset = np.degrees(abs(phase_P - phase_F))
                    # Calculate G' (storage modulus), G'' (loss modulus), and |G| (complex modulus)
                    G_prime = Amp_F / Amp_P * np.cos(np.radians(phase_offset))  # G'
                    G_dprime = Amp_F / Amp_P * np.sin(np.radians(phase_offset))  # G''
                    G_mag = np.sqrt(G_prime**2 + G_dprime**2)  # |G|
                    # Store the results in the Moduli array
                    phase_offset_row.append(phase_offset)
                    G_prime_row.append(G_prime)
                    G_dprime_row.append(G_dprime)
                    G_mag_row.append(G_mag)
                    # Close any open plots
                    plt.close('all')
            else:
                if 9 < n < 20:
                    pass
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # equivalent to tiled layout with two horizontal plots
                    plt.rcParams['text.usetex'] = False
                    # First subplot: Position vs. Time
                    axes[0].plot(t, P, '-b', linewidth=2, label='Position Data')
                    axes[0].plot(t, Off_P + Amp_P * np.sin(2 * np.pi * fcycle * t + phase_P), '--k', linewidth=2, label='Fit')
                    axes[0].set_xlabel('time, [s]')
                    axes[0].set_ylabel('Position, [mm]')
                    axes[0].legend()    
                    # Second subplot: Force vs. Time
                    plt.rcParams['text.usetex'] = False
                    axes[1].plot(t, F, '-r', linewidth=2, label='Force Data')
                    axes[1].plot(t, Off_F + Amp_F * np.sin(2 * np.pi * fcycle * t + phase_F), '--k', linewidth=2, label='Fit')
                    axes[1].set_xlabel('time, [s]')
                    axes[1].set_ylabel('Force, [gf]')
                    axes[1].legend()    
                    # Adjust layout to avoid overlap
                    plt.tight_layout()
                    # Save the figure
                    #filename = f'IVT188_TP0_Dynamic_{n}_{Samples[sample]}.png'
                    #plt.savefig(filename, format='png')
                    # Show the figure
                    plt.show()
                    # Calculate phase offset in degrees
                    phase_offset = np.degrees(abs(phase_P - phase_F))
                    # Calculate G' (storage modulus), G'' (loss modulus), and |G| (complex modulus)
                    G_prime = Amp_F / Amp_P * np.cos(np.radians(phase_offset))  # G'
                    G_dprime = Amp_F / Amp_P * np.sin(np.radians(phase_offset))  # G''
                    G_mag = np.sqrt(G_prime**2 + G_dprime**2)  # |G|
                    # Store the results in the Moduli array
                    phase_offset_row.append(phase_offset)
                    G_prime_row.append(G_prime)
                    G_dprime_row.append(G_dprime)
                    G_mag_row.append(G_mag)
                    # Close any open plots
                    plt.close('all')

            # Check for specific range of n
            if sample == 1:
                if 10 < n < 21:
                    # Reshape and detrend F array
                    amps = np.reshape(detrend(F), (Ncycle, -1))
                    #Calculate average amplitude
                    avg_amp = np.mean((np.max(amps, axis=0) - np.min(amps, axis=0)) / 2)
                    # Store the average amplitude for current n
                    averages_row.append([avg_amp])
                    #Append maximum amplitudes to Amps_total
                    Amps_total=np.concatenate((Amps_total, np.max(amps,axis=0)))
            elif sample == 3:
                if 10 < n < 21:
                   # Reshape and detrend F array
                    amps = np.reshape(detrend(F), (Ncycle, -1))
                    #Calculate average amplitude
                    avg_amp = np.mean((np.max(amps, axis=0) - np.min(amps, axis=0)) / 2)
                    # Store the average amplitude for current n                    
                    averages_row.append([avg_amp])
                    #Append maximum amplitudes to Amps_total
                    Amps_total=np.concatenate((Amps_total, np.max(amps,axis=0))) 
            elif sample == 6:
                if 12 < n < 23:
                    # Reshape and detrend F array
                    amps = np.reshape(detrend(F), (Ncycle, -1))
                    #Calculate average amplitude
                    avg_amp = np.mean((np.max(amps, axis=0) - np.min(amps, axis=0)) / 2)
                    # Store the average amplitude for current n                    
                    averages_row.append([avg_amp])
                    #Append maximum amplitudes to Amps_total
                    Amps_total=np.concatenate((Amps_total, np.max(amps,axis=0)))
            elif sample == 7:
                if 9 < n < 22: 
                    # Reshape and detrend F array
                    amps = np.reshape(detrend(F), (Ncycle, -1))
                    #Calculate average amplitude
                    avg_amp = np.mean((np.max(amps, axis=0) - np.min(amps, axis=0)) / 2)
                    # Store the average amplitude for current n                    
                    averages_row.append([avg_amp])
                    #Append maximum amplitudes to Amps_total
                    Amps_total=np.concatenate((Amps_total, np.max(amps,axis=0)))
            else:
                if 9 < n < 20:
                    # Reshape and detrend F array
                    amps = np.reshape(detrend(F), (Ncycle, -1))
                    #Calculate average amplitude
                    avg_amp = np.mean((np.max(amps, axis=0) - np.min(amps, axis=0)) / 2)
                    # Store the average amplitude for current n                    
                    averages_row.append([avg_amp])
                    #Append maximum amplitudes to Amps_total
                    Amps_total=np.concatenate((Amps_total, np.max(amps,axis=0)))

    plt.rcParams['text.usetex'] = False
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(wspace=0.3)

    # First plot for Amps_total
    y_limit_1 = [0, 14]
    axs[0].plot(Amps_total, 'b')
    axs[0].set_xlabel('Cycles')
    axs[0].set_ylabel('Amplitude, [gf]')
    axs[0].set_title('Amplitude over Cycles')
    axs[0].set_ylim(y_limit_1)
    
    # Second plot for averages
    y_limit_2 = [0, 14]
    axs[1].plot(averages_row, 'bo-')
    axs[1].set_xlabel('Repetition')
    axs[1].set_title('Average Amplitude per Repetition')
    axs[1].set_ylim(y_limit_2)
    # Save the figure as a PNG file
    plt.savefig(f'IVT188_TP0_Dynamic_Amplitude_{Samples[sample]}.png', format='png')
    plt.show()
    plt.close()

    # Store results in a dictionary
    Moduli = np.array([phase_offset_row, G_prime_row, G_dprime_row, G_mag_row])
    #metrics['Sample'] = {IDName}
    metrics['phaseoffset'] = Moduli[0, :] #First row
    metrics['Gprime'] = Moduli[1,:]
    metrics['Gdprime'] = Moduli[2,:]
    metrics['Gmag'] = Moduli[3,:]
    metrics['AvgAmp'] = np.array(averages_row).flatten()
    # Close the plot
    

# Save the 'metrics' dictionary to a .mat file
sio.savemat('IVT188_TP0_Dynamic.mat', {'metrics': metrics})

# Convert the 'metrics' dictionary into a Pandas DataFrame
#metrics_df = pd.DataFrame.from_dict(metrics)
metrics_df = pd.DataFrame(dict([(k, pd.Series(v))for k, v in metrics.items()]))

# Save the DataFrame to a CSV file with quotes around all strings
metrics_df.to_csv('IVT188_TP0_Dynamic.csv', sep=',', quotechar='"', quoting=1, index=False)





        

















   




            