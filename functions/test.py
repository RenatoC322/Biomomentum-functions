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
import numpy as np
import matplotlib.pyplot as plt 
from HayesElasticModel_PYTH import *
from Sinus_Analysis_PYTH import *
from read_mach_1_file_PYTH import read_mach_1_file, read_mach_1_files, select_mach_1_file
from Stress_Relaxation_Fit_PYTH import *
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
#%%
# Loading one mach-1 file at a time
mach_1_file = select_mach_1_file()
mach_1_data = read_mach_1_file(mach_1_file)
#%%
# Loading multiple mach-1 files in a directory
data_headers_selection = ["Time", "Position (z)", "Fz"] # Select specific data of mach-1 files
keyword = "" # input sample name to load mach-1 txt corresponding to the sample or group
mach_1_files_data, mach_1_dir = read_mach_1_files(data_headers_selection = data_headers_selection, keyword = keyword, showProgressBar=False)
#%%
# Get specific data values from mach_1_data: mach_1_data[function name]["<DATA>"], 
# for Stress-Relaxation (example first ramp) -> mach_1_data[function name]["<DATA>"]["Ramp-1]
time = mach_1_data["Sinusoid-1"]["<DATA>"]["Time"]
position_Z = mach_1_data["Sinusoid-1"]["<DATA>"]["Position (z)"]
Fz = mach_1_data["Sinusoid-1"]["<DATA>"]["Fz"]
#%%
# plot the signal from the loaded data values
fig = plt.figure(figsize=(25,15))
axes = fig.add_subplot(111)
axes.plot(time, Fz, "-b")
axes.set_ylabel("Normal Force (N)",size=20)
axes.set_xlabel("Time (s)",size=20)
axes.tick_params(axis='both', which='major', labelsize=20)
plt.show()
#%%
# Testing Hayes Model
data_headers_selection = ["Fz", "Time", "Position (z)"] # Select what data of mach-1 files to load - None if all data has to be loaded
sampleName = "Tarsus4" # Enter the name of sample or None to load all files in selected directory
mach_1_files_data, mach_1_dir = read_mach_1_files(data_headers_selection = data_headers_selection, keyword = sampleName, showProgressBar=False)
#%%
filename = list(mach_1_files_data.keys())[0]
time = mach_1_files_data[filename]["Stress Relaxation-1"]["<DATA>"]["Ramp-1"]["Time"]
position_Z = mach_1_files_data[filename]["Stress Relaxation-1"]["<DATA>"]["Ramp-1"]["Position (z)"]
Fz = mach_1_files_data[filename]["Stress Relaxation-1"]["<DATA>"]["Ramp-1"]["Fz"]
units_gf = "Single" in mach_1_files_data[filename]["Stress Relaxation-1"]["<INFO>"]["Load Cell Type:"]

# Hayes Model Inputs
indenter_R = 0.15
poisson_coefficient = 0.5
Rsq_req = -1000000000
sample_thickness = 100000
strain = abs((position_Z - position_Z[0])/position_Z[0])
maxStrain = np.max(strain)
G_inst, E_inst, Fit_inst, Rsq_adj_inst = HayesElasticModel(position_Z, Fz, units_gf, maxStrain, indenter_R, poisson_coefficient, Rsq_req, sample_thickness, True, False)

# Plotting Hayes Model Fit unto data
fig = plt.figure(figsize=(20,15))
axes = fig.add_subplot(111)
axes.plot(isNegative(position_Z), Fz, "-b", label = "Fz")
axes.plot(Fit_inst[:, 0], Fit_inst[:, 1], "--r", label = "Hayes Model")
axes.set_ylabel("Normal Force (gf)",size=20)
axes.set_xlabel("Position Z (mm)",size=20)
axes.tick_params(axis = 'both', which = 'major', labelsize=15)
axes.set_title(filename + " - Elastic Model in Indentation - Stress-Relaxation-1", size = 20)
plt.grid()
plt.legend(loc = 'lower right', fontsize = 20)
plt.show()
print("Stress-Relaxation-1")
print("Indentation Elastic Modulus (MPa): ", E_inst)
print(' R{}: '.format(get_super('2')), Rsq_adj_inst)
#%%
# Testing Dynamic - Sinusoid Analysis
mach_1_data = read_mach_1_file(select_mach_1_file())
#%%
PeakToPeak = 1
function = "Sinusoid-1"
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
#%%
# Testing Stress-Relaxation Analysis
mach_1_data = read_mach_1_file(select_mach_1_file())
#%%
plt.switch_backend('tkagg')
function = "Stress Relaxation-2"
D = 4.8 # Diameter of the cartilage upon extraction, mm
Strain = 2.5 # 2.5 percent strain / ramp
initialStrain = 10 # 10 percent strain precompression
vm = 0.36 # Poisson's ration (between 0 - 0.5) from litterature 0.36 (Garon 2007)
units_gf = "Single" in mach_1_data[function]["<INFO>"]["Load Cell Type:"]
fig, ax = plt.subplots()
for nRamp, ramp in enumerate(mach_1_data[function]["<DATA>"].values()):
    Fz = -ramp["Fz"]
    if units_gf:
        Fz = -Fz*0.00980665
    Time = ramp["Time"]
    sigZ = Fz/(np.pi * (D/2)**2)
    tau = relaxation_constant(Fz, Time, np.argmax(Fz))
    szfit, efibril, k0, e33_eq, t0, S11, szequ, K, tau_fit, em, nm, mse, veff_poisson = stressrelaxation_fit(Time, sigZ, D/2, Strain/100, vm)
    ax.plot(Time, sigZ, "-b")
    if nRamp == 0:
        ax.plot(Time, szfit, "--r", label = "Fit")
    else:
        ax.plot(Time, szfit, "--r")
    ax.relim()
    ax.set_ylabel("Stress (MPa)",size=20)
    ax.set_xlabel("Time (s)",size=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.5)
    print("===============================================")
    print(f"RAMP-{nRamp + 1}")
    print("Fibril Network Modulus (MPa): ", np.round(efibril, 4))
    print("Hydraulic Permeability " + '(µm{})'.format(get_super('2')) + "/[MPa*s]: ", np.round(k0,4))
    print("Poisson Ratio at Equilibrium: ", np.round(veff_poisson,4))
    print("Root Mean Squared Error of Fit (MPa): ", np.round(np.sqrt(mse), 4))
    print("Relaxation Constant (s): ", np.round(tau, 4))
    print("=============================================== \n")
plt.grid()
plt.legend(loc = 'lower right', fontsize = 20)
plt.show()  














