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
from read_mach_1_file_PYTH import read_mach_1_file, read_mach_1_files, select_mach_1_file
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
#%%
# Hayes Model Inputs
indenter_R = 0.15
poisson_coefficient = 0.5
Rsq_req = -1000000000
sample_thickness = 100000
strain = abs((position_Z - position_Z[0])/position_Z[0])
maxStrain = np.max(strain)
G_inst, E_inst, Fit_inst, Rsq_adj_inst = HayesElasticModel(position_Z, Fz, units_gf, maxStrain, indenter_R, poisson_coefficient, Rsq_req, sample_thickness, True, False)
#%%
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
#fig.legend(loc = 7, prop = { "size": 20 })
#fig.subplots_adjust(right=0.8)
plt.show()
print("Stress-Relaxation-1")
print("Indentation Elastic Modulus (MPa): ", E_inst)
print(' R{}: '.format(get_super('2')), Rsq_adj_inst)