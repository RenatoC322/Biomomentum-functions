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
from read_mach_1_file_PYTH import read_mach_1_file, read_mach_1_files, select_mach_1_file
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