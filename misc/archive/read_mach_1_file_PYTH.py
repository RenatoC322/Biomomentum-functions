#######################################################################################
# Contact:  -   castillo.renato.90@gmail.com
#           -   castillo@biomomentum.com
######################################################################################

import pandas as pd
import os
import tkinter
import re
import numpy as np
import concurrent.futures
from tkinter import filedialog, messagebox
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: sorted_alphanumeric
% Description: Function that sorts alphanumeric files
% Inputs:   data    - List of Mach-1 files
% Output:   sorted  - List of Mach-1 files 
% 
%   By: Renato Castillo, 1Oct2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def sorted_alphanumeric(data:list) -> list:
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: select_mach_1_file
% Description: Function that brings a pop up prompt to select the mach-1 file
% Inputs:   None
% Output:   filename    - String address of the mach-1 file 
% 
%   By: Renato Castillo, 29AUG2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def select_mach_1_file() -> str:
    pop_window = tkinter.Tk()
    pop_window.withdraw()
    pop_window.attributes('-topmost', True) 
    filename = filedialog.askopenfilename(parent = pop_window, initialdir= "/", title='Please select the Mach-1 .txt file')
    if len(filename) == 0 or not filename.endswith(".txt"):
        messagebox.showwarning("Warning", "No mach-1 file selected!")
        filename = None
    return filename
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: select_mach_1_files_dir
% Description: Function that brings a pop up prompt to select a folder contaning 
%              multiple mach-1 files.
% Inputs:   keyword     -  String Name of group of mach-1 files to load in folder
% Output:   mach_1_sets    -  A 2 by N matrix where the first column corresponds to
%                          to the directories of the files and the second column
%                          corresponds to the filenames.  
% 
%   By: Renato Castillo, 29AUG2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def select_mach_1_files_dir(keyword:str = ""):
    mach_1_dirs = []
    mach_1_fnames = []
    pop_window = tkinter.Tk()
    pop_window.withdraw()
    pop_window.attributes('-topmost', True) 
    main_dir = filedialog.askdirectory(parent = pop_window, initialdir= "/", title = 'Please select the Mach-1 files directory')
    try:
        print("Mach-1 files located at : " + main_dir)
        mach_1_files = sorted(filter(lambda x: os.path.isfile(os.path.join(main_dir, x)), os.listdir(main_dir)))
        mach_1_dirs = sorted_alphanumeric([os.path.join(main_dir, file) for file in mach_1_files if file.endswith(".txt") and keyword in file])
        mach_1_fnames = [file.split("\\")[-1].split(".")[0] for file in mach_1_dirs]

    except:
        print("No directory selected or directory does not exist...")
    mach_1_sets =  np.array([[mach_1_dirs[i], mach_1_fnames[i]] for i in range(len(mach_1_fnames))])
    return mach_1_sets
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: printProgressBar
% Description: Calls in a loop to create a terminal progress bar
% Inputs:   iteration   - Required  : current iteration (Int)
%           total       - Required  : total iterations (Int)
%           prefix      - Optional  : prefix string (Str)
%           suffix      - Optional  : suffix string (Str)
%           decimals    - Optional  : positive number of decimals in percent complete (Int)
%           length      - Optional  : character length of bar (Int)
%           fill        - Optional  : bar fill character (Str)
%           printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
% Output:   None
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    if total == 0:
        total = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: prepareData
% Description: Function that converts data array to dictionary.
% Inputs:   dataframe   - Panda dataframe
%           dataHeaders - Data headers of the dataframe
% Output:   df_Data     - Dictionary with the headers of the dataframe
% 
%   By: Renato Castillo, 8FEB2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def prepareData(dataframe, dataHeaders):
    df_Data = {}
    for idx, dataHeader in enumerate(dataHeaders):
        df_Data[dataHeader.split(",")[0]] = dataframe.values[:, idx]
    return df_Data
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: selectData
% Description: Function that selects specific data headers of Mach-1 file.
% Inputs:   filename               - filename of Mach-1 file
%           idxs_dividers          - Index of Mach-1 file dividers
%           idx_func               - Index of function row in Mach-1 file
%           data_headers_selection - Data headers selected
% Output:   data_indexes           - Data indexes of the selected data headers
% 
%   By: Renato Castillo, 8FEB2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def selectData(filename, idxs_dividers, idx_func, data_headers_selection):
    data_indexes = None
    if data_headers_selection is not None:
        all_data_headers = pd.read_csv(filename, sep="\t", skiprows = idxs_dividers[4*idx_func+2], nrows=0, engine="c", na_filter=False, low_memory=False).columns.values
        all_data_headers = [dataHeader.split(",")[0] for dataHeader in all_data_headers]
        data_indexes = [all_data_headers.index(selection) for selection in data_headers_selection]
    return data_indexes
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: read_mach_1_file
% Description: Loads all data of mach-1 file into a structure separated by the file 
%              dividers.
% Inputs:   filename        -   String containing the .txt file name generated by Mach-1 Motion
%           read_data       -   Integer indicating if the numerical data in the Mach-1 .txt
%                               file should be read. Default value is 1. 0: Read only the info.
%                               N-null: Read info and numerical data
%           showProgressBar -   Bool flag to show progress bar on command prompt
% Output:   dfMACH_1        -   Structure containing everything read from the Mach-1 Motion .txt
%                               file. Returns 0 if an error happened. See documentation for
%                               more info about the structure.
% 
%   By: Renato Castillo, 25AUG2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def read_mach_1_file(filename, read_data = 1, data_headers_selection = None, showProgressBar = True):
    dfMACH_1 = {} # dictionary to store all mach-1 data
    # Dividers used to separate steps in .txt file
    dividers = ["<INFO>","<END INFO>","<DATA>","<END DATA>"]
    # Verify the file is valid and Mach-1 file
    if not os.path.exists(filename) and not filename.endswith(".txt"):
        raise ValueError("Incorrect file extension or file does not exist.")
    idxs_dividers = search_dividers_idx(filename, dividers) # row position of dividers in .txt file
    flag_data_separators, idxs_data_separators = check_data_separator(filename, idxs_dividers)
    data_indexes = selectData(filename, idxs_dividers, 0, data_headers_selection)
    # Start to load data from Mach-1 file into dictionary
    dicFuncNames = {}
    for function in range(len(idxs_dividers)//4):
        dfINFO = pd.read_csv(filename, sep="\t", skiprows = idxs_dividers[4*function]-1, nrows=idxs_dividers[4*function+1]-idxs_dividers[4*function]-2, engine="c", na_filter=False, low_memory=False).to_dict() # extract mach-1 experiment info
        dfFUNCTION = pd.read_csv(filename, sep="\t", skiprows = idxs_dividers[4*function+1], nrows=idxs_dividers[4*function+2]-idxs_dividers[4*function+1]-2, engine="c", na_filter=False, low_memory=False).to_dict() # extract function info
        funcName = list(dfFUNCTION.keys())[0].split("<")[-1].split(">")[0] 
        if funcName in dicFuncNames:
            dicFuncNames[funcName] += 1
        else:
            dicFuncNames[funcName] = 1
        currentFunc = funcName + "-{}".format(dicFuncNames[funcName])
        if read_data == 1:
            if not flag_data_separators[function]:
                df = pd.read_csv(filename, sep="\t", skiprows = idxs_dividers[4*function+2], nrows=idxs_dividers[4*function+3]-idxs_dividers[4*function+2]-2, usecols=data_indexes, engine="c", na_filter=False, low_memory=False)
                dataHeaders = df.columns.values
                dfDATA = {"<DATA>" : prepareData(df, dataHeaders)}
            elif flag_data_separators[function]:
                visited_dividers = [idxs_data_separators[-1]]
                nSeparators = 0
                dfData = {}
                while idxs_data_separators[0] < idxs_dividers[4*function + 3]:
                    if len(idxs_data_separators) == 1:
                        df = pd.read_csv(filename, sep="\t", skiprows = idxs_dividers[4*function+2], nrows=idxs_data_separators[0]-idxs_dividers[4*function+2]-2, usecols=data_indexes, engine="c", na_filter=False, low_memory=False)
                        dataHeaders = df.columns.values
                        dfData['Ramp-{}'.format(nSeparators + 1)] = prepareData(df, dataHeaders)
                        idxs_data_separators.pop(0)
                        break
                    else:
                        if idxs_data_separators[0] > visited_dividers[-1]:
                            df = pd.read_csv(filename, sep="\t", header = None, skiprows = visited_dividers[-1], nrows=idxs_data_separators[0]-visited_dividers[-1]-1, usecols=data_indexes, engine="c", na_filter=False, low_memory=False)
                        else:
                            df = pd.read_csv(filename, sep="\t", skiprows = idxs_dividers[4*function+2], nrows=idxs_data_separators[0]-idxs_dividers[4*function+2]-2, usecols=data_indexes, engine="c", na_filter=False, low_memory=False)
                            dataHeaders = df.columns.values
                        dfData['Ramp-{}'.format(nSeparators + 1)] = prepareData(df, dataHeaders)
                        visited_dividers.append(idxs_data_separators.pop(0))
                    nSeparators += 1
                dfDATA = {"<DATA>": dfData}
            dfINFO.update(dfFUNCTION)
            dfINFO.update(dfDATA)
            dM = {currentFunc : dfINFO}
            dfMACH_1.update(dM)
        else:
            dfINFO.update(dfFUNCTION)
            dM = {funcName : dfINFO}
            dfMACH_1.update(dM)
        if  showProgressBar:
            printProgressBar(function, len(idxs_dividers)//4 - 1, prefix='Progress:', suffix='Complete', length=50)            
    return dfMACH_1
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: load_routine
% Description: Load routine for multiprocessing
% Inputs:   mach_1_set   - input parameters for read_mach_1_file function
% Output:   mach_1_files -  List containing structures of mach-1 files 
% 
%   By: Renato Castillo, 2FEB2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def load_routine(mach_1_set):
    mach_1_dir, mach_1_fname, read_data, data_headers_selection, showProgressBar = mach_1_set
    print(f"Loading - {mach_1_fname}...")
    dfMach_1 = {mach_1_fname : read_mach_1_file(mach_1_dir, read_data=read_data, data_headers_selection= data_headers_selection, showProgressBar=showProgressBar)}
    return dfMach_1 
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: read_mach1_files
% Description: Reads multiple mach-1 files in a folder
% Inputs:   read_data         -   Integer indicating if the numerical data in the Mach-1 .txt
%                                 file should be read. Default value is 1.
%                                 0: Read only the info
%                                 N-null: Read info and numerical data
%           data_indexes_file -   Load specific data columns - list of int
%           showProgressBar -   Bool flag to show progress bar on command prompt
% Output:   dfMach_1s         -   Dictionary of loaded mach-1 files data
% 
%   By: Renato Castillo, 29MAY2023
%                 UPDATED 2FEB2024
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def read_mach_1_files(read_data = 1, data_headers_selection = None, keyword=None, showProgressBar = True, multiproc = False):
    dfMACH_1s = {}
    mach_1_dir = None
    mach_1_sets = select_mach_1_files_dir(keyword)
    if len(mach_1_sets) > 0:
        if multiproc:
            for mach_1_set in list(mach_1_sets):
                mach_1_set.append(read_data)
                mach_1_set.append(data_headers_selection)
                mach_1_set.append(showProgressBar) 
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(load_routine, mach_1_sets)
            dfMACH_1s =  {}
            for result in results:
                dfMACH_1s.update(result)
            mach_1_dir = os.path.dirname(mach_1_sets[0][0])
        else:
            for index in range(len(mach_1_sets)):
                print("Loading: " + mach_1_sets[index][1])
                if data_headers_selection is not None:
                    dfMACH_1s[mach_1_sets[index, 1]] = read_mach_1_file(mach_1_sets[index, 0], read_data, data_headers_selection, showProgressBar)
                else:
                    dfMACH_1s[mach_1_sets[index, 1]] = read_mach_1_file(mach_1_sets[index, 0], read_data, showProgressBar=showProgressBar)
            mach_1_dir = os.path.dirname(mach_1_sets[0, 0])
    else:
        messagebox.showwarning("Warning", "No mach-1 files found!")
    return dfMACH_1s, mach_1_dir
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: search_dividers
% Description: Searches the mach-1 file for a specific divider
% Inputs:   mach_1_file     -   String containing the .txt file name generated by 
%                               Mach-1 Motion
%           divider         -   String containing the name of divider
% Output:   indexes         -   List of row position of the given divider in the .txt 
%                               file
% 
%   By: Renato Castillo, 25AUG2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def search_dividers(mach_1_file, divider):
    with open(mach_1_file) as file:
        indexes = [] # data structure to store divider row position in .txt file
        for idx, line in enumerate(file, 1):
            if divider in line:
                indexes.append(idx)
    return indexes
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: search_dividers_idx
% Description: Extracts the row position of all dividers in text mach-1 file
% Inputs:   filename    -   String containing the .txt file name generated by Mach-1
%                           Motion
%           dividers    -   List of strings containing all dividers used in mach-1 .txt 
%                           file 
%                           ("<INFO>", "<END INFO>", "<DATA>", "<END DATA>")
% Output:   indexes_file-   List of row positions in order of the given dividers found 
%                           in the .txt file
% 
%   By: Renato Castillo, 25AUG2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def search_dividers_idx(filename, dividers):
    indexes_file = []
    for divider in dividers:
        indexes_file += search_dividers(filename, divider)
    indexes_file.sort()
    return indexes_file
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: check_data_separator
% Description: Verifies that a function in the mach-1 file has a data separator 
%              indicated by "<divider>" .
% Inputs:   mach_1_file            -   String containing the .txt file name generated 
%                                      by Mach-1 Motion
% Output:   flag_data_separators   -   List Booleans that checks if "<divider>" is in a
%                                      function of the mach-1 file
%           idx                    -   Position of the row of "<divider>" if is in a 
%                                      function of the mach-1 file
% 
%   By: Renato Castillo, 25AUG2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def check_data_separator(mach_1_file, index_dividers):
    indexes = []
    flag_dividers = []
    flag_divider = False
    with open(mach_1_file) as file:
        for id_line, line in enumerate(file, 1):
            if "<divider>" in line:
                indexes.append(id_line)
    
    for idx in range(len(index_dividers)//4):
        if len(indexes) > 0:
            for idy in indexes:
                if idy > index_dividers[4*idx+2] and idy < index_dividers[4*idx+3]:
                    flag_divider = True
                    break
            flag_dividers.append(flag_divider)
            flag_divider = False
            
        else:
            flag_dividers.append(False)
    return flag_dividers, indexes



    