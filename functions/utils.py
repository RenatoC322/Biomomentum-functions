import pandas as pd
import os
import tkinter
import re
import numpy as np

from tkinter import filedialog, messagebox

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
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: select_file
% Description: Function that brings a pop up prompt to select the a file
% Inputs:   None
% Output:   filename    - String address of the file 
% 
%   By: Renato Castillo, 29AUG2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def select_mach_1_file(prompt = ""):
    pop_window = tkinter.Tk()
    pop_window.iconbitmap(os.getcwd() + r"\ICON.ico")
    pop_window.withdraw()
    pop_window.attributes('-topmost', True) 
    filename = filedialog.askopenfilename(parent=pop_window, initialdir= "/", title=prompt)
    if len(filename) == 0 or not filename.endswith(".txt"):
        messagebox.showwarning("Warning", "No mach-1 file selected!")
        filename = None
    return filename
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