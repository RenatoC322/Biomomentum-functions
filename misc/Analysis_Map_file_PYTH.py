
import numpy as np
import cv2
import os
import re
import tkinter
import matplotlib.pyplot as plt
import pandas as pd

from tkinter import messagebox
from tkinter import filedialog
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, griddata, NearestNDInterpolator, Rbf, CubicSpline, RBFInterpolator, SmoothBivariateSpline
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter, spline_filter

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def select_file():
    pop_window = tkinter.Tk()
    pop_window.withdraw()
    pop_window.attributes('-topmost', True) 
    filename = filedialog.askopenfilename(parent=pop_window, initialdir= "/", title='Please select the file')
    return filename

def select_files_dir(keyword=None):
    dir_files = []
    directories = []
    pop_window = tkinter.Tk()
    pop_window.iconbitmap(os.getcwd() + r"\ICON.ico")
    pop_window.withdraw()
    pop_window.attributes('-topmost', True) 
    main_dir = filedialog.askdirectory(parent=pop_window, initialdir= "/", title='Please select MAP files in directory')
    if main_dir:
        dir_files = sorted(filter(lambda x: os.path.isfile(os.path.join(main_dir, x)), os.listdir(main_dir)))
        MAP = [os.path.join(main_dir, file) for file in dir_files if file.endswith(".csv") or file.endswith(".map") or file.endswith(".jpg")]
        if keyword:
            MAP = [file for file in MAP if keyword in file.split("\\")[-1]]
        directories = sorted_alphanumeric(MAP)
    else:
        messagebox.showwarning(master=pop_window, title="Warning", message="No directory selected.")
    return directories, [file.split("\\")[-1] for file in directories]

def inDict(dic, key):
    flag = False
    if key in dic.keys():
        flag = True
    return flag

def get_subSurfaces(MAP_file, keyword = ""):
    """
    Function to separate surfaces from MAP in dictionary

    Parameters
    ----------
    MAP_file    :  Name of the MAP to load
    keyword     :  Name given to the measurements in the MAP file
    
    Returns
    -------
    subSurfaces :  Dictionary of all the surfaces identified in the MAP file                         
    """
    subSurfaces = {}
    df_MAP = pd.read_csv(MAP_file , sep="\t",  skiprows = 6, engine="c", na_filter=False, low_memory=False).to_dict()
    df_MAP_info = pd.read_csv(MAP_file , sep="\t",  nrows = 5, usecols = [1], engine="c", na_filter=False, low_memory=False).values
    subSurfaces["MAP-Info"] = {"Software version" : df_MAP_info[0, 0], "Image" : df_MAP_info[3, 0], "Image directory" : os.path.join(os.path.dirname(MAP_file), df_MAP_info[3,0])}
    keyword_flag = inDict(df_MAP, keyword)
    for id in range(len(df_MAP["PointType"])):
        pt_type = df_MAP["PointType"][id]
        if pt_type == 0 or pt_type == 2:
            subSurf_ID = df_MAP["Sub-SurfaceID"][id]
            if pt_type == 2:
                subSurf_ID = "references"
            pt = np.array([df_MAP["PixelX"][id], df_MAP["PixelY"][id]])
            pt_ID = df_MAP["PointID"][id]
            flag = inDict(subSurfaces, subSurf_ID)
            if flag:
                subSurfaces[subSurf_ID]["Position"].append(pt)
                subSurfaces[subSurf_ID]["Position ID"].append(pt_ID)
                if keyword_flag:
                    subSurfaces[subSurf_ID][keyword].append(float(df_MAP[keyword][id]))
            else:
                if keyword_flag:
                    subSurfaces[subSurf_ID] = {"Position": [pt], "Position ID": [pt_ID], keyword: [float(df_MAP[keyword][id])]}
                else:
                    subSurfaces[subSurf_ID] = {"Position": [pt], "Position ID": [pt_ID]}           
        elif pt_type == 1:
            subSurf_ID = df_MAP["Sub-SurfaceID"][id]
            Pixel = np.array([df_MAP["PixelX"][id], df_MAP["PixelY"][id]])
            flag = inDict(subSurfaces, subSurf_ID)
            if flag:
                bounds_flag = inDict(subSurfaces[subSurf_ID], "Bounds")
                if bounds_flag:
                    subSurfaces[subSurf_ID]["Bounds"].append(Pixel)
                else:
                    subSurfaces[subSurf_ID]["Bounds"] = [Pixel]
            else:
                subSurfaces[subSurf_ID] = {"Bounds": [Pixel]}                      
    return subSurfaces

def interpolateMAP(subSurfaces, interpolate_to_bounds = False, keyword = ""):
    """
    Function to apply 2D linear interpolation into the data

    Parameters
    ----------
    subSurfaces :           Dictionary of all the surfaces identified in the MAP file
    interpolate_to_bounds : Flag to indicate whether to extrapolate values to surface bounds
    keyword :           Name given to the measurements in the MAP file
    
    Returns
    -------
    QP_2D :  2D array of the interpolated values into the subSurface
    triangles : Triangles used for the interpolation
    grid_X : 2D array of the X values used to construct the interpolation
    grid_Y : 2D array of the Y values used to construct the interpolation
    """
    surface_1 = subSurfaces["QP"]
    pos = np.array(surface_1["Position"])
    QP = np.array(surface_1[keyword])
    boundary = np.array(surface_1["Bounds"])
    grid_X, grid_Y = np.meshgrid(np.linspace(min(pos[:, 0]), max(pos[:, 0]), int(np.ptp(pos[:, 0]))),
                                 np.linspace(min(pos[:, 1]), max(pos[:, 1]), int(np.ptp(pos[:, 1]))))
    if interpolate_to_bounds:
        rbf_interpolator = Rbf(pos[:,0], pos[:,1], QP, function='linear')
        QP = np.hstack((QP, rbf_interpolator(boundary[:,0], boundary[:, 1])))
        pos = np.vstack((pos, boundary))
        grid_X, grid_Y = np.meshgrid(np.linspace(min(pos[:, 0]), max(pos[:, 0]), int(np.ptp(pos[:, 0]))),
                                     np.linspace(min(pos[:, 1]), max(pos[:, 1]), int(np.ptp(pos[:, 1]))))
    interpolator = LinearNDInterpolator(pos, QP)
    QP_2D = interpolator(grid_X, grid_Y)
    triangles = Delaunay(pos)
    return QP_2D, triangles, grid_X, grid_Y
