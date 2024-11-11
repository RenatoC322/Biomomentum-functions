import re
import numpy as np

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, Rbf

def sorted_alphanumeric(files):
    """
    Sorts alpha numerically files from directory.

    Args:
        files (list): Files from directory to sort.

    Returns:
        files sorted: files sorted.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(files, key=alphanum_key)

def get_super(x):
    """
    Insert super script char on string

    Args:
        x (str): Character to super script.

    Returns:
        Char in super script format.
    """ 
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s)) 
    return x.translate(res)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Calls in a loop to create a terminal progress bar

    Args:
        iteration (int): current iteration
        total (int): total iterations
        prefix (str): String to put before loading bar
        suffix (str): String to put following loading bar
        decimals (int): positive number of decimals in percent complete
        length (int): character length of bar
        fill (str): bar fill character
        printEnd (str): end character (e.g. "\r", "\r\n")

    Returns:
        None
    """ 
    if total == 0:
        total = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total:
        print()

def inDict(dic, key):
    """
    Checks if key is in dictionary

    Args:
        key (str): Key to check if is in dictionary.

    Returns:
        flag (Bool): True if key is in dictionary.
    """ 
    flag = False
    if key in dic.keys():
        flag = True
    return flag

def ResidualStandardError(x, xfit, predictors):
    """
    Residual Error from the fit to function.

    Args:
        x (np.array): independent variable values (assumed to be error-free) 
        xfit (np.array): signal fit of xfit                                     
        predictor (int): number of predictors of the function
    
    Returns:
        ser (float): standard residual error                               
    """
    N = len(x)
    ssr = np.sum((x - xfit)**2)
    ser = np.sqrt(ssr/(N - predictors))
    return ser

def isNegative(posZ):
    """
    Check for positive sign for Z-position

    Args:
        posZ (np.array): Array Z-position (mm)
    
    Returns:
        posZ (np.array): Array Z-position (mm) with positive sign                               
    """
    if abs(posZ[-1]) < abs(posZ[0]):
        posZ = posZ + 2*abs(posZ[0])
    return posZ
def rsquared(Y, mse, poly_order):
    """
    Extracts statistical R-squared

    Args:
        Y (np.array): Signal Fitted 
        mse (float): Mean Squared Error of the fit                                     
        poly_order (int): number of predictors of the function
    
    Returns:
        Rsq_adj (float): Adjusted R-squared                           
    """
    N = len(Y)
    Rsq = 1 - mse/np.var(Y)
    Rsq_adj = 1 - (1 - Rsq)*(N - 1)/(N - poly_order - 1)
    return Rsq_adj

def check_data(loadZ, posZ, Rsq_req):
    """
    Checks if data passes statistical tests

    Args:
        loadZ (np.array): Array Z-load (N or gf)
        posZ (np.array): Array Z-position (mm)
        Rsq_req (float): required R**2 value for test to be accepted
    
    Returns:
        req (int): req == 1 it fails the test.                               
    """
    poly_order = 2
    N = len(loadZ)
    X = np.zeros((N, poly_order + 1))
    Y = loadZ
    
    for k in range(N):
        X[k,0] = 1
        for l in range(poly_order):
            X[k, 1 + l] = posZ[k]**(l+1)
    B = np.linalg.solve(np.dot(X.transpose(),X),np.dot(X.transpose(),Y))
    mse = np.sum((Y - np.dot(X,B)**2))/N
    Rsq_adj = rsquared(Y, mse, poly_order)
    if Rsq_adj < Rsq_req:
        req = 0
    else:
        req = 1
    return req

def linear_least_square(x,y):
    """
    Args:
    x     : data array - independent variable (data units)
    y     : data array - dependent variable (data units)
    
    Returns:
    A : matrix of linear fit  
    curveFit : linear fit on data
    Rsq_adj : R-squared                         
    """
    X = np.zeros((len(x),2))
    Y = y
    X[:,0] = x
    X[:,1] = 1
    A = np.linalg.solve(np.dot(X.transpose(),X),np.dot(X.transpose(), Y))
    N = len(Y)
    curveFit = np.dot(X,A)
    poly_order = 1
    mse = np.sum((Y - curveFit)**2)/N
    Rsq = 1 - mse/np.var(Y)
    Rsq_adj = 1 - (1 - Rsq)*(N - 1)/(N - poly_order - 1)
    return A, curveFit, Rsq_adj

def interpolateMAP(subSurfaces, interpolate_to_bounds = False, smooth_data = False, threshold = 4, keyword = ""):
    """
    Function to apply 2D linear interpolation into the data

    Args:
        subSurfaces : Dictionary of all the surfaces identified in the MAP file
        threshold : threshold standard deviation to control smoothing.
        interpolate_to_bounds : Flag to indicate whether to extrapolate values to surface bounds
        keyword : Name given to the measurements in the MAP file
    
    Returns:
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
    triangles = Delaunay(pos)
    if smooth_data:
        QP = smoothMAP(QP, triangles, threshold)
    interpolator = LinearNDInterpolator(pos, QP)
    QP_2D = interpolator(grid_X, grid_Y)
    return QP_2D, triangles, grid_X, grid_Y

def smoothMAP(QP, triangles, threshold):
    """
    Function to smooth data for interpolation

    Args:
        QP : the original measured data to be smoothed
        triangles : list of lists, each sublist contains the indices of neighbors for each data point.
        threshold : threshold standard deviation to control smoothing.
    
    Returns:
        smoothed_map : the smoothed data
    """
    neighbors = {i: set() for i in range(len(QP))}
    for simplex in triangles.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                neighbors[simplex[i]].add(simplex[j])
                neighbors[simplex[j]].add(simplex[i])
    neighbors = {key: list(val) for key, val in neighbors.items()}
    smoothed_map = np.copy(QP)
    for id in range(len(QP)):
        neighbors_idx = neighbors[id]
        neighbors_data = QP[neighbors_idx]
        mean_neighbor = np.mean(neighbors_data)
        std_neighbor = np.std(neighbors_data)
        if std_neighbor < threshold:
            smoothed_map[id] = mean_neighbor
    return smoothed_map