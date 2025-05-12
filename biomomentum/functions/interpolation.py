import numpy as np
import cv2

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, Rbf

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
    QP_2D, triangles, grid_X, grid_Y = [], [], [], []
    for surface in subSurfaces:
        if surface not in ["MAP-Info", "REF"]:
            surface_1 = subSurfaces[surface]
            pos = np.array(surface_1["Image Position"])
            QP = np.array(surface_1[keyword])
            boundary = np.array(surface_1["Bounds"])
            grid_x, grid_y = np.meshgrid(np.linspace(min(pos[:, 0]), max(pos[:, 0]), int(np.ptp(pos[:, 0]))),
                                         np.linspace(min(pos[:, 1]), max(pos[:, 1]), int(np.ptp(pos[:, 1]))))
            if interpolate_to_bounds:
                rbf_interpolator = Rbf(pos[:,0], pos[:,1], QP, function='linear')
                QP = np.hstack((QP, rbf_interpolator(boundary[:,0], boundary[:, 1])))
                pos = np.vstack((pos, boundary))
                grid_x, grid_y = np.meshgrid(np.linspace(min(pos[:, 0]), max(pos[:, 0]), int(np.ptp(pos[:, 0]))),
                                             np.linspace(min(pos[:, 1]), max(pos[:, 1]), int(np.ptp(pos[:, 1]))))
            triangle = Delaunay(pos)
            if smooth_data:
                QP = smoothMAP(QP, triangle, threshold)
            interpolator = LinearNDInterpolator(pos, QP)
            QP_2d = interpolator(grid_x, grid_y)
            if interpolate_to_bounds:
                boundary -= np.min(boundary, axis = 0)
                M, N = QP_2d.shape
                mask = np.zeros((M, N))
                boundary = boundary.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [boundary], 1)
                QP_2d = np.where(mask == 1, QP_2d, np.nan)
            QP_2D.append(QP_2d)
            triangles.append(triangle)
            grid_X.append(grid_x)
            grid_Y.append(grid_y)          
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