import numpy as np

def linear_least_square(x,y):
    """

    Parameters
    ----------
    x     : data array - independent variable (data units)
    y     : data array - dependent variable (data units)
    
    Returns
    -------
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

def compliance_correction(displacement, load, criteria = 1, interval = None):
    sort_displacement, indices = np.unique(displacement, return_inverse = True)
    sort_load = np.zeros_like(sort_displacement, dtype = float)
    for idx in range(len(sort_displacement)):
        sort_load[idx] = np.mean(load[indices == idx])
    if interval is None:
        interval = np.array([np.where(np.diff(sort_load) > criteria)[0][0], len(sort_load) - 1])
    A, _, _ = linear_least_square(sort_load[interval[0]:interval[1]], sort_displacement[interval[0]:interval[1]])
    correction_factor = A[0]
    return correction_factor