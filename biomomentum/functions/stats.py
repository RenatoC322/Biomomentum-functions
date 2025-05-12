import numpy as np

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