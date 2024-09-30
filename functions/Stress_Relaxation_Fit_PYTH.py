#######################################################################################
# Contact:  -   castillo.renato.90@gmail.com
#           -   castillo@biomomentum.com
######################################################################################
import numpy as np
import scipy as sp
import os

def funfminsearch(tau, x, yy, t1, t2, t0, sz1, w):
    szm = sp.interpolate.interp1d(x[0], np.log10(yy),fill_value="extrapolate")
    sz21a = 10**szm(t1/tau)
    sz21 = 0.125 - sz21a
    sz22a = 10**szm((t2-t0)/tau)
    sz22b = 10**szm(t2/tau)
    sz22 = sz22a - sz22b
    sz2 = np.hstack((sz21,sz22))
    A = w*sz1
    B = np.array([w*sz2])
    K = np.linalg.lstsq(B.T, A.T,rcond=None)[0]
    K = np.dot(A, np.linalg.pinv(B))[0] 
    ser = np.sum(w*(K*sz2 - sz1)**2)
    return ser

def function_K(R, vm, tau, szequ, K):
    f = K -1*(szequ*tau*(-1 + 2*vm)*(1 + R + R*vm)**2*(-1 + vm + R*(-1 + vm + 2*vm**2)))/((-1 + R*(-1 + vm))*(1 + vm)*(-1 + R*(-1 + vm + 2*vm**2)))
    dfdR = -1*(szequ*tau*(-1 + 2*vm)*(1 + R + R*vm)*(-1 - 3*vm + 4*vm**2 + 2*vm**3 + R**3*(1 - 2*vm)**2*(-1 + vm)*(1 + vm)**3 - R**2*(1 + vm)**2*(3 - 7*vm - 4*vm**2 + 12*vm**3) + R*(-3 - 4*vm + 15*vm**2 + 10*vm**3 - 6*vm**4)))/((-1 + R*(-1 + vm))**2*(1 + vm)*(-1 + R*(-1 + vm + 2*vm**2))**2)
    return f, dfdR

def Newton(R0, eps, vm, tau, szequ, K):
    R = R0
    f_value, df_value = function_K(R, vm, tau, szequ, K)
    iteration_counter = 0
    
    while abs(f_value) > eps and iteration_counter < 100:
        try:
            R = R - f_value/df_value
        except:
            print("Error! - derivative zero for R = ", R)
            return 0, -1
        f_value, df_value = function_K(R, vm, tau, szequ, K)
        iteration_counter += 1
    if abs(f_value) > eps:
        iteration_counter = -1
    return R, iteration_counter

def Newton_Raphson_Method(veff0, vm, tau, szequ, K):
    R_guess = (vm - veff0)/(veff0*(1 + vm)*(1 - 2*vm))
    eps = 1e-6
    R_new, nIterations = Newton(R_guess, eps, vm, tau, szequ, K)
    if nIterations > 0:
        veff = vm/(1 + (1 + vm)*(1 - 2*vm)*R_new)
    else:
        print("Abort execution...")
        return 0
    return veff

def relaxation_constant(data, time, startIdx):
    dataRange = data[startIdx] - data[-1]
    lc = np.argmax(data[startIdx:] - data[-1] < np.exp(-1)*abs(dataRange)) + startIdx
    tau = time[lc] - time[startIdx]
    return tau
def relaxation_constant_v2(data, time, startIdx):
    dataRange = data[startIdx] - data[-1]
    
    for lc in range(startIdx,len(data)):
        if abs(data[lc] - data[-1]) < np.exp(-1)*abs(dataRange):
            break
    tau = time[lc] - time[startIdx]
    return tau
def stressrelaxation_fit(t, sz, r0, e0, vm, t0in = None, filemat = None):
    """
    Function to fit stree relaxation curves.

    Parameters
    ----------
    t       :  Data array (s)
    sz      :  Stress array (MPa)
    r0      :  Disk radius (mm)
    e0      :  Disk stress
    vm      :  Poisson Ratio
    t0in    :  Compression time (s)
    filemat :  Matlab file (sveff_tab.mat)
    
    Returns
    -------
    szfit :    Stress array fit (MPa)
    ef    :    Elastic Fibril Modulus (MPa)
    k0    :    Hydraulic permeability (mm2/MPa.s)
    e33   :    Elastic Equilibrium Modulus (MPa)
    t0    :    Compression time (s)
    S11   :    Elastic Modulus S11 (MPa)
    szequ :    Equilibrium stress (MPa)
    K     :    Coefficient (MPa.s)
    tau   :    Time constant (s)
    em    :    Elastic Modulus of isotropic matrix (MPa)
    nm    :    Poisson coefficient of isotropic matrix
    mse   :    Mean quadratic error                            
    """
    if filemat is None:
        filemat = os.path.join(os.getcwd(),"sveff_tab.mat")

    sz0 = sz[0]
    sz = sz - sz[0]
    t = t - t[0]
    
    if t0in is None:
        t0i = np.argmax(sz)
        t0 = t[t0i]
    else:
        t0 = t0in
        t0is = np.where(t > t0)
        t0i = t0is[0] - 1
    szequ = sz[-1]
    e33 = szequ/e0
    matfile = sp.io.loadmat(filemat)
    sveffm = matfile["sveffm"] 
    veffv = matfile["veffv"]
    x = matfile["x"]
    
    w = t[1:] - t[:-1]
    w = np.hstack((0,w))
    t1 = t[:t0i+1]
    t2 = t[t0i+1:]
    sz11 = sz[:t0i+1]*t0 - szequ*t1
    sz12 = t0*(sz[t0i+1:] - szequ)
    sz1 = np.hstack((sz11, sz12))
    veff = 0.01
    veff_new = vm
    count = 1
    
    while abs(veff_new - veff) > 1e-6 and count < 100:
        veff = veff_new
        if veff == 0:
            yy = sveffm[0,:]
        else:
            _, M = np.shape(sveffm)
            yy = np.zeros(M)
            for id in range(M):
                cs = sp.interpolate.UnivariateSpline(veffv[0], sveffm[:,id])
                yy[id] = cs(veff)
            
        tau, se, _, _ = sp.optimize.fminbound(lambda tau : funfminsearch(tau, x, yy, t1, t2, t0, sz1, w), 0, 2*t[-1], full_output=True)
        szm = sp.interpolate.interp1d(x[0], np.log10(yy),fill_value="extrapolate")
        sz21a = 10**szm(t1/tau)
        sz21 = 0.125 - sz21a
        sz22a = 10**szm((t2-t0)/tau)
        sz22b = 10**szm(t2/tau)
        sz22 = sz22a - sz22b
        sz2 = np.hstack((sz21,sz22))
        A = w*sz1
        B = np.array([w*sz2])
        K = np.linalg.lstsq(B.T, A.T,rcond=None)[0]
        K = np.dot(A, np.linalg.pinv(B))[0]
        veff_new = Newton_Raphson_Method(veff, vm, tau, szequ, K)
        count = count + 1
    sz1fit = (szequ*t1 + K*sz21)/t0
    sz2fit = szequ + K*sz22/t0
    szfit = np.hstack((sz1fit, sz2fit))
    szfit = szfit + sz0
    k0 = e0*r0**2*(1 - 2*veff)**2/K
    S11 = r0**2/(k0*tau)
    ef = S11*(1 + veff)*(1 - 2*veff)/(1 - veff) - e33
    
    emnmd = e33 + ef*veff + 2*ef*veff**2
    em = (e33**2 + ef*e33*veff - 2*(ef*veff)**2)/emnmd
    nm = (ef + e33)*veff/emnmd
    mse = se/np.sum(w)
    return szfit, ef, k0, e33, t0, S11, szequ, K, tau, em, nm, mse, veff