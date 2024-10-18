import numpy as np
import scipy as sp
import os

from utils import isNegative, check_data, ResidualStandardError, rsquared, linear_least_square

def compliance_correction(displacement, load, criteria = 1, interval = None):
    """
    Extracts compliance correction factor for compression tests in mach-1

    Args:
        displacement (np.array): Displacement of sample in mm.
        load (np.array): Load of sample in N.
        criteria (float): Criteria to select data.
        interval (list): Interval to extract correction factor.

    Returns:
        correction_factor (float): Correction for compliance
    """
    sort_displacement, indices = np.unique(displacement, return_inverse = True)
    sort_load = np.zeros_like(sort_displacement, dtype = float)
    for idx in range(len(sort_displacement)):
        sort_load[idx] = np.mean(load[indices == idx])
    if interval is None:
        interval = np.array([np.where(np.diff(sort_load) > criteria)[0][0], len(sort_load) - 1])
    A, _, _ = linear_least_square(sort_load[interval[0]:interval[1]], sort_displacement[interval[0]:interval[1]])
    correction_factor = A[0]
    return correction_factor

def Hayes_Model(w0,h,R,v):
    """
    Extracts Hayes Model Coefficients

    Args:
        w0 (float): Indentation amplitude.
        h (float): Sample thickness.
        R (float): Indenter radius.
        v (float): Poisson's ratio.

    Returns:
        a (float): Hayes coefficient
        K (float): Kappa coefficient.
    """
    # Data extracted from Hayes1972
    if v == 0.3:
        A1, B1, C1 = (-0.1263,0.6325,1.015)
        A2, B2, C2 = (0.0841,0.5911,0.6293)
    elif v == 0.35:
        A1, B1, C1 = (-0.1342,0.668,1.0154)
        A2, B2, C2 = (0.1046,0.6235,0.6259)
    elif v == 0.4:
        A1, B1, C1 = (-0.1439,0.7203,1.0144)
        A2, B2, C2 = (0.1471,0.6547,0.6233)
    elif v == 0.45:
        A1, B1, C1 = (-0.1537,0.7974,1.0113)
        A2, B2, C2 = (0.2483,0.6539,0.6258)
    else:
        A1, B1, C1 = (-0.1462,0.9008,1.0067)
        A2, B2, C2 = (0.5737,0.4156,0.6875)
    if w0 > 0.0:
        # Take +ve root (physically makes sense that a > 0)
        a = (-B1 - np.sqrt(B1**2 - 4*(A1 - h**2 /(w0*R))*C1))/(2*(A1 - h**2/(w0*R)))*h
    else:
        a = 0   
    # Compute a and K to get G and E
    K = A2*(a/h)**2 + B2*(a/h) +C2 # compute Kappa
    return a, K

def HayesElasticModel(posZ, loadZ, gf_flag, maxStrain, R, v, Rsq_req, sampleThickness = None, origin_set=False, eqModulus=False):
    """
    Extracts Elastic Properties from Indentation

    Args:
        posZ (np.array): Array Z-position (mm).
        loadZ (np.array): Array Z-load (gf).
        gf_flag (Bool): Indicate whether loadZ units is gf.
        maxStrain (float): Maximum strain.
        R (float): Radius of indenter in mm.
        v (float): Poisson's ratio.
        Rsq_req (float): Required fit R square value (usually 0.95).
        sampleThickness (float): Sample thickness in mm.
        origin_set (Bool): Indicate whether signal starts at origin.
        eqModulus (Bool): Indicate whether signal only fits 2 points.

    Returns:
        G (float): Indentation Shear Modulus in MPa.
        E (float): Indentation Elastic Modulus in MPa.
        Fit (np.array): Fit for posZ and loadZ using Hayes spherical model 1972.
        Rsq_adj (float): Adjusted R-squared for the fit.
    """
    gf_to_N = 0.00980655
    if gf_flag:
        loadZ = loadZ*gf_to_N
    posZ = isNegative(posZ)
    req = check_data(loadZ, posZ, Rsq_req)
    if req == 0:
        print('... DATA INVALID. Bad Recording ...\n')
        G = -1
        E = -1
        Fit = None
        Rsq_adj = None
    else:
        req = 0
        l = len(posZ)
        M = np.zeros((l,2))
        ZmaxIdx = np.argmax(posZ)
        z0 = posZ[0]
        if eqModulus:
            ZmaxIdx = len(posZ)
        for k in range(ZmaxIdx):
            if not eqModulus:
                if posZ[k] > (1 + maxStrain) * z0:
                    req = 1
                    break
            w0 = posZ[k] - z0
            P = loadZ[k]
            if sampleThickness is not None:
                a, K = Hayes_Model(w0, sampleThickness, R, v)
            else:
                a, K = Hayes_Model(w0, z0, R, v)
            M[k,0] = P
            M[k,1] = 4*w0*a*K/(1-v)   
        if not req == 1:
            print('WARNING! Max strain is %.2g. Curvefit is up to max strain! ', w0/z0)
        M = M[:k + 1, :]
        N = len(M)
        if origin_set:
            b0 = M[0, 0]
            B = M[:, 0] - b0
            A = M[:, 1]
            
            G = np.linalg.solve(np.dot(A.reshape(1, N), A.reshape(N, 1)), np.dot(A.reshape(1, N), B))
            FitLoadZ = b0 + G*A
            G = G[0]
        else:
            A = np.zeros((N, 2))
            B = M[:, 0] 
            A[:, 0] = 1
            A[:, 1] = M[:, 1]
            Soln = np.linalg.solve(np.dot(A.transpose(), A), np.dot(A.transpose(), B))
            G = Soln[1]
            FitLoadZ = np.dot(A, Soln)
        E = 2*G*(1 + v)
        FitPosZ = posZ[:k + 1]
        FitLoadZ_c = FitLoadZ/gf_to_N
        Fit = np.hstack((FitPosZ.reshape(N, 1),FitLoadZ_c.reshape(N, 1)))
        mse = np.sum((B - FitLoadZ)**2)/N
        if eqModulus:
            Rsq_adj = -1
        else:
            Rsq_adj = rsquared(B, mse, 1)
    return G, E, Fit, Rsq_adj

def relaxation_constant(data, time, startIdx):
    """
    Extracts Relaxation Constant from time data

    Args:
        data (np.array): Data to extract tau.
        time (np.array): Time (s).
        startIdx (int): Index to indicate where to start the analysis.

    Returns:
        tau (float): Time constant.
    """
    dataRange = data[startIdx] - data[-1]
    lc = np.argmax(data[startIdx:] - data[-1] < np.exp(-1)*abs(dataRange)) + startIdx
    tau = time[lc] - time[startIdx]
    return tau

def stressrelaxation_fit(t, sz, r0, e0, vm, t0in = None, filemat = None):
    """
    Function to fit stress relaxation curves.

    Args
        t       :  Data array (s)
        sz      :  Stress array (MPa)
        r0      :  Disk radius (mm)
        e0      :  Disk stress
        vm      :  Poisson Ratio
        t0in    :  Compression time (s)
        filemat :  Matlab file (sveff_tab.mat)
    
    Returns
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
    def funfminsearch(tau, x, yy, t1, t2, t0, sz1, w):
        szm = sp.interpolate.interp1d(x[0], np.log10(yy), fill_value="extrapolate")
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