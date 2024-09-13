#######################################################################################
# Contact:  -   castillo.renato.90@gmail.com
#           -   castillo@biomomentum.com
######################################################################################
import numpy as np
#from read_mach_1_file_PYTH import *
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: isNegative
% Description: Check for positive sign for Z-position
% Inputs:   posZ  - Array Z-position (mm)
% Output:   posZ  -  Array Z-position (mm) with positive sign
% 
%   By: Renato Castillo, 11Dec2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def isNegative(posZ):
    if abs(posZ[-1]) < abs(posZ[0]):
        posZ = posZ + 2*abs(posZ[0])
    return posZ
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: rsquared
% Description: Extracts statistical R-squared
% Inputs:   Y           - Signal Fitted
%           mse         - Mean Squared Error
%           poly_order  - Order of fit polynomial
% Output:   Rsq_adj     -  R-squared
% 
%   By: Renato Castillo, 11Dec2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def rsquared(Y, mse, poly_order):
    N = len(Y)
    Rsq = 1 - mse/np.var(Y)
    Rsq_adj = 1 - (1 - Rsq)*(N - 1)/(N - poly_order - 1)
    return Rsq_adj
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: check_data
% Description: Checks if data passes statistical tests
% Inputs:   loadZ     - Array Z-load (N or gf)
%           posZ      - Array Z-position (mm)
%           test_num  - test# (default = 1)
%           Rsq_req   - required R**2 value for test to be accepted
% Output:   Rsq_adj   - R-squared
% 
%   By: Renato Castillo, 11Dec2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def check_data(loadZ, posZ, test_num, Rsq_req):
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
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: Hayes_Model
% Description: Extracts Hayes Model Coefficients
% Inputs:   w0  - Indentation amplitude
%           h   - Sample thickness
%           R   - Indenter radius
%           v   - Poisson's ratio
% Output:   a   - Hayes coefficient
%           K   - Kappa coefficient
% 
%   By: Renato Castillo, 11Dec2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def Hayes_Model(w0,h,R,v):
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
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: Hayes_Model
% Description: Extracts Elastic Properties from Indentation
% Inputs:   posZ                - Array Z-position (mm)
%           loadZ               - Array Z-load (gf)
%           gf_flag             - Bool to indicate whether loadZ units is gf
%           maxStrain           - Maximum strain
%           R                   - Radius of indenter in mm
%           v                   - Poisson's ratio
%           test_num            - Test number
%           Rsq_req             - Required fit R square value (usually 0.95)
%           sampleThickness     - Sample thickness in mm
%           origin_set          - Bool to indicate whether signal starts at origin
%           eqModulus           - Bool to indicate whether signal only fits 2 points
% Output:   G                   - Indentation Shear Modulus in MPa
%           E                   - Indentation Elastic Modulus in MPa
%           Fit                 - Fit for posZ and loadZ using Hayes spherical model 1972
%           Rsq_adj             - R-squared value for fit
% 
%   By: Renato Castillo, 11Dec2023
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def HayesElasticModel(posZ, loadZ, gf_flag, maxStrain, R, v, test_num, Rsq_req, sampleThickness = None, origin_set=False, eqModulus=False):
    gf_to_N = 0.00980655
    if gf_flag:
        loadZ = loadZ*gf_to_N
    posZ = isNegative(posZ)
    req = check_data(loadZ, posZ, test_num, Rsq_req)
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
        #Zmax = posZ[ZmaxIdx]
        z0 = posZ[0]
        if eqModulus:
            ZmaxIdx = len(posZ)
        for k in range(ZmaxIdx):
            if not eqModulus:
                if posZ[k] > (1+maxStrain)*z0:
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
            print('WARNING! Max strain is %.2g. Curvefit is up to max strain!\n', w0/z0)
        M = M[:k+1,:]
        #condId = ~np.all(M == 0, axis=1)
        #M = M[condId]
        #condId2 = np.argwhere(M[:,1] != 0)
        #condId2 = np.reshape(condId2,(1,len(condId2)))
        #M = M[condId2][0]
        #print(M[condId2])
        N = len(M)
        if origin_set:
            b0 = M[0,0]
            B = M[:,0] - b0
            A = M[:,1]
            
            G = np.linalg.solve(np.dot(A.reshape(1,N),A.reshape(N,1)),np.dot(A.reshape(1,N),B))
            FitLoadZ = b0 + G*A
            G = G[0]
        else:
            A = np.zeros((N,2))
            B = M[:,0] 
            A[:,0] = 1
            A[:,1] = M[:,1]
            Soln = np.linalg.solve(np.dot(A.transpose(),A),np.dot(A.transpose(),B))
            G = Soln[1]
            FitLoadZ = np.dot(A,Soln)
        E = 2*G*(1+v)
        FitPosZ = posZ[:k+1]
        FitLoadZ_c = FitLoadZ/gf_to_N
        #Fit = np.hstack((FitPosZ.reshape(N,1),FitLoadZ_c.reshape(N,1)))
        #FitPosZ = FitPosZ[condId]
        #FitPosZ = FitPosZ[condId2]
        Fit = np.hstack((FitPosZ.reshape(N,1),FitLoadZ_c.reshape(N,1)))
        mse = np.sum((B - FitLoadZ)**2)/N
        if eqModulus:
            Rsq_adj = -1
        else:
            Rsq_adj = rsquared(B, mse, 1)
    return G, E, Fit, Rsq_adj


