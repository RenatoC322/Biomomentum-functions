import numpy as np

def compliance_correction(displacement, load):
    sort_displacement, indices = np.unique(displacement, return_inverse = True)
    sort_load = np.zeros_like(sort_displacement, dtype = float)
    for idx in range(len(sort_displacement)):
        x = 0
    return 0