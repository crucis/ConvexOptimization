import numpy as np
from copy import copy


def MatthewsDaviesAlgorithm(hess_k):
    # step 1
    hessian_k = copy(hess_k)
    hessian_size = hessian_k.shape[0]
    lowerTriangular = np.zeros_like(hessian_k)
    diagonalMatrix = np.zeros_like(hessian_k)
    if hessian_k[0][0] > 0:
        h_00 = hessian_k[0][0]
    else:
        h_00 = 1
    
    # step 2
    for k in range(1, hessian_size):
        m = k - 1
        lowerTriangular[m][m] = 1
        if hessian_k[m][m] <= 0:
            hessian_k[m][m] = h_00
        # step 2.1
        for i in range(k, hessian_size):
            lowerTriangular[i][m] = -hessian_k[i][m]/hessian_k[m][m]
            #lowerTriangular[i][m] = hessian_k[i][m]/hessian_k[m][m]
            hessian_k[i][m] = 0
            # step 2.1.1
            for j in range(k, hessian_size):
                hessian_k[i][j] = hessian_k[i][j] + lowerTriangular[i][m] * hessian_k[m][j]
        if (0 < hessian_k[k][k]) and (hessian_k[k,k] < h_00):
            h_00 = hessian_k[k,k]
    # step 3
    lowerTriangular[hessian_size - 1,hessian_size - 1] = 1
    if hessian_k[hessian_size - 1,hessian_size - 1] <= 0:
        hessian_k[hessian_size - 1,hessian_size - 1] = h_00
    for i in range (hessian_size):
        diagonalMatrix[i,i] = hessian_k[i,i]
    return hessian_k, lowerTriangular, diagonalMatrix