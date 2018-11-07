from models.utils import MatthewsDaviesAlgorithm
import numpy as np
from autograd import hessian

f_x = lambda x: x[0]**2 - x[1] **0.5 + x[0]*x[1]*x[2]**2 + x[3]

x = np.array([0, 1,2, 0.5], dtype = np.float64)

hessian_f  = hessian(f_x)
hessian_k = hessian_f(x)

H, L, D  = MatthewsDaviesAlgorithm(hessian_k)