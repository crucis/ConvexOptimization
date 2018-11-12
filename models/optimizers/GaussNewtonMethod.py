from .optimizer import optimizer
from ..utils import MatthewsDaviesAlgorithm
import autograd.numpy as np
from autograd import jacobian
from copy import copy

def _gaussian_newton_algo(x_0, objectiveFunction_list, line_search, maxIter=1000, xtol=1-6):
    """
    objectiveFunction_list must be a callable that returns an array of floats.
    Example:
    objectiveFunction_list = lambda x: np.array([f1(x), f2(x)], dtype=np.float64)    
    """
    assert hasattr(objectiveFunction_list, '__call__'),     """
                            objectiveFunction_list must be a callable that returns an array of floats.
                            Example:
                            objectiveFunction_list = lambda x: np.array([f1(x), f2(x)], dtype=np.float64)    
                            """
    
    # step 1
    x_k = np.array(x_0, dtype=np.float64)
    
    # step 2
    f_k = objectiveFunction_list(x_k)
    F_k = f_k.T @ f_k
    jac_f = jacobian(objectiveFunction_list)

    for _ in range(maxIter):
        # step 3
        jac_k = jac_f(x_k)
        g_k = 2* jac_k.T @ f_k
        H_k = 2* jac_k.T @ jac_k

        # step 4
        H_k, L_k, D_k = MatthewsDaviesAlgorithm(H_k)
        y_k = - L_k @ g_k
        d_k = (L_k.T * np.linalg.inv(D_k)) @ y_k


        # step 5
        alpha_k, F_k_new = line_search(x_k, d_k)#, H_k, g_k)

        #print(alpha_k)
        #input()
        # step 6
        x_k = x_k + alpha_k * d_k
        #f_k = objectiveFunction_list(x_k)
        #F_k_new = f_k.T @ f_k
        #print(alpha_k)
        #input()
        # step 7
        if np.abs(F_k_new - F_k) < xtol:
            break
    return x_k, f_k, F_k_new


class GaussNewtonMethod(optimizer):
    def __init__(self,
                func,
                x_0,
                line_search_optimizer,
                maxIter=1e3,
                xtol=1e-6,
                ftol=1e-6):
        if hasattr(line_search_optimizer, '_line_search'):
                self.line_search = line_search_optimizer._line_search
        else:
            self.line_search = line_search_optimizer
        self.objectiveFunction = func
        self.x_k = x_0
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol)


    def find_min(self):
        self.x_k, _, _ = _gaussian_newton_algo(self.x_k,
                                                self.objectiveFunction,
                                                self.line_search,
                                                self.maxIter,
                                                self.xtol)
        return self.x_k