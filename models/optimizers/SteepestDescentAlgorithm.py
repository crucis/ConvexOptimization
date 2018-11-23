from .optimizer import optimizer
import autograd.numpy as np
from copy import copy


def _steepest_descent_algorithm(x_0, objectiveFunction, maxIter, line_search=None, xtol=1e-6):
    x_k = np.array(x_0, dtype=np.float64)
    if line_search is None:
        a0 = 1
        f0 = objectiveFunction(x_k)
    for _ in range(maxIter):
        grad_x = objectiveFunction.grad(x_k)
        direction_vector = copy(-grad_x)
       
        if line_search is not None:
            a0, _  = line_search(x_k, direction_vector)
        else:
            f_hat = objectiveFunction(x_k - a0*grad_x)
            a0 = (np.dot(grad_x.T, grad_x) * (a0 **2)) / (2 * (f_hat - f0 + a0 * np.dot(grad_x.T, grad_x)))
        x_k = x_k + a0* direction_vector
        if line_search is None:
            f0 = objectiveFunction(x_k)

        if np.linalg.norm(a0 * direction_vector, ord=2) < xtol:
            break
    return x_k, a0

class SteepestDescentAlgorithm(optimizer):
    def __init__(self,
                 func,
                 x_0,
                 line_search_optimizer=None,
                 maxIter = 1e3,
                 xtol = 1e-6,
                 ftol = 1e-6):
        if hasattr(line_search_optimizer, '_line_search'):
            self.line_search = line_search_optimizer._line_search
        else:
            self.line_search = line_search_optimizer
        self.objectiveFunction = func
        self.x_k = x_0
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol)

    def find_min(self):
        self.x_k, a0 = _steepest_descent_algorithm(self.x_k, 
                                                    self.objectiveFunction, 
                                                    self.maxIter, 
                                                    self.line_search, 
                                                    self.xtol)
        return self.x_k