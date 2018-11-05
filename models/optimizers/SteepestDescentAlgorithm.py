from .optimizer import optimizer
import autograd.numpy as np
from copy import copy


def _steepest_descent_algorithm(x_0, objectiveFunction, maxIter, line_search=None, xtol=1e-6):
    x_k = copy(x_0)
    grad_x = copy(objectiveFunction.grad(x_k))
    direction_vector = copy(-grad_x)
    if line_search is None:
        a0 = 1
        f0 = objectiveFunction(x_k)
    for _ in range(maxIter):
        if line_search is not None:
            a0, f0  = line_search(direction_vector)
        else:
            f_hat = objectiveFunction(x_k - a0*grad_x)
            a0 = (np.dot(grad_x, grad_x) * (a0 **2)) / (2 * (f_hat - f0 + a0 * np.dot(grad_x, grad_x)))
        x_k = x_k + a0* direction_vector
        if line_search is None:
            f0 = objectiveFunction(x_k)
        grad_x = copy(objectiveFunction.grad(x_k))
        if np.linalg.norm(a0 * direction_vector) < xtol:
            break
        direction_vector = copy(-grad_x)
        #direction_vector = copy(-grad_x/np.linalg.norm(grad_x))
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