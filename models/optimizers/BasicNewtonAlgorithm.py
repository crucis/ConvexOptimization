# Modify the Newton algorithm described in Algorithm 5.3 by incorporating
# Eq. (5.13) into the algorithm. Give a step-by-step description of the
# modified algorithm.

from .optimizer import optimizer
import autograd.numpy as np
from autograd import hessian
from copy import copy

def _basic_newton_algo(x_0, objectiveFunction, line_search, maxIter=1000, xtol=1-6):
    x_k = copy(x_0)
    for _ in range(maxIter):
        # step 2
        grad_x = objectiveFunction.grad(x_k)
        hessian_x = hessian(objectiveFunction)(x_k)

        # modified hessian_x
        if _is_pos_def(hessian_x):
            beta = 1e-5
        else:
            beta = 1e5
        
        hessian_x = (hessian_x + beta * np.eye(hessian_x.shape[0])) / (1+beta)

        # step 3
        inv_hessian = np.linalg.inv(hessian_x)
        direction = - inv_hessian @ grad_x

        # step 4
        alpha, _ = line_search(x_k, direction)

        # step 5
        x_k = x_k + alpha * direction

        # step 6
        if np.linalg.norm(alpha * direction) < xtol:
            break
    return x_k, alpha


def _is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


class BasicNewtonAlgorithm(optimizer):
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
        self.x_k, alpha = _basic_newton_algo(self.x_k,
                                             self.objectiveFunction,
                                             self.line_search,
                                             self.maxIter,
                                             self.xtol)
        return self.x_k