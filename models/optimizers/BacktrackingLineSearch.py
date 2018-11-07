from .optimizer import optimizer
from .SteepestDescentAlgorithm import _steepest_descent_algorithm
import autograd.numpy as np
from copy import copy



class BacktrackingLineSearch(optimizer):
    def __init__(self, func, 
                        initial_x,
                        delta_x = None,
                        alpha = 0.49999, 
                        beta = 0.7,
                        interval = [-100, 100],
                        maxIter = 1e6, 
                        xtol = 1e-6, 
                        ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol, interval=interval)  
        assert (alpha > 0) and (alpha < 0.5), "Alpha must be in the range (0, 0.5)"
        assert (beta > 0) and (beta < 1), "Alpha must be in the range (0, 1.0)"

        self.alpha = alpha
        self.beta = beta
        self.x = initial_x
        self.delta_x = delta_x

    
    def find_min(self, method='sda'):
        if method == 'sda':
            self.x, a0 =  _steepest_descent_algorithm(self.x, 
                                                    self.objectiveFunction, 
                                                    self.interval, 
                                                    self.maxIter, 
                                                    self._line_search, 
                                                    self.xtol)
        #grad_x = copy(self.objectiveFunction.grad(self.x))
        #if (self.delta_x is None) or (self.objectiveFunction.best_x < np.inf):
        #    self.delta_x = copy(-grad_x)

        #for _ in range(self.maxIter):
        #    self.t = 1
        #    self._backtracking_line_search(grad_x)
        #    self.x = self.x + self.t * self.delta_x
        #    grad_x = copy(self.objectiveFunction.grad(self.x))
        #    if np.linalg.norm(grad_x) <= self.xtol:
        #        break
        #    self.delta_x = copy(-grad_x)
        return self.x


    def _line_search(self, x, dk):
        t = 1
        delta_x = dk
        grad_x = copy(-dk)
        f_x = self.objectiveFunction(x)
        if hasattr(f_x, '__iter__'):
            f_x = f_x.T @ f_x
        f_x_tdeltax = self.objectiveFunction(x + t * delta_x)
        if hasattr(f_x_tdeltax, '__iter__'):
            f_x_tdeltax = f_x_tdeltax.T @ f_x_tdeltax
        while f_x_tdeltax  >  f_x + self.alpha*t*(np.transpose(grad_x) @ delta_x):
            t = self.beta * t
            f_x_tdeltax = self.objectiveFunction(x + t * delta_x)
            if hasattr(f_x_tdeltax, '__iter__'):
                f_x_tdeltax = f_x_tdeltax.T @ f_x_tdeltax
            if t < 2*self.xtol:
                break
        return t, f_x_tdeltax