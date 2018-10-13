from .optimizer import optimizer
import autograd.numpy as np
from copy import copy


class BacktrackingLineSearch(optimizer):
    def __init__(self, func, 
                        initial_x,
                        alpha = 0.3, 
                        beta = 0.5,
                        interval = [-100, 100],
                        maxIter = 1e6, 
                        xtol = 1e-6, 
                        ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol, interval=interval)  
        assert (alpha > 0) and (alpha < 0.5), "Alpha must be in the range (0, 0.5)"
        assert (beta > 0) and (beta < 1), "Alpha must be in the range (0, 1.0)"

        self.alpha = alpha
        self.beta = beta
        self.t = 1
        self.x = initial_x

    
    def find_min(self):
        grad_x = copy(self.objectiveFunction.grad(self.x))
        for _ in range(self.maxIter):
            self.t = 1
            self.delta_x = copy(-grad_x)
            self._backtracking_line_search(grad_x)
            self.x = self.x + self.t * self.delta_x
            grad_x = copy(self.objectiveFunction.grad(self.x))
            if np.linalg.norm(grad_x) <= self.xtol:
                break
        return self.x


    def _backtracking_line_search(self, grad_x):
            f_x = self.objectiveFunction(self.x)
            f_x_tdeltax = self.objectiveFunction(self.x + self.t * self.delta_x)
            while f_x_tdeltax  >  f_x + self.alpha*self.t*np.transpose(grad_x)*self.delta_x:
                self.t = self.beta * self.t
                f_x_tdeltax = self.objectiveFunction(self.x + self.t * self.delta_x)