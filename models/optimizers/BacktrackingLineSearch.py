from .optimizer import optimizer
from autograd import grad
import autograd.numpy as np


class BacktrackingLineSearch(optimizer):
    def __init__(self, func, 
                        initial_x,
                        delta_x,
                        alpha = 0.01, 
                        beta = 0.5,
                        interval = [-100, 100],
                        maxIter = 1e3, 
                        xtol = 1e-6, 
                        ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol, interval=interval)  
        assert (alpha > 0) and (alpha < 0.5), "Alpha must be in the range (0, 0.5)"
        assert (beta > 0) and (beta < 1), "Alpha must be in the range (0, 1.0)"

        self.alpha = alpha
        self.beta = beta
        self.delta_x = delta_x
        self.t = 10
        self.grad_func = grad(func)
        self.x = initial_x

    
    def find_min(self):
        for _ in range(self.maxIter):
            if self.objectiveFunction(self.x + self.t*self.delta_x) \
                <= self.objectiveFunction(self.x) + \
                    self.alpha*self.t*np.transpose(self.grad_func(self.x))*self.delta_x:
                break
            self.t = self.beta*self.t

        return self.t
