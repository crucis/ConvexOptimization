from .optimizer import optimizer
from autograd import grad
import autograd.numpy as np


class BacktrackingLineSearch(optimizer):
    def __init__(self, func, 
                        initial_x,
                        alpha = 0.01, 
                        beta = 0.1,
                        interval = [-100, 100],
                        maxIter = 1e3, 
                        xtol = 1e-6, 
                        ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol, interval=interval)  
        assert (alpha > 0) and (alpha < 0.5), "Alpha must be in the range (0, 0.5)"
        assert (beta > 0) and (beta < 1), "Alpha must be in the range (0, 1.0)"

        self.alpha = alpha
        self.beta = beta
        self.t = 1
        self.grad_func = grad(func)
        self.x = initial_x

    
    def find_min(self):
        grad_x = self.grad_func(self.x)
        for _ in range(self.maxIter):
            self.t = 1
            self.delta_x = grad_x
            f_x = self.objectiveFunction(self.x)
            while self.objectiveFunction(self.x + self.t*self.delta_x) \
                >  f_x + self.alpha*self.t*np.transpose(grad_x)*self.delta_x:
                self.t = self.beta * self.t

            self.x = self.x + self.t * self.delta_x
            grad_x = self.grad_func(self.x)
            if np.linalg.norm(grad_x) <= self.xtol:
                break
        return self.x
