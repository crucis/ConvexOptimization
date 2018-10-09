from .optimizer import optimizer
from autograd import grad
import autograd.numpy as np


class InexactLineSearch(optimizer):
    def __init__(self, func, 
                    x_k = None, 
                    direction_vector = None, 
                    rho = 0.25,
                    alpha = 0.1,
                    sigma = 0.1,
                    tau = 0.1,
                    qsi = 0.1,
                    maxIter = 1e3, 
                    xtol = 1e-6, 
                    ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol, interval=[0, np.inf])
        assert (rho >= 0) and (rho < 0.5), 'Rho must be in the range [0,0.5)'
        self.direction_vector = direction_vector
        self.rho = rho
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau
        self.qsi = qsi
        self.x_k = x_k
        self.alpha = self.interval[0]
        self.grad_f_alpha = grad(self.f_alpha)

    
    def find_min(self):
        # step 2
        self.f_L = self.objectiveFunction(self.x_k + self.interval[0]*self.direction_vector)
        self.grad_f_L = self.grad_f_alpha(self.x_k, self.interval[0])        

        # step 3

        pass

    def f_alpha(self, x, alpha):
        return grad(self.objectiveFunction(x + alpha*self.direction_vector))