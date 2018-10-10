from .optimizer import optimizer
from autograd import grad
import autograd.numpy as np
from copy import copy


class InexactLineSearch(optimizer):
    def __init__(self, func, 
                    x_k, 
                    direction_k,
                    rho = 0.1,
                    alpha = 0.1,
                    sigma = 0.1,
                    tau = 0.1,
                    chi = 0.75,
                    mhat = 400,
                    maxIter = 1e3, 
                    xtol = 1e-6, 
                    ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol, interval=[0, np.inf])
        assert (rho >= 0) and (rho < 0.5), 'Rho must be in the range [0,0.5)'
        self.rho = rho
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau
        self.chi = chi
        self.x_k = x_k
        self.alpha = self.interval[0]
        self.grad_f_alpha = grad(self.f_alpha)
        self.grad_f = grad(self.objectiveFunction)
        self.g_k = self.grad_f(self.x_k)
        self.f0 = self.objectiveFunction(self.x_k)
        self.direction_vector = direction_k
        self.delta_f0 = copy(self.f0)

    
    def find_min(self):
        # step 2
        self.f_L = self.objectiveFunction(self.x_k + self.interval[0]*self.direction_vector)
        self.grad_f_L = self.grad_f_alpha(self.x_k, self.interval[0])        
        #self.grad_f_L = np.sum(self.g_k * self.direction_vector)
        if np.abs(self.grad_f_L) > self.xtol:
            self.a0 = -2*self.delta_f0 / self.grad_f_L
        else:
            self.a0 = 1
        if (self.a0 <= xtol) or (self.a0 > 1):
            self.a0 = 1
        
        # step 3
        while True:
            self.delta_k = self.a0 * self.direction_vector
            self.f0 = self.objectiveFunction(self.x_k + self.delta_k)
            # step 5 interpolation
            if (self.f0 > (self.f_L + self.rho * (self.a0 - self.interval[0]) * self.grad_f_L)) and (np.abs(self.f_L - self.f0) > self.xtol):
                if (self.a0 < self.interval[1]):
                    self.interval[1] = self.a0
                a0hat = self.interval[0] + ((self.a0 - self.interval[0])**2 * self.direction_vector)/(2*(self.f_L - self.f0 + (self.a0 - self.interval[0] & self.grad_f_L)))
                a0Lhat = self.interval[0] + self.tau * (self.interval[1] - self.interval[0])
                if (a0hat < a0Lhat):
                    a0hat = a0Lhat
                a0Uhat = self.interval[1] - self.tau * (self.interval[1] - self.interval[0])
                if (a0hat > a0Uhat):
                    a0hat = a0Uhat
                a0 = a0hat
            # step 6
            else:
                df0 = np.sum(self.grad_f(self.x_k + a0 * self.delta_k))
                if (df0 < self.sigma * self.grad_f_L) and (np.abs(self.f_L - self.f0) > self.ftol) and (self.f_L != self.f0):
                    deltaa0 = (a0 - self.interval[0]) * df0 / (self.grad_f_L - df0)
                    if (deltaa0 <= 0):
                        a0hat = 2*a0
                    else:
                        a0hat = a0 + deltaa0
                    a0Uhat = a0 + self.chi * (self.interval[1] - self.interval[0])
                    if (a0hat > a0Uhat):
                        a0hat = a0Uhat
                    self.interval[0] = a0
                    a0 = a0hat
                    self.f_L = self.f0
                    self.grad_f_L = df0
                else:
                    break
            return a0


        pass

    def f_alpha(self, x, alpha):
        return grad(self.objectiveFunction(x + alpha*self.direction_vector))