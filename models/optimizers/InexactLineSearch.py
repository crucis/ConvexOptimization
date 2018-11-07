from .optimizer import optimizer
from .SteepestDescentAlgorithm import _steepest_descent_algorithm
from autograd import grad, hessian
import autograd.numpy as np
from copy import copy

import sys
sys.path.append('../..')
from functions import functionObj

class InexactLineSearch(optimizer):
    def __init__(self, func, 
                    x_0, 
                    d_0 = None,
                    rho = 0.1,
                    sigma = 0.1,
                    tau = 0.1,
                    chi = 9,
                    interval = [0, np.inf],
                    maxIter = 1e3, 
                    xtol = 1e-6, 
                    ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol, interval=interval)
        assert (rho >= 0) and (rho < 0.5), 'Rho must be in the range [0,0.5)'

        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.chi = chi
        if type(x_0) == list:
            self.x_k = np.array(x_0, dtype=np.float64)
        else:
            self.x_k = x_0
        self.alpha = self.interval[0]
        if type(d_0) == list:
            d_0 = np.array(d_0, dtype=np.float64)
        
        assert (d_0 is None) or (x_0.shape == d_0.shape), "x_0 and d_0 must have the same dimensions."
        
        self.direction_vector = d_0
        alphaL = self.interval[0]
        alphaU = self.interval[1]

    
    def find_min(self, method='sda'):
        if method == 'sda':
            self.x_k, a0 =  _steepest_descent_algorithm(self.x_k, 
                                                       self.objectiveFunction, 
                                                       self.interval, 
                                                       self.maxIter, 
                                                       self._line_search, 
                                                       self.xtol)
    
        return self.x_k


    def find_min_iterator(self):
        grad_x = copy(self.objectiveFunction.grad(self.x_k))
        alphaL = self.interval[0]
        alphaU = self.interval[1]
        if self.direction_vector is None:
            self.direction_vector = copy(-grad_x)
        for _ in range(self.maxIter):
            a0, _ = self._line_search()
            self.x_k = self.x_k + a0* self.direction_vector
            grad_x = copy(self.objectiveFunction.grad(self.x_k))
            if np.linalg.norm(grad_x) <= self.xtol:
                break
            self.direction_vector = copy(-grad_x)
            yield self.x_k


    def _compute_a0tilde(self, a0, f0, f_L, grad_f_L, alphaL):
        numerator = (a0 - alphaL)**2 * grad_f_L
        denominator = 2*(f_L - f0 + (a0 - alphaL) * grad_f_L)
        return alphaL + numerator/denominator

    def _line_search(self, xk, dk=None, H=None, g0=None):
        self.x_k = xk
        if self.direction_vector is None:
            assert dk is not None, "Initial direction must be input."
            self.direction_vector = dk
        alphaL = self.interval[0]
        alphaU = self.interval[1]
        # step 2
        f_L = self.objectiveFunction(self.x_k + alphaL * self.direction_vector)
        if hasattr(f_L, '__iter__'):
            f_L = f_L.T @ f_L
        if g0 is None:
            gL = self.objectiveFunction.grad(self.x_k + alphaL * self.direction_vector)
        else:
            x = self.x_k + alphaL * self.direction_vector
            jac_x = self.objectiveFunction.grad(x)
            f = self.objectiveFunction(x)
            gL = 2* jac_x.T @ f
        grad_f_L = (np.transpose(gL) @ self.direction_vector)

        # step 3
        if H is None:
            H = hessian(self.objectiveFunction)(self.x_k)
        if g0 is None:    
            g0 = self.objectiveFunction.grad(self.x_k)
        a0 = (np.transpose(g0) @ g0)/(np.transpose(g0) @ H @ g0)
        if a0 > alphaU:
            a0 = alphaU
        if a0 < alphaL:
            a0 = alphaL

        # step 4
        for _ in range(self.maxIter):
            f0 = self.objectiveFunction(self.x_k + a0 * self.direction_vector)
            if hasattr(f0, '__iter__'):
                f0 = f0.T @ f0
            # step 5 Interpolation
            if f0 > f_L + self.rho*(a0 - alphaL) * grad_f_L:
                if a0 < alphaU:
                    alphaU = a0
                a0tilde = self._compute_a0tilde(a0, f0, f_L, grad_f_L, alphaL)

                if a0tilde < alphaL + self.tau * (alphaU - alphaL):
                    a0tilde = alphaL + self.tau * (alphaU - alphaL)
                if a0tilde > alphaU - self.tau * (alphaU - alphaL):
                    a0tilde = alphaU - self.tau * (alphaU - alphaL)
                a0 = a0tilde

            else:
                # step 6
                if type(self.objectiveFunction) is functionObj:
                    gk = self.objectiveFunction.grad(self.x_k + a0 * self.direction_vector)
                else:
                    x = self.x_k + a0 * self.direction_vector
                    jac_x = self.objectiveFunction.grad(x)
                    f = self.objectiveFunction(x)
                    gk = 2* jac_x.T @ f

                self.grad_f0 = np.transpose(gk) @ self.direction_vector

                # step 7 Extrapolation
                if self.grad_f0 < self.sigma * grad_f_L:
                    self.delta_a0 = ((a0 - alphaL) * self.grad_f0 ) / (grad_f_L - self.grad_f0)
                    if self.delta_a0 < self.tau * (a0 - alphaL):
                        self.delta_a0 = self.tau * (a0 - alphaL)
                    if self.delta_a0 > self.chi * (a0 - alphaL):
                        self.delta_a0 = self.chi * (a0 - alphaL)
                    if self.delta_a0 <= 1e-4: #
                        self.delta_a0 = a0 #
                    a0tilde = a0 + self.delta_a0

                    # Prepare next iteration
                    alphaL = copy(a0)
                    a0 = copy(a0tilde)
                    f_L = copy(f0)
                    grad_f_L = copy(self.grad_f0)
            # step 8
                else:
                    break
        return a0, f0