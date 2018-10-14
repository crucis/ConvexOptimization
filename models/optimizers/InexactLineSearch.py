from .optimizer import optimizer
from autograd import grad
import autograd.numpy as np
from copy import copy


class InexactLineSearch(optimizer):
    def __init__(self, func, 
                    x_k, 
                    d_0 = None,
                    rho = 0.1,
                    sigma = 0.7,
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
        if type(x_k) == list:
            self.x_k = np.array(x_k, dtype=np.float64)
        else:
            self.x_k = x_k
        self.alpha = self.interval[0]
        if type(d_0) == list:
            d_0 = np.array(d_0, dtype=np.float64)
        self.direction_vector = d_0

    
    def find_min(self):
        grad_x = copy(self.objectiveFunction.grad(self.x_k))
        self.alpha_L = self.interval[0]
        self.alpha_U = self.interval[1]
        if self.direction_vector is None:
            self.direction_vector = copy(-grad_x)
        for _ in range(self.maxIter):
            self.a0 = 1
            self._line_search()
            self.x_k = self.x_k + self.a0* self.direction_vector
            grad_x = copy(self.objectiveFunction.grad(self.x_k))
            if np.linalg.norm(grad_x) <= self.xtol:
                break
            self.direction_vector = copy(-grad_x)
        return self.x_k


    def find_min_iterator(self):
        grad_x = copy(self.objectiveFunction.grad(self.x_k))
        self.alpha_L = self.interval[0]
        self.alpha_U = self.interval[1]
        if self.direction_vector is None:
            self.direction_vector = copy(-grad_x)
        for _ in range(self.maxIter):
            self.a0 = 1
            self._line_search()
            self.x_k = self.x_k + self.a0* self.direction_vector
            grad_x = copy(self.objectiveFunction.grad(self.x_k))
            if np.linalg.norm(grad_x) <= self.xtol:
                break
            self.direction_vector = copy(-grad_x)
            yield self.x_k, self.a0, self.direction_vector


    def _compute_a0tilde(self):
        numerator = (self.a0 - self.alpha_L)**2 * self.norm_grad_f_L
        denominator = 2*(self.f_L - self.f0 + (self.a0 - self.alpha_L) * self.norm_grad_f_L)
        return self.alpha_L + numerator/denominator

    def _line_search(self):
        # step 2
        self.f0 = self.objectiveFunction(self.x_k)
        delta_f0 = copy(self.f0)
        self.f_L = self.objectiveFunction(self.x_k + self.alpha_L * self.direction_vector)
        gk = self.objectiveFunction.grad(self.x_k + self.alpha * self.direction_vector)
        self.grad_f_L = gk * self.direction_vector if not hasattr(gk, '__iter__') \
                                                    else (np.transpose(gk[:, np.newaxis]) * self.direction_vector).flatten()
        self.norm_grad_f_L = np.linalg.norm(self.grad_f_L)

        # step 3
        if self.norm_grad_f_L > self.xtol:
            self.a0 = -2*delta_f0 / self.norm_grad_f_L
        else:
            self.a0 = 1
        if (self.a0 <= self.xtol) or (self.a0 > 1):
            self.a0 = 1
        
        # step 4
        for _ in range(self.maxIter):
            self.f0 = self.objectiveFunction(self.x_k + self.a0 * self.direction_vector)
            entered_step5 = False
            entered_step7 = False

            # step 5
            if self.f0 > self.f_L + self.rho*(self.a0 - self.alpha_L) * self.norm_grad_f_L:
                if self.a0 < self.alpha_U:
                    self.alpha_U = self.a0
                self.a0tilde = self._compute_a0tilde()
                if self.a0tilde < self.alpha_L + self.tau * (self.alpha_U - self.alpha_L):
                    self.a0tilde = self.alpha_L + self.tau * (self.alpha_U - self.alpha_L)
                if self.a0tilde > self.alpha_U - self.tau * (self.alpha_U - self.alpha_L):
                    self.a0tilde = self.alpha_U - self.tau * (self.alpha_U - self.alpha_L)
                self.a0 = self.a0tilde
                entered_step5 = True

            if not entered_step5:
                # step 6
                gk = self.objectiveFunction.grad(self.x_k + self.a0 * self.direction_vector)
                self.grad_f0 = gk * self.direction_vector if not hasattr(gk, '__iter__') \
                                                            else (np.transpose(gk[:, np.newaxis]) * self.direction_vector).flatten()
                self.norm_grad_f0 = np.linalg.norm(self.grad_f0)
                # step 7
                if self.norm_grad_f0 < self.sigma * self.norm_grad_f_L:
                    self.delta_a0 = ((self.a0 - self.alpha_L) * self.norm_grad_f0 ) / (self.norm_grad_f_L - self.norm_grad_f0)
                    if self.delta_a0 < self.tau * (self.a0 - self.alpha_L):
                        self.delta_a0 = self.tau * (self.a0 - self.alpha_L)
                    if self.delta_a0 > self.chi * (self.a0 - self.alpha_L):
                        self.delta_a0 = self.chi * (self.a0 - self.alpha_L)
                    self.a0tilde = self.a0 + self.delta_a0
                    self.alpha_L = self.a0
                    self.a0 = self.a0tilde
                    self.f_L = self.f0
                    self.grad_f_L = self.grad_f0
                    self.norm_grad_f_L = self.norm_grad_f0
                    entered_step7 = True
            if (not entered_step5) and (not entered_step7):
                break
        return self.a0, self.f0