from .optimizer import optimizer
import autograd.numpy as np
from autograd import grad
from copy import copy

class CubicInterpolation(optimizer):
    def __init__(self, func, initial_xs, maxIter = 1e3, xtol = 1e-6, ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol)
        self.initial_x = initial_xs
        self.grad_func = grad(self.objectiveFunction)

    
    def find_min(self):
        self.x = np.array(self.initial_x)

        self.x_oldmean = np.inf
        self.grad_f_1 = self.grad_func(self.x[0][np.newaxis])[0]
        self.f = np.array([self.objectiveFunction(y) for y in self.x])

        for _ in range(self.maxIter):
            self.beta = self._get_beta()
            self.gamma = self._get_gamma()
            self.theta = self._get_theta()
            self.phi = self._get_phi()
            self.a_3 = self._get_a3()
            self.a_2 = self._get_a2()
            self.a_1 = self._get_a1()

            self.x_mean = self._get_x_mean_minimizer()
            self.f_mean = self.objectiveFunction(self.x_mean)

            if np.abs(self.x_mean - self.x_oldmean) < self.xtol:
                break
            best_index = np.argmax(self.f)
            self.x_oldmean = self.x_mean
            self.x[best_index] = self.x_mean
            self.f[best_index] = self.f_mean
            if best_index == 0:
                self.grad_f_1 = self.grad_func(self.x_mean[np.newaxis])[0]

        return self.x_mean


    def _get_beta(self):
        numerator = self.f[1] - self.f[0] + self.grad_f_1*(self.x[0] - self.x[1])
        denominator = (self.x[0] - self.x[1])**2
        return numerator/denominator
    

    def _get_gamma(self):
        numerator = self.f[2] -self.f[0] + self.grad_f_1*(self.x[0] - self.x[2])
        denominator = (self.x[0] - self.x[2])**2
        return numerator/denominator


    def _get_theta(self):
        numerator = 2*self.x[0]**2 - self.x[1]*(self.x[0] + self.x[1])
        denominator = self.x[0] - self.x[1]
        return numerator/denominator


    def _get_phi(self):
        numerator = 2*self.x[0]**2 - self.x[2]*(self.x[0] + self.x[2])
        denominator = self.x[0] - self.x[2]
        return numerator/denominator


    def _get_a3(self):
        return (self.beta - self.gamma)/(self.theta - self.phi)


    def _get_a2(self):
        return self.beta - self.theta*self.a_3


    def _get_a1(self):
        return self.grad_f_1 - 2*self.a_2*self.x[0] - 3*self.a_3*(self.x[0]**2)

    
    def _get_extremum_points(self):
        positive_result = (1/(3*self.a_3)) * (-self.a_2 + np.sqrt(self.a_2**2 - 3*self.a_1*self.a_3))
        negative_result = (1/(3*self.a_3)) * (-self.a_2 + np.sqrt(self.a_2**2 + 3*self.a_1*self.a_3))
        return (positive_result, negative_result)

    def _get_x_mean_minimizer(self):
        first_result, second_result = self._get_extremum_points()
        if first_result > - (self.a_2/(3*self.a_3)):
            return first_result
        else:
            return second_result