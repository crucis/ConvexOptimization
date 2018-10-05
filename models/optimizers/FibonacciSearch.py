from .optimizer import optimizer
import numpy as np
from copy import copy

class FibonacciSearch(optimizer):
    def __init__(self, func, epsilon, maxIter = 1e18, interval = [-1e6, 1e6], xtol = 1e-6, ftol = 1e-6):
         self.epsilon = epsilon
         super().__init__(func, maxIter, interval, xtol, ftol)
         self.fibArray = self._fibonacci(self.maxIter)


    def find_min(self):
        self.I = self.interval[1] - self.interval[0]
        self.I = (self.fibArray[-2]/self.fibArray[-1])*self.I
        
        self.x_a = self.interval[1] - self.I
        self.x_b = self.interval[0] - self.I

        self.fx_a = self.objectiveFunction(self.x_a)
        self.fx_b = self.objectiveFunction(self.x_b)

        for iteration in range(1, self.maxIter):
            self.I = self._get_intervalSize(iteration)
            self._get_new_interval()
            if iteration == self.maxIter - 2:
                break
            elif self.x_a > self.x_b:
                break

        return self.x_a

    def _get_intervalSize(self, k):
        old_I = self.I
        new_I = (self.fibArray[-2 - k]/self.fibArray[-1 - k]) * old_I
        return new_I

    def _get_new_interval(self):
        if self.fx_a >= self.fx_b:
            self.interval[0] = self.x_a
            old_x_a = copy(self.x_a)
            self.x_a = self.x_b
            self.x_b = old_x_a + self.I
            self.fx_a = self.fx_b
            self.fx_b = self.objectiveFunction(self.x_b)
        else:
            self.interval[1] = self.x_b
            old_x_a = self.x_a
            self.x_a = self.interval[1] - self.I
            self.x_b = old_x_a
            self.fx_b = self.fx_a
            self.fx_a = self.objectiveFunction(self.x_a)


    def _fibonacci(self, n):
        fib_list = []
        x_new = 1
        x_old = 1
        fib_list += [x_old]
        fib_list += [x_new]
        for i in range(n):
            x_aux = x_new
            x_new = x_new + x_old
            if np.isnan(x_new):
                self.maxIter = i - 1
                break
            fib_list += [x_new]
            x_old = x_aux
        return fib_list