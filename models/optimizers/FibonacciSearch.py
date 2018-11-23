from .optimizer import optimizer
import numpy as np
from copy import copy

class FibonacciSearch(optimizer):
    def _line_search(self, x=None, dk=None):
        if x is None:
            self.x = 0
            self.dk = 1
        else:
            self.x = x
            self.dk = 1
        if dk is not None:
            self.dk = dk
        self.internal_interval = copy(self.interval)
        self.fibArray = self._fibonacci(self.maxIter)
        self.I = self.internal_interval[1] - self.internal_interval[0]
        self.I = (self.fibArray[-2]/self.fibArray[-1])*self.I
        
        self.alpha_a = self.internal_interval[1] - self.I
        self.alpha_b = self.internal_interval[0] + self.I
        self.fx_a = self.objectiveFunction(self.x + self.alpha_a*self.dk)
        self.fx_b = self.objectiveFunction(self.x + self.alpha_b*self.dk)

        for iteration in range(1, self.maxIter):
            self.I = self._get_intervalSize(iteration)
            self._update_interval()
            if iteration == self.maxIter - 2:
                break
            elif self.alpha_a > self.alpha_b:
                break
            elif (self.alpha_b - self.alpha_a) <= self.xtol:
                break
        return self.alpha_a, self.fx_a

    def _get_intervalSize(self, k):
        old_I = copy(self.I)
        new_I = (self.fibArray[-2 - k]/self.fibArray[-1 - k]) * old_I
        return new_I

    def _update_interval(self):
        if self.fx_a >= self.fx_b:
            self.internal_interval[0] = copy(self.alpha_a)
            self.alpha_a = copy(self.alpha_b)
            self.alpha_b = self.internal_interval[0] + self.I
            self.fx_a = copy(self.fx_b)
            self.fx_b = self.objectiveFunction(self.x + self.alpha_b*self.dk)
        else:
            self.internal_interval[1] = copy(self.alpha_b)
            old_x_a = copy(self.alpha_a)
            self.alpha_a = self.internal_interval[1] - self.I
            self.alpha_b = old_x_a
            self.fx_b = copy(self.fx_a)
            self.fx_a = self.objectiveFunction(self.x + self.alpha_a*self.dk)


    def _fibonacci(self, n):
        fib_list = []
        x_new = 1
        x_old = 1
        fib_list += [x_old]
        if n == 1:
            return fib_list
        fib_list += [x_new]
        if n == 2:
            return fib_list
        for _ in range(n - 2):
            x_aux = x_new
            x_new = x_new + x_old
            if x_new > 1e30:
                self.maxIter = len(fib_list)
                break
            fib_list += [x_new]
            x_old = x_aux
        return fib_list