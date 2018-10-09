from .optimizer import optimizer
import numpy as np
from copy import copy

class FibonacciSearch(optimizer):
    def find_min(self):
        self.fibArray = self._fibonacci(self.maxIter)
        self.I = self.interval[1] - self.interval[0]
        self.I = (self.fibArray[-2]/self.fibArray[-1])*self.I
        
        self.x_a = self.interval[1] - self.I
        self.x_b = self.interval[0] + self.I

        self.fx_a = self.objectiveFunction(self.x_a)
        self.fx_b = self.objectiveFunction(self.x_b)

        for iteration in range(1, self.maxIter):
            self.I = self._get_intervalSize(iteration)
            self._update_interval()
            if iteration == self.maxIter - 2:
                break
            elif self.x_a > self.x_b:
                break
            elif (self.x_b - self.x_a) <= self.xtol:
                break
        return self.x_a

    def _get_intervalSize(self, k):
        old_I = copy(self.I)
        new_I = (self.fibArray[-2 - k]/self.fibArray[-1 - k]) * old_I
        return new_I

    def _update_interval(self):
        if self.fx_a >= self.fx_b:
            self.interval[0] = copy(self.x_a)
            self.x_a = copy(self.x_b)
            self.x_b = self.interval[0] + self.I
            self.fx_a = copy(self.fx_b)
            self.fx_b = self.objectiveFunction(self.x_b)
        else:
            self.interval[1] = copy(self.x_b)
            old_x_a = copy(self.x_a)
            self.x_a = self.interval[1] - self.I
            self.x_b = old_x_a
            self.fx_b = copy(self.fx_a)
            self.fx_a = self.objectiveFunction(self.x_a)


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