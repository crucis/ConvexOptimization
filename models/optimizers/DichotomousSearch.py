from .optimizer import optimizer
import numpy as np

class DichotomousSearch(optimizer):
    def __init__(self, func, epsilon, maxIter = 1e18, interval = [-1e6, 1e6], xtol = 1e-6, ftol = 1e-6):
         self.epsilon = epsilon
         super().__init__(func, maxIter, interval, xtol, ftol)


    def find_min(self):
        for iteration in range(self.maxIter):
            self.x_1 = self._get_middle_interval()
            self.x_a = self.x_1 - self.epsilon
            self.x_b = self.x_1 + self.epsilon
            self.fx_a = self.objectiveFunction(self.x_a)
            self.fx_b = self.objectiveFunction(self.x_b)
            self._get_new_interval()
            if (self.interval[1] - self.interval[0]) <= self.xtol:
                break
        return self._get_middle_interval()


    def _get_middle_interval(self):
        return (self.interval[1] + self.interval[0])/2
    

    def _get_new_interval(self):
        if self.fx_a < self.fx_b:
            self.interval[1] = self.x_b
        elif self.fx_a > self.fx_b:
            self.interval[0] = self.x_a
        else:
            self.interval[1] = self.x_b
            self.interval[0] = self.x_a

