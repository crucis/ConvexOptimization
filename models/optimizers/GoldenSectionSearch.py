from .optimizer import optimizer
import numpy as np
from copy import copy

class GoldenSectionSearch(optimizer):
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
        self.I = self.internal_interval[1] - self.internal_interval[0]
        self.golden_ratio = (1 + 5 ** 0.5)/2
        self.I = self.I/self.golden_ratio

        self.alpha_a = self.internal_interval[1] - self.I
        self.alpha_b = self.internal_interval[0] + self.I
        self.fx_a = self.objectiveFunction(self.x + self.alpha_a*self.dk)
        self.fx_b = self.objectiveFunction(self.x + self.alpha_b*self.dk)

        for _ in range(1, self.maxIter):
            self.I = self.I/self.golden_ratio
            self._update_interval()
            if (self.I < self.xtol) or (self.alpha_a > self.alpha_b):
                break
        if self.fx_a > self.fx_b:
            alpha = 0.5*(self.alpha_b + self.internal_interval[1])
            f = self.fx_b
        elif self.fx_a == self.fx_b:
            alpha = 0.5*(self.alpha_a + self.alpha_b)
            f = self.fx_a
        else:
            alpha = 0.5*(self.internal_interval[0] + self.alpha_a)
            f = self.fx_a
        return alpha, f


    def _update_interval(self):
        if self.fx_a >= self.fx_b:
            self.internal_interval[0] = copy(self.alpha_a)
            self.alpha_a = copy(self.alpha_b)
            self.alpha_b = self.internal_interval[0] + self.I
            self.fx_a = copy(self.fx_b)
            self.fx_b = self.objectiveFunction(self.x + self.alpha_b*self.dk)
        else:
            self.internal_interval[1] = copy(self.alpha_b)
            self.alpha_b = copy(self.alpha_a)
            self.alpha_a = self.internal_interval[1] - self.I
            self.fx_b = copy(self.fx_a)
            self.fx_a = self.objectiveFunction(self.x + self.alpha_a*self.dk)