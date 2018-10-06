from .optimizer import optimizer
import numpy as np
from copy import copy

class GoldenSectionSearch(optimizer):
    def find_min(self):
        self.I = self.interval[1] - self.interval[0]
        self.golden_ration = (1 + 5 ** 0.5)/2
        self.I = self.I/self.golden_ration

        self.x_a = self.interval[1] - self.I
        self.x_b = self.interval[0] + self.I

        self.fx_a = self.objectiveFunction(self.x_a)
        self.fx_b = self.objectiveFunction(self.x_b)

        for _ in range(1, self.maxIter):
            self.I = self.I/self.golden_ration
            self._update_interval()
            if (self.I < self.xtol) or (self.x_a > self.x_b):
                break
        if self.fx_a > self.fx_b:
            x = 0.5*(self.x_b + self.interval[1])
        elif self.fx_a == self.fx_b:
            x = 0.5*(self.x_a + self.x_b)
        else:
            x = 0.5*(self.interval[0] + self.x_a)
        return x    


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