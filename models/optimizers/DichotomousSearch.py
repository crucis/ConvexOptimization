from optimizer import optimizer
import autograd.numpy as np

class DichotomousSearch(optimizer):
    def find_min(self):
        for iteration in self.maxIter:
            self.x_1 = self._get_middle_interval()
            self.x_a = self.x_1 - self.xtol/2
            self.x_b = self.x_1 + self.xtol/2
            self.fx_a = self.objectiveFunction(self.x_a)
            self.fx_b = self.objectiveFunction(self.x_b)
            self._get_new_interval()
            if self.interval[1] - self.interval[0] == self.ftol:
                break


    def _get_middle_interval(self):
        return (self.interval[1] - self.interval[0])/2
    
    def _get_new_interval(self):
        if self.fx_a < self.fx_b:
            self.interval[1] = self.x_b
        elif self.fx_a > self.fx_b:
            self.interval[0] = self.x_a
        else:
            self.interval[1] = self.x_b
            self.interval[0] = self.x_a

