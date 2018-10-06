from .optimizer import optimizer
import autograd.numpy as np
from copy import copy

class QuadraticInterpolationSearch(optimizer):
    def find_min(self):
        self.x_oldmean = np.inf
        self.x_1 = copy(self.interval[0])
        self.x_3 = copy(self.interval[1])
        self.x_2 = 0.5 * (self.x_1 + self.x_3)

        self.f_1 = self.objectiveFunction(self.x_1)
        self.f_2 = self.objectiveFunction(self.x_2)
        self.f_3 = self.objectiveFunction(self.x_3)

        for _ in range(self.maxIter):
            self.x_mean = self._get_new_mean()
            self.f_mean = self.objectiveFunction(self.x_mean)
            if np.abs(self.x_mean - self.x_oldmean) < self.xtol:
                break
            self._get_new_interval()
            self.x_oldmean = copy(self.x_mean)
        return self.x_mean


    def _get_new_mean(self):
        numerator = (self.x_2**2 - self.x_3**2)*self.f_1 + (self.x_3**2 - self.x_1**2)*self.f_2 + (self.x_1**2 - self.x_2**2)*self.f_3
        denominator = 2 * ( (self.x_2 - self.x_3)*self.f_1 + (self.x_3 - self.x_1)*self.f_2 + (self.x_1 - self.x_2)*self.f_3)
        return numerator/denominator

    def _get_new_interval(self):
        if (self.x_1 < self.x_mean) and (self.x_mean < self.x_2):
            if self.f_mean <= self.f_2:
                self.x_3 = copy(self.x_2)
                self.f_3 = copy(self.f_2)
                self.x_2 = copy(self.x_mean)
                self.f_2 = copy(self.f_mean)
            else:
                self.x_1 = copy(self.x_mean)
                self.f_1 = copy(self.f_mean)
        elif (self.x_2 < self.x_mean) and (self.x_mean < self.x_3):
            if self.f_mean <= self.f_2:
                self.x_1 = self.x_2
                self.f_1 = self.f_2
                self.x_2 = self.x_mean
                self.f_2 = self.f_mean
            else:
                self.x_3 = self.x_mean
                self.f_3 = self.f_mean