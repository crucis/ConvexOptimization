from .optimizer import optimizer
import numpy as np
from copy import copy


class DaviesSwannCampey(optimizer):
    def __init__(self, func, 
                        x_0 = None, 
                        initial_increment = None, 
                        scaling_constant = 0.1,
                        interval = [-100, 100],
                        maxIter = 1e3, 
                        xtol = 1e-6, 
                        ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol, interval=interval)
        self.x = [np.random.uniform(low = self.interval[0], high = self.interval[1])] if x_0 == None else [x_0]
        self.increment = 0.1*np.abs(self.x[0]) if initial_increment == None else initial_increment
        assert (scaling_constant > 0) and (scaling_constant < 1), "Scaling constanst must be in the range 0 to 1."
        self.scaling_constant = scaling_constant


    def find_min(self):
        for _ in range(1, self.maxIter):
            step7 = False
            # step 2
            self.x = [self.x[0]]
            self.x_left = self.x[0] - self.increment
            self.x += [self.x[0] + self.increment]
            self.f = [self.objectiveFunction(self.x[0])]
            self.f += [self.objectiveFunction(self.x[1])]
            self.n = len(self.f) - 1

            # step 3
            if self.f[0] > self.f[1]:
                self.p = 1
                # step 4
                self._compute_new_f()
            else: # f[0] <= f[1]
                self.f_left = self.objectiveFunction(self.x_left)
                if self.f_left < self.f[0]:
                    self.p = -1 
                    # step 4
                    self._compute_new_f()
                else: # f[-1] >= f[0] <= f[1]
                    # step 7
                    aux = self._compute_new_x0_based_on_x0()
                    if aux == False:
                        break
                    self.x[0] = aux
                    if self.increment <= self.xtol:
                        break
                    else:
                        self.increment = self.scaling_constant*self.increment
                        step7 = True
            if step7 == False:
                # step 5
                self.x_m = self.x[self.n - 1] + 2**(self.n - 2)*self.p*self.increment
                self.f_m = self.objectiveFunction(self.x_m)
                # step 6
                if self.f_m >= self.f[self.n-1]:
                    aux = self._compute_new_x0_based_on_xn()
                    if aux == False:
                        break
                    self.x[0] = aux
                else: # f_m < f[n-1]
                    aux = self._compute_new_x0_based_on_xm()
                    if aux == False:
                        break
                    self.x[0] = aux
                if 2**(self.n - 2)*self.increment <= self.xtol:
                    break
                else:
                    self.increment = self.scaling_constant*self.increment
        # step 8
        return self.x[0]


    # step 4
    def _compute_new_f(self):
        while True:
            self.x += [self.x[self.n] + 2**(self.n)*self.p*self.increment]
            self.f += [self.objectiveFunction(self.x[self.n + 1])]
            self.n = self.n + 1
            if self.f[self.n] > self.f[self.n-1]:
                break

            

    # step 7
    def _compute_new_x0_based_on_x0(self):
        numerator = self.increment*(self.f_left - self.f[1])
        denominator = 2*(self.f_left - 2*self.f[0] + self.f[1])
        if np.isclose(denominator, 0, atol=self.xtol):
            return False
        else:
            return self.x[0] + numerator/denominator


    # step 6
    def _compute_new_x0_based_on_xn(self):
        numerator = 2**(self.n-2)*self.p*self.increment*(self.f[self.n-2] - self.f_m)
        denominator = 2*(self.f[self.n-2] - 2*self.f[self.n-1] + self.f_m)
        if np.isclose(denominator, 0, atol=self.xtol):
            return False
        else:
            return self.x[self.n-1] + numerator/denominator


    # step 6
    def _compute_new_x0_based_on_xm(self):
        numerator = 2**(self.n-2)*self.p*self.increment*(self.f[self.n-1] - self.f[self.n])
        denominator = 2*(self.f[self.n-1] - 2*self.f_m + self.f[self.n])
        if np.isclose(denominator, 0, atol=self.xtol):
            return False
        else:            
            return self.x_m + numerator/denominator