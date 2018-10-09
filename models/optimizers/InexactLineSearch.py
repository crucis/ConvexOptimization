from .optimizer import optimizer
from autograd import grad
import autograd.numpy as np


class InexactLineSearch(optimizer):
    def __init__(self, func, 
                    x_k = None, 
                    direction_vector = None, 
                    interval = [0, np.inf],
                    maxIter = 1e3, 
                    xtol = 1e-6, 
                    ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol, interval=interval)

    
    def find_min(self):
        pass
