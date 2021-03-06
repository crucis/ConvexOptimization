from .optimizer import optimizer
from .SteepestDescentAlgorithm import _steepest_descent_algorithm
import autograd.numpy as np
from copy import copy



class BacktrackingLineSearch(optimizer):
    def __init__(self, func, 
                        initial_x,
                        delta_x = None,
                        alpha = 0.03, 
                        beta = 0.7,
                        interval = [-100, 100],
                        maxIter = 200, 
                        xtol = 1e-6, 
                        ftol = 1e-6):
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol, interval=interval)  
        assert (alpha > 0) and (alpha < 0.5), "Alpha must be in the range (0, 0.5)"
        assert (beta > 0) and (beta < 1), "Alpha must be in the range (0, 1.0)"

        self.alpha = alpha
        self.beta = beta
        self.x = initial_x
        self.delta_x = delta_x

    
    def find_min(self, method='sda'):
        if method == 'sda':
            self.x, a0 =  _steepest_descent_algorithm(self.x, 
                                                    self.objectiveFunction, 
                                                    self.interval, 
                                                    self.maxIter, 
                                                    self._line_search, 
                                                    self.xtol)

        return self.x


    def _line_search(self, x, dk):
        t = 1
        delta_x = dk
        grad_x = copy(-dk)
        f_x = self.objectiveFunction(x)
        if np.shape(f_x) is not ():
            print('opa')
            f_x = np.dot(f_x.T,f_x)
        f_x_tdeltax = self.objectiveFunction(x + t * delta_x)

        if np.shape(f_x_tdeltax) is not ():
            f_x_tdeltax = np.dot(f_x_tdeltax.T, f_x_tdeltax)

        while ~np.isclose(f_x_tdeltax, f_x + self.alpha*t*(np.transpose(grad_x) @ delta_x), rtol=1e-3):
            t = self.beta * t
            f_x_tdeltax = self.objectiveFunction(x + t * delta_x)
            if np.shape(f_x_tdeltax) is not ():
                f_x_tdeltax = np.dot(f_x_tdeltax.T, f_x_tdeltax)
            if t < 2*self.xtol:
                break
        return t, f_x_tdeltax