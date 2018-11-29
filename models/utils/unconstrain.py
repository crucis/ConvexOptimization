from autograd import numpy as np


class UnconstrainProblem:
    def __init__(self, func, x_0, opt, mu=20, line_search_optimizer=None, maxIter = 1e3, xtol=1e-6):
        self.objectiveFunction = func
        self.x_0 = x_0
        self.optimizer = opt
        self.maxIter = maxIter
        self.xtol = xtol
        self.mu = mu
        self.line_search = line_search_optimizer
        self.params = { 'xtol': self.xtol,
                        'maxIter': self.maxIter,
                        }
        if self.line_search is not None:
            self.params['line_search_optimizer'] = self.line_search

    def find_min(self):
        self.objectiveFunction.smooth_log_constant = 1
        x0 = self.x_0
        for _ in range(int(self.maxIter)):
            # centering Step
            opt = self.optimizer(self.objectiveFunction, x0, **self.params)
            opt.find_min()
        
            # Update x_0
            if self.objectiveFunction._has_eqc:
                x0 = self.objectiveFunction.best_z
            else:
                x0 = self.objectiveFunction.best_x
            x0 = np.array(x0, dtype=np.float64)
            # Break if m/t < epsilon
            if self.objectiveFunction.niq/self.objectiveFunction.smooth_log_constant < self.xtol:
                break
            # Update t
            self.objectiveFunction.smooth_log_constant *= self.mu
        return self.objectiveFunction.best_x