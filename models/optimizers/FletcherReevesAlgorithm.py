from .optimizer import optimizer
import autograd.numpy as np
#from scipy import optimize

def _fletcher_reeves_algorithm(x, objectiveFunction, line_search, xtol=1e-6, maxIter=1e3):
    for _ in range(maxIter):
        grad_f = objectiveFunction.grad(x)
        direction_vector = -grad_f
        for i in range(1000):
            alpha, _ = line_search(x, direction_vector)
            x = x + alpha * direction_vector
            if np.linalg.norm(alpha * direction_vector) < xtol:
                return x
            if i == 100:
                break
            grad_f_new = objectiveFunction.grad(x)
            beta = np.dot(grad_f_new.T, grad_f_new)/np.dot(grad_f.T, grad_f)
            direction_vector = -grad_f_new + beta * direction_vector
            grad_f = grad_f_new
        

class FletcherReevesAlgorithm(optimizer):
    def __init__(self,
                 func,
                 x_0,
                 line_search_optimizer,
                 maxIter = 1e3,
                 xtol = 1e-6,
                 ftol = 1e-6):
        if hasattr(line_search_optimizer, '_line_search'):
            self.line_search = line_search_optimizer._line_search
        else:
            self.line_search = line_search_optimizer
        self.objectiveFunction = func
        self.x_k = x_0
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol)

    def find_min(self):
            return _fletcher_reeves_algorithm(self.x_k, self.objectiveFunction, self.line_search, self.xtol, self.maxIter)