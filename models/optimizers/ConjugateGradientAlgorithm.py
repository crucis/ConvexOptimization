from .optimizer import optimizer
import autograd.numpy as np
from autograd import hessian

def _conjugate_gradient_algorithm(x, objectiveFunction, xtol, maxIter=1e3):
    grad_f = objectiveFunction.grad(x)
    direction_vector = -grad_f
    for _ in range(maxIter):
        hessian_f = hessian(objectiveFunction)(x)
        alpha = np.dot(grad_f.T, grad_f)/(np.dot(np.dot(direction_vector.T, hessian_f), direction_vector)+1e-9)
        x = x + alpha * direction_vector
        if np.linalg.norm(alpha*direction_vector) < xtol:
            break
        grad_f_new = objectiveFunction.grad(x)
        beta = np.dot(grad_f_new.T, grad_f_new)/np.dot(grad_f.T, grad_f)
        grad_f = grad_f_new
        direction_vector = -grad_f + beta*direction_vector
    return x

class ConjugateGradientAlgorithm(optimizer):
    def __init__(self,
                 func,
                 x_0,
                 maxIter = 1e3,
                 xtol = 1e-6,
                 ftol = 1e-6):
        self.objectiveFunction = func
        self.x_k = x_0
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol)

    def find_min(self):
            return _conjugate_gradient_algorithm(self.x_k, self.objectiveFunction, self.xtol, self.maxIter)