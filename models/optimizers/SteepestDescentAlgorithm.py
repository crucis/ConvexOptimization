from .optimizer import optimizer
import autograd.numpy as np
from copy import copy


def steepest_descent_algorithm(x_0, objectiveFunction, interval, maxIter, line_search=None, xtol=1e-6):
    x_k = copy(x_0)
    grad_x = copy(objectiveFunction.grad(x_k))
    alpha_L = interval[0]
    alpha_U = interval[1]
    direction_vector = copy(-grad_x)
    if line_search is None:
        a0 = 1
        f0 = objectiveFunction(x_k)
    for _ in range(maxIter):
        if line_search is not None:
            a0, f0  = line_search(direction_vector)
        else:
            f_hat = objectiveFunction(x_k - a0*grad_x)
            a0 = (np.dot(grad_x, grad_x) * (a0 **2)) / (2 * (f_hat - f0 + a0 * np.dot(grad_x, grad_x)))
        x_k = x_k + a0* direction_vector
        if line_search is None:
            f0 = objectiveFunction(x_k)
        grad_x = copy(objectiveFunction.grad(x_k))
        if np.linalg.norm(a0 * direction_vector) < xtol:
            break
        direction_vector = copy(-grad_x)
        #direction_vector = copy(-grad_x/np.linalg.norm(grad_x))
    return x_k, a0

