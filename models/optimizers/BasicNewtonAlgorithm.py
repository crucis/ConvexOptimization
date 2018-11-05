# Modify the Newton algorithm described in Algorithm 5.3 by incorporating
# Eq. (5.13) into the algorithm. Give a step-by-step description of the
# modified algorithm.

from .optimizer import optimizer
import autograd.numpy as np
from autograd import hessian
from copy import copy

def _basic_newton_algo(x_0, objectiveFunction, xtol=1-6, maxIter=1000):
    x_k = copy(x_0)
    for _ in range(maxIter):
        grad_x = objectiveFunction.grad(x_k)
        hessian_x = hessian(objectiveFunction)(x_k)