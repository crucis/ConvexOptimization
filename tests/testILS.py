from functions import functionObj
from models.optimizers import InexactLineSearch
import autograd.numpy as np
from autograd import grad
from copy import copy

f_x = lambda x: x**2 - 4*x + 4
f_x_obj = functionObj(f_x)

x_0 = np.random.randn(1)*100
d_0 = copy(grad(f_x_obj)(x_0))
opt = InexactLineSearch(f_x_obj, x_0, d_0)

x_min = opt.find_min()
print('X: %.9f \nF_x: %.9f'%(x_min, f_x_obj(x_min)))
print('Function evals: %d'%(f_x_obj.fevals - 1))
